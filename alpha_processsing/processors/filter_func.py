from tkinter import X
from typing import Tuple
import numpy as np
import pandas as pd
import torch
import pywt
from utils.data_utils import interpolator_arr
import scipy.signal as signal
from scipy.interpolate import interp1d
from scipy import fft, sparse
from scipy.sparse.linalg import spsolve
from statsmodels.tsa.filters import hpfilter
# Import pykalman only when needed to avoid ImportError if not installed
from joblib import Parallel, delayed
from pykalman import KalmanFilter


def _validate_numpy_input(arr):
    """Validate input as 1D or 2D NumPy array"""
    arr = np.asarray(arr)
    if arr.ndim not in (1, 2):
        raise ValueError("Input must be 1D or 2D NumPy array")
    return arr

def _validate_torch_input(arr, device='cpu'):
    """Validate input as 1D or 2D PyTorch tensor"""
    if isinstance(arr, np.ndarray):
        arr = torch.from_numpy(arr).double()
    elif not isinstance(arr, torch.Tensor):
        raise ValueError("Input must be NumPy array or PyTorch tensor")
    if arr.ndim not in (1, 2):
        raise ValueError("Input must be 1D or 2D tensor")
    return arr.to(device)


def _to_numpy(arr):
    """Convert to NumPy array"""
    return arr.cpu().numpy() if isinstance(arr, torch.Tensor) else arr

def _handle_all_nan(arr, shape, backend="numpy", device="cpu"):
    """Return NaN array for all-NaN input"""
    if backend == "torch":
        return torch.full(shape, float('nan'), device=device)
    return np.full(shape, np.nan)



def _dwt_ca_1d(x, wavelet, level):
    """Single-column DWT approximation coefficients (pywt)"""
    coeffs = pywt.wavedec(x, wavelet, level=level)
    return coeffs[0]

def _dwt_da_1d(x, wavelet, level):
    """Single-column DWT detail coefficients"""
    coeffs = pywt.wavedec(x, wavelet, level=level)
    return coeffs[1] if len(coeffs) > 1 else np.zeros_like(coeffs[0])

def _kalman_filter_1d(x, transition_covariance, observation_covariance):
    """Single-column Kalman filter"""
    x = _validate_numpy_input(x)
    if np.all(np.isnan(x)):
        return np.full_like(x, np.nan)
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=x[~np.isnan(x)][0] if np.any(~np.isnan(x)) else 0.0,
        initial_state_covariance=1.0,
        transition_covariance=transition_covariance,
        observation_covariance=observation_covariance
    )
    smoothed_state_means, _ = kf.smooth(x)
    return smoothed_state_means.flatten()

def _hp_filter_1d(x, lamb):
    """Single-column HP filter
    Optimized Hodrick-Prescott filter for a batch of 1D series (columns).
    Vectorized using sparse matrix operations.

    Args:
        x (np.ndarray): Input data array, shape (num_samples, num_features).
                                 Assumed to be NaN-free.
        lamb (float): Smoothing parameter.

    Returns:
        np.ndarray: Trend component, same shape as x.
    """
    num_samples, num_features = x.shape

    # Construct the matrix A = I + lambda * D'D (D is the second difference matrix)
    # The matrix A has a known sparse, five-diagonal structure.
    
    diagonals_A_main = np.zeros(num_samples)
    diagonals_A_off1 = np.zeros(num_samples - 1)
    diagonals_A_off2 = np.zeros(num_samples - 2)
    
    diagonals_A_main[0] = 1 + lamb
    diagonals_A_main[1] = 1 + 5 * lamb
    diagonals_A_main[2:-2] = 1 + 6 * lamb
    diagonals_A_main[num_samples-2] = 1 + 5 * lamb
    diagonals_A_main[num_samples-1] = 1 + lamb

    diagonals_A_off1[0] = -2 * lamb
    diagonals_A_off1[1:-1] = -4 * lamb
    diagonals_A_off1[num_samples-2] = -2 * lamb

    diagonals_A_off2[:] = lamb
    
    # Create sparse matrix in CSC format (efficient for solving)
    A = sparse.diags(
        [diagonals_A_main, diagonals_A_off1, diagonals_A_off1, diagonals_A_off2, diagonals_A_off2],
        [0, 1, -1, 2, -2],
        shape=(num_samples, num_samples),
        format='csc'
    )

    # Solve the system A * trend = x
    # spsolve can handle a 2D array (batch of right-hand sides) for 'b', performing vectorized solves.
    trend_batch = spsolve(A, x)
    
    return trend_batch


@torch.no_grad()
def _fft_core(array: np.ndarray, backend="numpy", device="cpu"):
    """Single-column Fast Fourier Transform (FFT)"""
    n_samples, n_features = array.shape
    
    # Prepare output array for complex numbers, shape is same as input
    output_array = np.full((n_samples, n_features), np.nan, dtype=np.complex128)

    if backend == 'torch':
        if not torch.cuda.is_available() and device == 'cuda':
            print("Warning: CUDA not available. Falling back to CPU.")
            device = 'cpu'
        
        valid_cols_mask = ~np.isnan(array).any(axis=0)
        valid_cols_data = array[:, valid_cols_mask]
        
        if valid_cols_data.shape[1] > 0:
            data_tensor = torch.from_numpy(valid_cols_data).double().to(device)
            # Perform full fft along the samples dimension (dim=0)
            fft_result_torch = torch.fft.fft(data_tensor, dim=0)
            output_array[:, valid_cols_mask] = fft_result_torch.cpu().numpy()
            
    else: # numpy backend (Optimized)
        # Identify all columns that are completely free of NaNs
        valid_cols_mask = ~np.isnan(array).any(axis=0)
        valid_cols_data = array[:, valid_cols_mask]

        if valid_cols_data.shape[1] > 0:
            # Perform full fft on all valid columns at once along the samples axis (axis=0)
            fft_result_np = np.fft.fft(valid_cols_data, axis=0)
            # Place the results back into the correct columns of the output array
            output_array[:, valid_cols_mask] = fft_result_np
                
    return output_array

def fft_angle_interp1d(x, backend="numpy", device="cpu", amplitude_threshold: float = 1e-10):
    """Fast Fourier Transform (FFT) with amplitude interpolation"""
    fft_complex = _fft_core(x, backend, device)
    # Calculate amplitude to identify low-power frequencies
    amplitude = np.abs(fft_complex)
    
    # Calculate phase
    phase = np.angle(fft_complex)
    
    # Mask the phase where amplitude is too low to be meaningful
    phase[amplitude < amplitude_threshold] = np.nan
    
    return phase

def fft_amp_interp1d(x, backend="numpy", device="cpu"):
    """Fast Fourier Transform (FFT) with real part interpolation"""
    fft_complex = _fft_core(x, backend, device)
    n_samples, n_features = x.shape
    # Calculate amplitude and normalize by the number of samples
    amplitude = np.abs(fft_complex) / n_samples
    
    # Double the amplitude of all non-DC and non-Nyquist frequencies
    # to account for the energy in the negative frequencies.
    # The range is from the first frequency bin (index 1) up to, but not including,
    # the last bin if N is even (Nyquist frequency).
    if n_samples % 2 == 0:
        amplitude[1:-1] *= 2
    else:
        amplitude[1:] *= 2        
    return amplitude

def fft_real_interp1d(x, backend="numpy", device="cpu"):
    """Fast Fourier Transform (FFT) with real part interpolation"""
    fft_complex = _fft_core(x, backend, device)
    n_samples, n_features = x.shape
    # Extract the real part of the FFT result
    real_part = np.real(fft_complex)
    
    # Normalize by the number of samples
    real_part /= n_samples
    
    # Double the real part of all non-DC and non-Nyquist frequencies
    if n_samples % 2 == 0:
        real_part[1:-1] *= 2
    else:
        real_part[1:] *= 2
        
    return real_part 

@torch.no_grad()
def dwt_ca_fixed(array, wavelet="db1", backend="torch", device='cpu'):
    """
    Fixed-coefficient DWT approximation coefficients.
    MODIFIED to output shape (N, M) for (N, M) input, by removing downsampling.
    """
    if array.ndim != 2:
        raise ValueError(f"Input array must be 2D, but got shape {array.shape}")

    # 1. Get wavelet decomposition filters using the standard PyWavelets library
    wavelet_obj = pywt.Wavelet(wavelet)
    lowpass_filter = wavelet_obj.dec_lo # Low-pass filter for approximation coefficients

    n_samples, n_features = array.shape
    output_array = np.full_like(array, np.nan, dtype=np.float64)
    filter_len = len(lowpass_filter)

    if backend == 'torch':
        # --- PyTorch Backend (Vectorized with Reflect Padding) ---
        if not torch.cuda.is_available() and device == 'cuda':
            print("Warning: CUDA not available. Falling back to CPU.")
            device = 'cpu'
        
        valid_cols_mask = ~np.isnan(array).any(axis=0)
        valid_cols_data = array[:, valid_cols_mask]
        
        if valid_cols_data.shape[1] == 0:
            return output_array

        kernel = torch.tensor(lowpass_filter[::-1].copy(), dtype=torch.float64, device=device)
        kernel = kernel.view(1, 1, -1)
        
        padding = filter_len - 1
        
        data_tensor = torch.from_numpy(valid_cols_data.T).to(device).to(torch.float64)
        data_tensor = data_tensor.unsqueeze(1)
        
        data_padded = torch.nn.functional.pad(data_tensor, (padding, 0), mode='reflect')
        cA_tensor_batch = torch.nn.functional.conv1d(data_padded, kernel)
        
        valid_result = cA_tensor_batch.squeeze(1).T.cpu().numpy()
        output_array[:, valid_cols_mask] = valid_result

    else:
        # --- NumPy Backend (with Reflect Padding) ---
        kernel = np.array(lowpass_filter[::-1], dtype=np.float64)
        padding = filter_len - 1

        for i in range(n_features):
            col = array[:, i]
            
            if np.isnan(col).any():
                continue
            
            col_padded = np.pad(col, (padding, 0), 'reflect')
            cA_col = signal.convolve(col_padded, kernel, mode='valid')
            output_array[:, i] = cA_col
            
    return output_array


@torch.no_grad()
def dwt_da_fixed(array, wavelet="db1", backend="torch", device='cpu'):
    """Fixed-coefficient DWT detail coefficients.
    MODIFIED to output shape (N, M) for (N, M) input, by removing downsampling.
    """ 
    """
    Calculates the detail coefficients of the Stationary Wavelet Transform (SWT).
    SWT is also known as undecimated DWT, meaning it avoids downsampling.

    This function operates column-wise, assuming each column is a time series.
    It is robust to NaN values; if a column contains any NaNs, its output will be all NaNs.

    Args:
        array (np.ndarray): Input array with shape (n_samples, n_features).
        wavelet (str): The name of the wavelet to use (e.g., 'db1', 'haar', 'sym4').
        backend (str): The backend to use, either 'numpy' or 'torch'.
        device (str): The device to use for the 'torch' backend ('cpu' or 'cuda').

    Returns:
        np.ndarray: An array of the same shape as the input, containing the
                    detail coefficients (cD) for each column.
    """
    if array.ndim != 2:
        raise ValueError(f"Input array must be 2D, but got shape {array.shape}")

    # 1. Get wavelet decomposition filters using the standard PyWavelets library
    wavelet_obj = pywt.Wavelet(wavelet)
    highpass_filter = wavelet_obj.dec_hi # High-pass filter for detail coefficients

    n_samples, n_features = array.shape
    output_array = np.full_like(array, np.nan, dtype=np.float64)
    filter_len = len(highpass_filter)

    if backend == 'torch':
        # --- PyTorch Backend (Vectorized with Reflect Padding) ---
        if not torch.cuda.is_available() and device == 'cuda':
            print("Warning: CUDA not available. Falling back to CPU.")
            device = 'cpu'
        
        valid_cols_mask = ~np.isnan(array).any(axis=0)
        valid_cols_data = array[:, valid_cols_mask]
        
        if valid_cols_data.shape[1] == 0:
            return output_array

        kernel = torch.tensor(highpass_filter[::-1].copy(), dtype=torch.float64, device=device)
        kernel = kernel.view(1, 1, -1)
        
        # For 'same' convolution, total padding is filter_len - 1
        padding = filter_len - 1
        
        data_tensor = torch.from_numpy(valid_cols_data.T).to(device).to(torch.float64)
        data_tensor = data_tensor.unsqueeze(1)
        
        # Apply 'reflect' padding. PyTorch handles this efficiently.
        # We add all padding to one side for 'valid' convolution to act like 'same'.
        data_padded = torch.nn.functional.pad(data_tensor, (padding, 0), mode='reflect')
        cD_tensor_batch = torch.nn.functional.conv1d(data_padded, kernel)
        
        valid_result = cD_tensor_batch.squeeze(1).T.cpu().numpy()
        output_array[:, valid_cols_mask] = valid_result

    else:
        # --- NumPy Backend (with Reflect Padding) ---
        kernel = np.array(highpass_filter[::-1], dtype=np.float64)
        
        # For 'same' convolution, total padding is filter_len - 1
        padding = filter_len - 1

        for i in range(n_features):
            col = array[:, i]
            
            if np.isnan(col).any():
                continue
            
            # Manually apply reflect padding
            col_padded = np.pad(col, (padding, 0), 'reflect')
            
            # Use 'valid' convolution on the padded signal
            cD_col = signal.convolve(col_padded, kernel, mode='valid')
            output_array[:, i] = cD_col
            
    return output_array

    

def dwt_ca(x, wavelet="db1", level=1, backend="numpy", device="cpu", n_jobs=-1):
    """
    Compute discrete wavelet transform (DWT) approximation coefficients (pywt)
    This function computes the approximation coefficients of the discrete wavelet transform (DWT)
    using the specified wavelet and level. It supports both NumPy and PyTorch backends.
    Calculation formula:
        cA[n] = sum_{k=0}^{L-1} x[n + k] * lowpass[L-1-k],  step=2
    where:
        - x is the input signal,
        - lowpass is the wave
        - L is the length of the filter,
        - cA[n] is the approximation coefficient at position n.
    """
    x = _validate_numpy_input(x)
    if np.all(np.isnan(x)):
        return _handle_all_nan(x, x.shape, backend="numpy")
    if x.ndim == 1:
        return _dwt_ca_1d(x, wavelet, level)
    result = Parallel(n_jobs=n_jobs)(
        delayed(_dwt_ca_1d)(x[:, i], wavelet, level)
        for i in range(x.shape[1])
    )
    # Ensure consistent output shape
    max_len = max(len(r) for r in result)
    result = [np.pad(r, (0, max_len - len(r)), mode='constant', constant_values=np.nan) for r in result]
    return np.column_stack(result)

def dwt_da(x, wavelet="db1", level=1, backend="numpy", device="cpu", n_jobs=-1):
    """
    Compute discrete wavelet transform (DWT) detail coefficients
    This function computes the detail coefficients of the discrete wavelet transform (DWT)
    using the specified wavelet and level. It supports both NumPy and PyTorch backends.
        Calculation formula:
            
            cD[n] = sum_{k=0}^{L-1} x[n + k] * highpass[L-1-k],  step=2
            where:
            - x is the input signal,
            - highpass is the wavelet high-pass filter coefficients,
            - L is the length of the filter,
            - cD[n] is the detail coefficient at position n.
    """
    x = _validate_numpy_input(x)
    if np.all(np.isnan(x)):
        return _handle_all_nan(x, x.shape, backend="numpy")
    if x.ndim == 1:
        return _dwt_da_1d(x, wavelet, level)
    result = Parallel(n_jobs=n_jobs)(
        delayed(_dwt_da_1d)(x[:, i], wavelet, level)
        for i in range(x.shape[1])
    )
    # Ensure consistent output shape
    max_len = max(len(r) for r in result)
    result = [np.pad(r, (0, max_len - len(r)), mode='constant', constant_values=np.nan) for r in result]
    return np.column_stack(result)

@torch.no_grad()
def savgol_filter(x, window_length=5, polyorder=2, backend="numpy", device="cpu"):
    """
    Apply Savitzky-Golay filter
        This function applies a Savitzky-Golay filter to smooth the input signal.
    Calculation formula:
        y[n] = sum_{k=0}^{M} b_k * x[n + k - (M//2)]
    where:
        - y[n] is the smoothed value at position n,
        - b_k are the filter coefficients,
        - x[n + k - (M//2)] are the input signal values within the window.
    """
    if backend == "torch":
        x = _validate_torch_input(x, device)
        if torch.all(torch.isnan(x)):
            return _to_numpy(_handle_all_nan(x, x.shape, backend="torch", device=device))
        x_np = x.cpu().numpy()
        if x_np.shape[0] < window_length:
            return _to_numpy(torch.full_like(x, float('nan')))
        filtered = signal.savgol_filter(x_np, window_length, polyorder, axis=0)
        return filtered
    else:
        x = _validate_numpy_input(x)
        if np.all(np.isnan(x)):
            return _handle_all_nan(x, x.shape, backend="numpy")
        if x.shape[0] < window_length:
            return np.full_like(x, np.nan)
        return signal.savgol_filter(x, window_length, polyorder, axis=0)

@torch.no_grad()
def median_filter(x, size=5, backend="numpy", device="cpu"):
    """
    Apply median filter
    This function applies a median filter to smooth the input signal.   
    Calculation formula:
        y[n] = median(x[n - (size//2):n + (size//2)])
    where:
        - y[n] is the smoothed value at position n,
        - x[n - (size//2):n + (size//2)] are the input signal values within the window.
    """
    if backend == "torch":
        x = _validate_torch_input(x, device)
        if torch.all(torch.isnan(x)):
            return _to_numpy(_handle_all_nan(x, x.shape, backend="torch", device=device))
        # PyTorch does not have a built-in median filter, so use unfold for 1D/2D
        if x.ndim == 1:
            x = x.unsqueeze(1)
        pad = size // 2
        x_padded = torch.nn.functional.pad(x, (0, 0, pad, pad), mode='reflect')
        result = []
        for i in range(x.shape[1]):
            patches = x_padded[:, i].unfold(0, size, 1)
            med = patches.median(dim=1).values
            # Pad result to match input length
            if med.shape[0] < x.shape[0]:
                med = torch.nn.functional.pad(med, (0, x.shape[0] - med.shape[0]), mode='replicate')
            result.append(med)
        result = torch.stack(result, dim=1)
        return _to_numpy(result)
    else:
        x = _validate_numpy_input(x)
        if np.all(np.isnan(x)):
            return _handle_all_nan(x, x.shape, backend="numpy")
        return signal.medfilt(x, kernel_size=size)

@torch.no_grad()
def lowpass_filter(x, cutoff=0.1, fs=10.0, order=5, backend="numpy", device="cpu"):
    """
    Apply low-pass filter
    This function applies a low-pass filter to the input signal.
    Calculation formula:
        y[n] = sum_{k=0}^{M} b_k * x[n - k]
    where:  
        - y[n] is the filtered value at position n,
        - b_k are the filter coefficients,          
        - x[n - k] are the input signal values.        
    """
    if backend == "torch":
        x = _validate_torch_input(x, device)
        if torch.all(torch.isnan(x)):
            return _to_numpy(_handle_all_nan(x, x.shape, backend="torch", device=device))
        x_np = x.cpu().numpy()
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        filtered = signal.lfilter(b, a, x_np, axis=0)
        return filtered
    else:
        x = _validate_numpy_input(x)
        if np.all(np.isnan(x)):
            return _handle_all_nan(x, x.shape, backend="numpy")
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        return signal.lfilter(b, a, x, axis=0)

@torch.no_grad()
def highpass_filter(x, cutoff=0.1, fs=10.0, order=5, backend="numpy", device="cpu"):
    """
    Apply high-pass filter
    This function applies a high-pass filter to the input signal.
        Calculation formula:        
        y[n] = sum_{k=0}^{M} b_k * x[n - k]
    where:  
        - y[n] is the filtered value at position n,
        - b_k are the filter coefficients,
        - x[n - k] are the input signal values.
    """
    if backend == "torch":
        x = _validate_torch_input(x, device)
        if torch.all(torch.isnan(x)):
            return _to_numpy(_handle_all_nan(x, x.shape, backend="torch", device=device))
        x_np = x.cpu().numpy()
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        filtered = signal.lfilter(b, a, x_np, axis=0)
        return filtered
    else:
        x = _validate_numpy_input(x)
        if np.all(np.isnan(x)):
            return _handle_all_nan(x, x.shape, backend="numpy")
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        return signal.lfilter(b, a, x, axis=0)

@torch.no_grad()
def bandpass_filter(x, lowcut=0.1, highcut=0.5, fs=10.0, order=5, backend="numpy", device="cpu"):
    """
    Apply bandpass filter
    This function applies a bandpass filter to the input signal.
        Calculation formula:                    
        y[n] = sum_{k=0}^{M} b_k * x[n - k]
        where:
        - y[n] is the filtered value at position n,
        - b_k are the filter coefficients,
        - x[n - k] are the input signal values.
    """
    if backend == "torch":
        x = _validate_torch_input(x, device)
        if torch.all(torch.isnan(x)):
            return _to_numpy(_handle_all_nan(x, x.shape, backend="torch", device=device))
        x_np = x.cpu().numpy()
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band', analog=False)
        filtered = signal.lfilter(b, a, x_np, axis=0)
        return filtered
    else:
        x = _validate_numpy_input(x)
        if np.all(np.isnan(x)):
            return _handle_all_nan(x, x.shape, backend="numpy")
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band', analog=False)
        return signal.lfilter(b, a, x, axis=0)

@torch.no_grad()
def wavelet_denoise(x, wavelet="db4", level=1, mode="soft", backend="numpy", device="cpu", n_jobs=-1):
    """
    Denoise using wavelet transform
    This function applies wavelet denoising to the input signal using the specified wavelet and level.
    Calculation formula:
        cD[n] = sum_{k=0}^{L-1} x[n + k] * highpass[L-1-k],  step=2
    where:
        - x is the input signal,
        - highpass is the wavelet high-pass filter coefficients,        
        - L is the length of the filter,
        - cD[n] is the detail coefficient at position n.
    """
    
    x_tensor = _validate_torch_input(x, device)
    if torch.all(torch.isnan(x_tensor)):
        return _to_numpy(_handle_all_nan(x_tensor, x_tensor.shape, backend="torch", device=device))
    
    x_np_original = _to_numpy(x_tensor) # Convert to NumPy for pywt and pandas processing
    num_samples, num_features = x_np_original.shape

    # Store original NaN mask to re-apply later
    original_nan_mask = np.isnan(x_np_original)
    
    x_np_interp = interpolator_arr(x_np_original)

    # Explicitly set columns with less than 2 valid points back to NaN if Pandas didn't
    valid_counts_per_col = (~original_nan_mask).sum(axis=0)
    x_np_interp[:, valid_counts_per_col < 2] = np.nan

    # --- DWT/IDWT Processing (fully vectorized using pywt's axes parameter) ---
    # pywt functions will correctly handle columns that are now all NaNs from interpolation
    # (they will typically return all-NaN coefficients/reconstructions).

    # Perform multi-level DWT for all columns at once using axes=0
    coeffs_list = pywt.wavedecn(x_np_interp, wavelet, level=level, axes=0)
    
    # Calculate threshold using standard deviation of finest detail coefficients
    # Flatten the finest detail coefficients across all features for a single threshold
    all_finest_detail_coeffs = coeffs_list[-1]['d'].flatten()
    # Handle case where finest detail coeffs are all NaNs (e.g., if input was all NaNs or single point)
    if np.all(np.isnan(all_finest_detail_coeffs)):
        threshold = 0.0 # Or some default, or propagate NaNs
    else:
        threshold = np.nanstd(all_finest_detail_coeffs) * np.sqrt(2 * np.log(num_samples))
        # Use np.nanstd to ignore NaNs in std calculation if any survived somehow

    # Apply thresholding to detail coefficients
    denoised_coeffs_list = [coeffs_list[0]] # Approximation coefficients (cA)
    for i in range(1, len(coeffs_list)): # Iterate through detail coefficient levels (cD1, cD2, ...)
        detail_coeffs_dict = coeffs_list[i]
        denoised_level_dict = {}
        for key, detail_array in detail_coeffs_dict.items():
            # Apply thresholding to the NumPy array
            denoised_level_dict[key] = pywt.threshold(detail_array, threshold, mode=mode)
        denoised_coeffs_list.append(denoised_level_dict) # Append the modified dictionary
    
    # Perform Multi-level IDWT to reconstruct the signal (fully vectorized)
    reconstructed_data_np_raw = pywt.waverecn(denoised_coeffs_list, wavelet, axes=0)

    # Ensure output shape matches original (num_samples, num_features)
    # pywt.waverecn might sometimes return slightly longer/shorter depending on padding mode.
    if reconstructed_data_np_raw.shape[0] > num_samples:
        reconstructed_data_np = reconstructed_data_np_raw[:num_samples, :]
    elif reconstructed_data_np_raw.shape[0] < num_samples:
        reconstructed_data_np = np.pad(reconstructed_data_np_raw, 
                                        ((0, num_samples - reconstructed_data_np_raw.shape[0]), (0, 0)), 
                                        mode='edge')
    else:
        reconstructed_data_np = reconstructed_data_np_raw

    # Re-apply original NaNs to the denoised result (column-wise, vectorized)
    denoised_final_np = reconstructed_data_np
    denoised_final_np[original_nan_mask] = np.nan # This is a vectorized operation
    return denoised_final_np

@torch.no_grad()
def ewma_filter(x, span=10, backend="torch", device='cpu'):
    """
    Apply exponential weighted moving average (EWMA)
    This function applies an exponential weighted moving average (EWMA) filter to the input signal.
    Calculation formula:
        y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
    where:
        - y[n] is the filtered value at position n,
        - x[n] is the input signal value at position n,
        - alpha = 2 / (span + 1) is the smoothing factor.
    """
    x = _validate_torch_input(x, device)
    if torch.all(torch.isnan(x)):
        return _to_numpy(_handle_all_nan(x, x.shape, backend="torch", device=device))
    alpha = 2 / (span + 1)
    result = torch.zeros_like(x)
    result[0] = x[0]
    for i in range(1, x.shape[0]):
        result[i] = alpha * x[i] + (1 - alpha) * result[i-1]
    return _to_numpy(result)

def kalman_filter(x, transition_covariance=0.01, observation_covariance=0.1, backend="numpy", device="cpu", n_jobs=-1):
    """
    Apply Kalman filter
    This function applies a Kalman filter to the input signal.
    Calculation formula:
        x_t = A * x_{t-1} + w_t
        y_t = H * x_t + v_t
    where:
        - x_t is the state at time t,
        - A is the state transition matrix,
        - w_t is the process noise,
        - y_t is the observation at time t,
        - H is the observation matrix,
        - v_t is the observation noise.
    """
    x = _validate_numpy_input(x)
    if np.all(np.isnan(x)):
        return _handle_all_nan(x, x.shape, backend="numpy")
    if x.ndim == 1:
        return _kalman_filter_1d(x, transition_covariance, observation_covariance)
    result = [
        _kalman_filter_1d(x[:, i], transition_covariance, observation_covariance)
        for i in range(x.shape[1])
    ]
    
    return np.column_stack(result)

def hp_filter(x, lamb=1600, backend="numpy", device="cpu", n_jobs=-1):
    """
    Apply Hodrick-Prescott filter
    This function applies a Hodrick-Prescott filter to the input signal.
    Calculation formula:
        (1 - 4 * L^2) * c_t + 4 * L^2 * c_{t-1} - c_{t-2} = lamb * (x_t - c_t)
    where:
        - c_t is the trend component at time t,
        - x_t is the input signal at time t,
        - L is the smoothing parameter.
    """
    x_np_original = _validate_numpy_input(x)
    
    if np.all(np.isnan(x_np_original)):
        return _handle_all_nan(x_np_original, x_np_original.shape)
    
    if x_np_original.ndim == 1:
        num_samples = x_np_original.shape[0]
        nan_mask_1d = np.isnan(x_np_original)
        
        if np.any(nan_mask_1d):
            if np.sum(~nan_mask_1d) < 2:
                trend_np_1d = np.full_like(x_np_original, np.nan)
            else:
                non_nan_indices = np.arange(num_samples)[~nan_mask_1d]
                non_nan_values = x_np_original[~nan_mask_1d]
                interp_func = interp1d(non_nan_indices, non_nan_values, 
                                       kind='linear', bounds_error=False, fill_value='extrapolate')
                x_np_interp_1d = interp_func(np.arange(num_samples))
                # Use statsmodels hpfilter for 1D, as it's optimized C code
                _, trend_np_1d = hpfilter(x_np_interp_1d, lamb=lamb)
        else:
            _, trend_np_1d = hpfilter(x_np_original, lamb=lamb)
        
        if np.any(nan_mask_1d):
            trend_np_1d[nan_mask_1d] = np.nan
        
        return trend_np_1d

    elif x_np_original.ndim == 2:
        num_samples, num_features = x_np_original.shape

        original_nan_mask = np.isnan(x_np_original)
        # --- Vectorized NaN Interpolation using Pandas ---
        x_np_interp = interpolator_arr(x_np_original)

        # Explicitly set columns with less than 2 valid points back to NaN
        valid_counts_per_col = (~original_nan_mask).sum(axis=0)
        x_np_interp[:, valid_counts_per_col < 2] = np.nan
        
        # --- Fully Vectorized HP Filter on the batch ---
        # This calls the _hp_filter_optimized_batch_numpy function
        trend_np_raw = _hp_filter_1d(x_np_interp, lamb)

        # Re-apply original NaNs to the filtered result
        trend_final_np = trend_np_raw
        trend_final_np[original_nan_mask] = np.nan

        return trend_final_np

    else:
        raise ValueError("Input data must be 1D or 2D.")


@torch.no_grad()
def robust_zscore_filter(x, threshold=3.0, backend="torch", device='cpu'):
    """
    Robust filter based on median and Z-score
    This function applies a robust Z-score filter to the input signal.
    Calculation formula:
        z[n] = 0.6745 * (x[n] - median) / mad
    where:
        - z[n] is the Z-score at position n,
        - x[n] is the input signal value at position n,
        - median is the median of the input signal,
        - mad is the median absolute deviation of the input signal.
    """
    x = _validate_torch_input(x, device)
    if torch.all(torch.isnan(x)):
        return _to_numpy(_handle_all_nan(x, x.shape, backend="torch", device=device))
    median = torch.median(x, dim=0, keepdim=True).values
    mad = torch.median(torch.abs(x - median), dim=0, keepdim=True).values
    z_scores = torch.where(mad != 0, 0.6745 * (x - median) / mad, torch.zeros_like(x))
    result = torch.where(
        torch.abs(z_scores) > threshold,
        torch.tensor(float('nan'), device=device),
        x
    )
    return _to_numpy(result)

def rolling_rank_filter(x, window=5, backend="numpy", device="cpu"):
    """
    Apply rolling rank filter
    This function applies a rolling rank filter to the input signal.
    Calculation formula:
        y[n] = rank(x[n-window+1:n]) if n >= window else rank(x[0:n])
    where:
        - y[n] is the filtered value at position n,
        - x[n] is the input signal value at position n,
        - window is the size of the rolling window.
    """
    x_np_original = _validate_numpy_input(x)
    
    if np.all(np.isnan(x_np_original)):
        return _handle_all_nan(x_np_original, x_np_original.shape)

    # Convert to Pandas DataFrame for efficient rolling operations
    df = pd.DataFrame(x_np_original)
    original_ndim = x_np_original.ndim
    if original_ndim == 1:
        df = pd.DataFrame(x_np_original.reshape(-1, 1))
    else:
        df = pd.DataFrame(x_np_original)

    # 1. 插值NaN (与之前HP Filter的优化类似)
    df_interp = df.interpolate(method='linear', axis=0, limit_direction='both', limit_area=None)
    
    # 2. 标记少于2个有效点的列 (插值后可能仍为NaN)
    original_nan_mask = np.isnan(x_np_original) # 重新使用原始的NaN mask
    valid_counts_per_col = (~original_nan_mask).sum(axis=0)
    
    # 3. 计算滚动秩
    # .rolling().transform('rank', pct=True) 是Pandas中计算滚动秩的向量化、内置方法。
    # 它会为窗口中的每个元素计算其在当前窗口中的百分比排名。
    # min_periods=1 确保即使窗口不完整也会计算（对于开头的元素）。
    # rank method='average' is default.
    rolling_ranks_df = df_interp.rolling(window=window, min_periods=1).rank(pct=True)

    result_np = rolling_ranks_df.values

    # 4. 重新应用原始的NaN
    # 对于插值前就是NaN的原始位置，我们去噪/滤波后也应该将其设为NaN。
    # 另外，对于少于2个有效点的列，结果也应该全部是NaN。
    
    # 首先，对于插值前就是NaN的位置，恢复为NaN
    result_np[original_nan_mask] = np.nan

    # 其次，对于原始就是少于2个有效点的列，确保它们全是NaN
    if original_ndim == 1: # For 1D input, need to check its single column
        if valid_counts_per_col.item() < 2: # .item() for 0-dim array
            result_np[:] = np.nan
    else: # For 2D input
        result_np[:, valid_counts_per_col < 2] = np.nan

    # 如果原始输入是1D，则将结果压缩回1D
    if original_ndim == 1:
        result_np = result_np.flatten()
    return result_np

@torch.no_grad()
def hilbert_transform_instantaneous_phase(
    array: np.ndarray, 
    backend: str = "numpy", 
    device: str = "cpu"
) -> np.ndarray:
    """
    Calculates the Hilbert Transform Instantaneous Phase consistently across backends.

    This function assumes that each row of the input array is a separate signal,
    and the transform should be applied along axis=1.

    Args:
        array (np.ndarray): Input array with shape (n_signals, n_samples_per_signal).
                            For your case, this is (48, 3988).
        backend (str): Backend to use ('numpy' or 'torch').
        device (str): Device to use for torch backend ('cpu' or 'cuda').

    Returns:
        np.ndarray: The unwrapped instantaneous phase of the analytic signal.
    """
    # --- Parameter Validation ---
    if array.ndim != 2:
        raise ValueError(f"Input array must be 2D, but got shape {array.shape}")
    # 1. Apply the Hilbert transform along axis=1.
    # This is the crucial change to make it consistent with the torch backend
    # and the likely intent for a (signals, samples) shaped array.
    analytic_signal = signal.hilbert(array, axis=1)

    # 2. Calculate the angle of the complex analytic signal.
    angle = np.angle(analytic_signal)

    # 3. Unwrap the phase along axis=1 to match the transform axis.
    instantaneous_phase = np.unwrap(angle, axis=1)
    
    return instantaneous_phase


@torch.no_grad()
def kaufman_adaptive_moving_average(x, fast_period=2, slow_period=10, backend="torch", device='cpu'):
    """
    Apply Kaufman Adaptive Moving Average (KAMA)
    This function applies the Kaufman Adaptive Moving Average (KAMA) to the input signal.
    Calculation formula:
        KAMA[n] = KAMA[n-1] + alpha * (x[n] - KAMA[n-1])
    where:
        - KAMA[n] is the KAMA at position n,
        - x[n] is the input signal value at position n,
        - alpha is the smoothing factor based on volatility.
    Default parameters are set for 5min/15min frequency data.
    """
    if backend == "torch":
        x = _validate_torch_input(x, device)
        if torch.all(torch.isnan(x)):
            return _to_numpy(_handle_all_nan(x, x.shape, backend="torch", device=device))
        n = x.shape[0]
        kama = torch.zeros_like(x)
        kama[0] = x[0]
        fast_alpha = 2 / (fast_period + 1)
        slow_alpha = 2 / (slow_period + 1)
        for i in range(1, n):
            change = torch.abs(x[i] - x[i-1])
            volatility = torch.sum(torch.abs(x[max(0, i-slow_period+1):i+1] - x[max(0, i-slow_period+1):i+1].mean()))
            if volatility == 0:
                alpha = 0
            else:
                efficiency_ratio = change / volatility
                alpha = (efficiency_ratio * (fast_alpha - slow_alpha) + slow_alpha)
            kama[i] = kama[i-1] + alpha * (x[i] - kama[i-1])
        return _to_numpy(kama)
    else:
        x = _validate_numpy_input(x)
        if np.all(np.isnan(x)):
            return _handle_all_nan(x, x.shape, backend="numpy")
        n = x.shape[0]
        kama = np.zeros_like(x)
        kama[0] = x[0]
        fast_alpha = 2 / (fast_period + 1)
        slow_alpha = 2 / (slow_period + 1)
        for i in range(1, n):
            change = np.abs(x[i] - x[i-1])
            window_start = max(0, i-slow_period+1)
            window = x[window_start:i+1]
            volatility = np.sum(np.abs(window - np.mean(window)))
            if volatility == 0:
                alpha = 0
            else:
                efficiency_ratio = change / volatility
                alpha = (efficiency_ratio * (fast_alpha - slow_alpha) + slow_alpha)
            kama[i] = kama[i-1] + alpha * (x[i] - kama[i-1])
        return kama

@torch.no_grad()
def mesa_adaptive_moving_average(x, fast_period=2, slow_period=10, backend="numpy", device="cpu"):
    """
    mesa Adaptive Moving Average (MAMA)
    This function applies the Mesa Adaptive Moving Average (MAMA) to the input signal.
    Calculation formula:
        MAMA[n] = MAMA[n-1] + alpha * (x[n] - MAMA[n-1])
        FAMA[n] = FAMA[n-1] + beta * (x[n] - FAMA[n-1])
    where:
        - MAMA[n] is the MAMA at position n,
        - x[n] is the input signal value at position n,
        - alpha is the smoothing factor based on volatility.
    Default parameters are set for 5min/15min frequency data.
    """
    if backend == "torch":
        x = _validate_torch_input(x, device)
        if torch.all(torch.isnan(x)):
            return _to_numpy(_handle_all_nan(x, x.shape, backend="torch", device=device))
        n = x.shape[0]
        mama = torch.zeros_like(x)
        mama[0] = x[0]
        fama = torch.zeros_like(x)  
        fama[0] = x[0]  # FAMA starts the same as MAMA
        fast_alpha = 2 / (fast_period + 1)
        slow_alpha = 2 / (slow_period + 1)
        for i in range(1, n):
            change = torch.abs(x[i] - x[i-1])
            volatility = torch.sum(torch.abs(x[max(0, i-slow_period+1):i+1] - x[max(0, i-slow_period+1):i+1].mean()))
            if volatility == 0:
                alpha = 0
            else:
                efficiency_ratio = change / volatility
                alpha = (efficiency_ratio * (fast_alpha - slow_alpha) + slow_alpha)
            mama[i] = mama[i-1] + alpha * (x[i] - mama[i-1])
            fama[i] = fama[i-1] + (0.5 * (mama[i] - fama[i-1]))  # FAMA is smoothed MAMA
        # Return MAMA and FAMA as a tuple
        return _to_numpy(mama-fama)
    else:
        x = _validate_numpy_input(x)
        if np.all(np.isnan(x)):
            return _handle_all_nan(x, x.shape, backend="numpy")
        n = x.shape[0]
        mama = np.zeros_like(x)
        fama = np.zeros_like(x)
        fama[0] = x[0]  # FAMA starts the same as M
        mama[0] = x[0]
        fast_alpha = 2 / (fast_period + 1)
        slow_alpha = 2 / (slow_period + 1)
        for i in range(1, n):
            change = np.abs(x[i] - x[i-1])
            window_start = max(0, i-slow_period+1)
            window = x[window_start:i+1]
            volatility = np.sum(np.abs(window - np.mean(window)))
            if volatility == 0:
                alpha = 0
            else:
                efficiency_ratio = change / volatility
                alpha = (efficiency_ratio * (fast_alpha - slow_alpha) + slow_alpha)
            mama[i] = mama[i-1] + alpha * (x[i] - mama[i-1])
            fama[i] = fama[i-1] + (0.5 * (mama[i] - fama[i-1]))
        return mama - fama



