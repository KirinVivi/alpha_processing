import numpy as np
import torch
import scipy.signal as signal
from scipy.interpolate import interp1d
import pywt
from statsmodels.tsa.filters.hp_filter import hpfilter
from pykalman import KalmanFilter
import pandas as pd
from joblib import Parallel, delayed

def _validate_numpy_input(arr):
    """Validate input as 1D or 2D NumPy array"""
    arr = np.asarray(arr)
    if arr.ndim not in (1, 2):
        raise ValueError("Input must be 1D or 2D NumPy array")
    return arr

def _validate_torch_input(arr, device='cpu'):
    """Validate input as 1D or 2D PyTorch tensor"""
    if isinstance(arr, np.ndarray):
        arr = torch.from_numpy(arr).float()
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

def _get_wavelet_filters(wavelet="db1"):
    """
    Get fixed wavelet filter coefficients
    Supported wavelets: 'db1' (Haar), 'db4' (Daubechies 4)
    
    db1 (Haar):
        lowpass = [1/√2, 1/√2]
        highpass = [1/√2, -1/√2]
    db4 (Daubechies 4):
        lowpass = [
            0.4829629131445341, 0.8365163037378079,
            0.2241438680420134, -0.1294095225512604
        ]
        highpass = [
            -0.1294095225512604, -0.2241438680420134,
            0.8365163037378079, -0.4829629131445341
        ]
    """
    if wavelet == "db1":
        lowpass = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        highpass = np.array([1/np.sqrt(2), -1/np.sqrt(2)])
    elif wavelet == "db4":
        lowpass = np.array([
            0.4829629131445341, 0.8365163037378079,
            0.2241438680420134, -0.1294095225512604
        ])
        highpass = np.array([
            -0.1294095225512604, -0.2241438680420134,
            0.8365163037378079, -0.4829629131445341
        ])
    else:
        raise ValueError("Unsupported wavelet: choose 'db1' or 'db4'")
    return lowpass, highpass

@torch.no_grad()
def _dwt_fixed_1d_torch(x, lowpass, highpass, device='cpu'):
    """Single-column fixed-coefficient DWT (PyTorch)"""
    x = _validate_torch_input(x, device)
    if x.ndim == 1:
        x = x.view(-1, 1)
    n = x.shape[0]
    pad_len = len(lowpass) - 1
    x_padded = torch.nn.functional.pad(x, (pad_len//2, pad_len//2), mode='reflect')
    x_padded = x_padded.unsqueeze(0).unsqueeze(-1)  # [1, n+pad, 1, 1]
    lowpass = torch.tensor(lowpass, dtype=torch.float32, device=device).view(1, 1, -1, 1)
    highpass = torch.tensor(highpass, dtype=torch.float32, device=device).view(1, 1, -1, 1)
    cA = torch.conv2d(x_padded, lowpass, stride=(2, 1)).squeeze(-1).squeeze(0)  # [n//2, 1]
    cD = torch.conv2d(x_padded, highpass, stride=(2, 1)).squeeze(-1).squeeze(0)  # [n//2, 1]
    return cA, cD

def _dwt_fixed_1d_numpy(x, lowpass, highpass):
    """Single-column fixed-coefficient DWT (NumPy)"""
    x = _validate_numpy_input(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    pad_len = len(lowpass) - 1
    x_padded = np.pad(x, ((pad_len//2, pad_len//2), (0, 0)), mode='reflect')
    cA = signal.convolve(x_padded, lowpass[::-1].reshape(-1, 1), mode='valid')[::2]
    cD = signal.convolve(x_padded, highpass[::-1].reshape(-1, 1), mode='valid')[::2]
    return cA, cD

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
    """Single-column HP filter"""
    x = _validate_numpy_input(x)
    if np.all(np.isnan(x)):
        return np.full_like(x, np.nan)
    cycle, trend = hpfilter(x, lamb=lamb)
    return trend

def _wavelet_denoise_1d(x, wavelet, level, mode):
    """Single-column wavelet denoising"""
    x = _validate_numpy_input(x)
    if np.all(np.isnan(x)):
        return np.full_like(x, np.nan)
    coeffs = pywt.wavedec(x, wavelet, level=level)
    threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(x)))
    coeffs = [pywt.threshold(c, threshold, mode=mode) for c in coeffs]
    return pywt.waverec(coeffs, wavelet)

@torch.no_grad()
def dwt_ca_fixed(x, wavelet="db1", backend="torch", device='cpu'):
    """
    Fixed-coefficient DWT approximation coefficients
    This function computes the approximation coefficients of the discrete wavelet transform (DWT)
    using fixed wavelet filters. It supports both PyTorch and NumPy backends.

    Calculation formula:
        cA[n] = sum_{k=0}^{L-1} x[n + k] * lowpass[L-1-k],  step=2
    where:
        - x is the input signal,
        - lowpass is the wavelet low-pass filter coefficients,
        - L is the length of the filter,
        - cA[n] is the approximation coefficient at position n.
    """
    if backend == "torch":
        x = _validate_torch_input(x, device)
        if torch.all(torch.isnan(x)):
            return _to_numpy(_handle_all_nan(x, x.shape, backend="torch", device=device))
        lowpass, highpass = _get_wavelet_filters(wavelet)
        if x.ndim == 1:
            cA, _ = _dwt_fixed_1d_torch(x, lowpass, highpass, device=device)
            return _to_numpy(cA.squeeze(-1))
        result = []
        for i in range(x.shape[1]):
            cA, _ = _dwt_fixed_1d_torch(x[:, i], lowpass, highpass, device=device)
            result.append(cA.squeeze(-1))
        return _to_numpy(torch.stack(result, dim=1))
    else:
        x = _validate_numpy_input(x)
        if np.all(np.isnan(x)):
            return _handle_all_nan(x, x.shape, backend="numpy")
        lowpass, highpass = _get_wavelet_filters(wavelet)
        if x.ndim == 1:
            cA, _ = _dwt_fixed_1d_numpy(x, lowpass, highpass)
            return cA.flatten()
        result = []
        for i in range(x.shape[1]):
            cA, _ = _dwt_fixed_1d_numpy(x[:, i:i+1], lowpass, highpass)
            result.append(cA)
        return np.column_stack(result)

@torch.no_grad()
def dwt_da_fixed(x, wavelet="db1", backend="torch", device='cpu'):
    """
    Fixed-coefficient DWT detail coefficients
    This function computes the detail coefficients of the discrete wavelet transform (DWT)
    using fixed wavelet filters. It supports both PyTorch and NumPy backends.
    Calculation formula:    
        cD[n] = sum_{k=0}^{L-1} x[n + k] * highpass[L-1-k],  step=2
    where:
        - x is the input signal,
        - highpass is the wavelet high-pass filter coefficients,    
        - L is the length of the filter,
        - cD[n] is the detail coefficient at position n.

    """
    if backend == "torch":
        x = _validate_torch_input(x, device)
        if torch.all(torch.isnan(x)):
            return _to_numpy(_handle_all_nan(x, x.shape, backend="torch", device=device))
        lowpass, highpass = _get_wavelet_filters(wavelet)
        if x.ndim == 1:
            _, cD = _dwt_fixed_1d_torch(x, lowpass, highpass, device=device)
            return _to_numpy(cD.squeeze(-1))
        result = []
        for i in range(x.shape[1]):
            _, cD = _dwt_fixed_1d_torch(x[:, i], lowpass, highpass, device=device)
            result.append(cD.squeeze(-1))
        return _to_numpy(torch.stack(result, dim=1))
    else:
        x = _validate_numpy_input(x)
        if np.all(np.isnan(x)):
            return _handle_all_nan(x, x.shape, backend="numpy")
        lowpass, highpass = _get_wavelet_filters(wavelet)
        if x.ndim == 1:
            _, cD = _dwt_fixed_1d_numpy(x, lowpass, highpass)
            return cD.flatten()
        result = []
        for i in range(x.shape[1]):
            _, cD = _dwt_fixed_1d_numpy(x[:, i:i+1], lowpass, highpass)
            result.append(cD)
        return np.column_stack(result)

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
    if backend == "torch":
        x = _validate_torch_input(x, device)
        if torch.all(torch.isnan(x)):
            return _to_numpy(_handle_all_nan(x, x.shape, backend="torch", device=device))
        x_np = x.cpu().numpy()
        if x_np.ndim == 1:
            return _wavelet_denoise_1d(x_np, wavelet, level, mode)
        result = Parallel(n_jobs=n_jobs)(
            delayed(_wavelet_denoise_1d)(x_np[:, i], wavelet, level, mode)
            for i in range(x_np.shape[1])
        )
        return np.column_stack(result)
    else:
        x = _validate_numpy_input(x)
        if np.all(np.isnan(x)):
            return _handle_all_nan(x, x.shape, backend="numpy")
        if x.ndim == 1:
            return _wavelet_denoise_1d(x, wavelet, level, mode)
        result = Parallel(n_jobs=n_jobs)(
            delayed(_wavelet_denoise_1d)(x[:, i], wavelet, level, mode)
            for i in range(x.shape[1])
        )
        return np.column_stack(result)

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
    result = Parallel(n_jobs=n_jobs)(
        delayed(_kalman_filter_1d)(x[:, i], transition_covariance, observation_covariance)
        for i in range(x.shape[1])
    )
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
    x = _validate_numpy_input(x)
    if np.all(np.isnan(x)):
        return _handle_all_nan(x, x.shape, backend="numpy")
    if x.ndim == 1:
        return _hp_filter_1d(x, lamb)
    result = Parallel(n_jobs=n_jobs)(
        delayed(_hp_filter_1d)(x[:, i], lamb)
        for i in range(x.shape[1])
    )
    return np.column_stack(result)

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
    x = _validate_numpy_input(x)
    if np.all(np.isnan(x)):
        return _handle_all_nan(x, x.shape, backend="numpy")
    result = np.zeros_like(x)
    for i in range(x.shape[1]):
        series = pd.Series(x[:, i])
        result[:, i] = series.rolling(window=window, min_periods=1).apply(
            lambda y: pd.Series(y).rank(pct=True).iloc[-1] if not np.all(np.isnan(y)) else np.nan,
            raw=True
        )
    return result
@torch.no_grad()

def hilbert_transform_instantaneous_phase(array: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
    """
    Calculate the Hilbert Transform Instantaneous Phase
    Args:
        array (np.ndarray): Input array with shape (n_samples, n_features)
        backend (str): Backend to use ('numpy' or 'torch')
        device (str): Device to use for torch backend ('cpu' or 'cuda') 
    """
    # Use scipy.signal.hilbert for both backends to ensure correct analytic signal calculation
    if backend == "torch":
        x = torch.from_numpy(signal).float().to(device)
        N = x.shape[0]

        # Perform FFT
        Xf = torch.fft.fft(x)

        # Create Hilbert filter in frequency domain
        h = torch.zeros(N, dtype=torch.complex64, device=device)
        h[0] = 1
        if N % 2 == 0:
            h[1:N//2] = 2
            h[N//2] = 1
        else:
            h[1:(N+1)//2] = 2

        # Apply filter and get analytic signal
        analytic_signal = torch.fft.ifft(Xf * h)

        # Get phase
        instantaneous_phase = torch.unwrap(torch.angle(analytic_signal))
        return instantaneous_phase.cpu().numpy()
    else:
        analytic_signal = signal.hilbert(array, axis=0)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
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
        return mama


def interpolator_arr(arr):
    """Interpolate array to handle NaNs"""
    arr = _validate_numpy_input(arr)
    if arr.ndim == 1:
        x = np.arange(len(arr))
        if np.all(np.isnan(arr)):
            return np.full_like(arr, np.nan)
        valid = ~np.isnan(arr)
        if np.sum(valid) < 2:
            return np.full_like(arr, np.nan)
        interpolator = interp1d(x[valid], arr[valid], bounds_error=False, fill_value="extrapolate")
        return interpolator(x)
    result = Parallel(n_jobs=-1)(
        delayed(interpolator_arr)(arr[:, i])
        for i in range(arr.shape[1])
    )
    return np.column_stack(result)