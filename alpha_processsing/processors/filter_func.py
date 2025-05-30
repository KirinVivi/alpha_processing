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
    """Get fixed wavelet filter coefficients"""
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
    """Fixed-coefficient DWT approximation coefficients"""
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
    """Fixed-coefficient DWT detail coefficients"""
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
    """Compute discrete wavelet transform (DWT) approximation coefficients (pywt)"""
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
    """Compute discrete wavelet transform (DWT) detail coefficients"""
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
    """Apply Savitzky-Golay filter"""
    x = _validate_numpy_input(x)
    if np.all(np.isnan(x)):
        return _handle_all_nan(x, x.shape, backend="numpy")
    if x.shape[0] < window_length:
        return np.full_like(x, np.nan)
    return signal.savgol_filter(x, window_length, polyorder, axis=0)

@torch.no_grad()
def median_filter(x, size=5, backend="numpy", device="cpu"):
    """Apply median filter"""
    x = _validate_numpy_input(x)
    if np.all(np.isnan(x)):
        return _handle_all_nan(x, x.shape, backend="numpy")
    return signal.medfilt(x, kernel_size=size)

def lowpass_filter(x, cutoff=0.1, fs=10.0, order=5, backend="numpy", device="cpu"):
    """Apply low-pass filter"""
    x = _validate_numpy_input(x)
    if np.all(np.isnan(x)):
        return _handle_all_nan(x, x.shape, backend="numpy")
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return signal.lfilter(b, a, x, axis=0)

def highpass_filter(x, cutoff=0.1, fs=10.0, order=5, backend="numpy", device="cpu"):
    """Apply high-pass filter"""
    x = _validate_numpy_input(x)
    if np.all(np.isnan(x)):
        return _handle_all_nan(x, x.shape, backend="numpy")
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return signal.lfilter(b, a, x, axis=0)

def butter_highpass_filter(x, cutoff=0.1, fs=10.0, order=5, backend="numpy", device="cpu"):
    """Apply Butterworth high-pass filter"""
    return highpass_filter(x, cutoff, fs, order, backend=backend, device=device)

def bandpass_filter(x, lowcut=0.1, highcut=0.5, fs=10.0, order=5, backend="numpy", device="cpu"):
    """Apply bandpass filter"""
    x = _validate_numpy_input(x)
    if np.all(np.isnan(x)):
        return _handle_all_nan(x, x.shape, backend="numpy")
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band', analog=False)
    return signal.lfilter(b, a, x, axis=0)

def wavelet_denoise(x, wavelet="db4", level=1, mode="soft", backend="numpy", device="cpu", n_jobs=-1):
    """Denoise using wavelet transform"""
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
    """Apply exponential weighted moving average (EWMA)"""
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
    """Apply Kalman filter"""
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
    """Apply Hodrick-Prescott filter"""
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
    """Robust filter based on median and Z-score"""
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
    """Apply rolling rank filter"""
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
def adaptive_moving_average(x, window=10, sensitivity=2.0, backend="torch", device='cpu'):
    """Apply adaptive moving average (AMA)"""
    x = _validate_torch_input(x, device)
    if torch.all(torch.isnan(x)):
        return _to_numpy(_handle_all_nan(x, x.shape, backend="torch", device=device))
    volatility = torch.abs(torch.diff(x, dim=0))
    volatility = torch.nn.functional.pad(volatility, (0, 0, 1, 0), mode='replicate')
    kernel = torch.ones(1, 1, window, device=device) / window
    volatility = torch.conv1d(volatility.unsqueeze(0).unsqueeze(-1), kernel, padding=0).squeeze(-1).squeeze(0)
    volatility = torch.nn.functional.pad(volatility, (window-1, 0), mode='replicate')
    volatility = torch.nan_to_num(volatility, nan=1e-10)
    fast = 2 / (2 + 1)
    slow = 2 / (30 + 1)
    scaling = (volatility / torch.max(volatility, dim=0, keepdim=True).values) * sensitivity
    alpha = torch.clamp(fast + scaling * (fast - slow), slow, fast)
    result = torch.zeros_like(x)
    result[0] = x[0]
    for i in range(1, x.shape[0]):
        result[i] = alpha[i] * x[i] + (1 - alpha[i]) * result[i-1]
    return _to_numpy(result)

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