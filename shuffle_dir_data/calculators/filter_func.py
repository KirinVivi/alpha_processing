import numpy as np
import scipy.signal as signal
from scipy.interpolate import interp1d

# common filter func
def ftt(x):
    """
    Computes the Fast Fourier Transform (FFT) of a 1D numpy array,
    returns magnitude, phase, and the original signal.
    """
    N = len(x)
    # Compute FFT
    X = np.fft.fft(x)
    # Compute magnitude and phase
    magnitude = np.abs(X) * 2/N  # Normalize magnitude
    phase = np.angle(X)
    return magnitude, phase, X

def dwt(x):
    """
    Performs Discrete Wavelet Transform (DWT) on a 1D numpy array.

    Args:
        x (np.ndarray): The input signal.

    Returns:
        tuple: A tuple containing the approximation coefficients (cA) and detail coefficients (cD).
    """
    # Use numpy arrays for filters for faster computation
    low_pass_filter = np.array([-0.12940952255092145, 0.2241438680420134, 0.83651630373746899, 0.48296])
    high_pass_filter = np.array([-0.48296, 0.83651630373746899, 0.2241438680420134, -0.12940952255092145])
    n = len(x)
    fnlen = n // 2
    # Only pad if necessary
    n_pad = int(2**np.ceil(np.log2(n)) - n)
    x_padded = np.pad(x, (0, n_pad), mode='constant')
    # Use 'valid' mode for less computation if signal is long
    cA = np.convolve(x_padded, low_pass_filter, mode='valid')[::2]
    cD = np.convolve(x_padded, high_pass_filter, mode='valid')[::2]
    # Truncate to expected length
    cA = cA[:fnlen]
    cD = cD[:fnlen]
    return cA, cD
   
   

def np_savitzky_golay(y):
    """
    Savitzky-Golay
    :param
    y(numpy array):
    window_size: Automatically determined based on the length of the input array.
    order(int): Polynomial order for the filter. Higher values (e.g., >3) may lead to overfitting.
    :return
    numpy array:
    """
    # check if it's all nan or all zero
    if np.all(np.isnan(y)) or np.all(y == 0):
        return np.full_like(y, np.nan)
    # initialize the params
    original_length = len(y)
    n = original_length
    order = 2  # Polynomial order for the filter, can be adjusted
    # selected on the length
    if n<= 8:
    # for the 15min 30min
        window_size =3
    elif n<= 24: # for the 15min 5min
        window_size =5
    else:
        std_dev = np.nanstd((y - np.nanmean(y, keepdims=True)) / np.nanstd(y, keepdims=True))
        if std_dev > 1.0:
            window_size = max(7,min(len(y)// 5,25))
        else:
            window_size =max(5,min(len(y)//10 ,15))
    window_size = max(5, min(len(y) // 10, 15))
    window_size += 1 if window_size % 2 == 0 else 0
    # fill the nan data, forward fill
    y_copy = y.copy()
    mask = np.isnan(y_copy)
    y_copy[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y_copy[~mask])
    # cal part
    half_window=(window_size-1)// 2
    x = np.arange(-half_window,half_window + 1)
    X = np.vander(x, order +1,increasing=True)
    XTX = np.dot(X.T, X)
    # Use inv if XTX is full rank, otherwise use pinv
    # inv is computationally more efficient for full-rank matrices as it avoids the additional computations required for the pseudo-inverse.
    if np.linalg.matrix_rank(XTX) == XTX.shape[0]:
        XTX_inv = np.linalg.inv(XTX)
    else:
        XTX_inv = np.linalg.pinv(XTX)
    coefficients = np.dot(XTX_inv, X.T)[0]  # Ensure coefficients is 1D
    coefficients = coefficients.flatten()
    y_padded = np.pad(y_copy, (half_window, half_window), mode='edge')
    y_smooth = np.convolve(y_padded, coefficients, mode='valid')
    return y_smooth

def ewma_filter(data, alpha=0.5):
    """
    Exponential Weighted Moving Average (EWMA) filter.

    Parameters:
        data (np.ndarray): Input 1D array.
        alpha (float, optional): Smoothing factor, 0 < alpha <= 1. Default is 0.5.

    Returns:
        np.ndarray: Smoothed array.
    """
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1].")
    if np.all(np.isnan(data)) or np.all(data == 0):
        return np.full_like(data, np.nan)
    # Fill NaNs by linear interpolation
    data_filled = data.copy()
    mask = np.isnan(data_filled)
    if np.any(mask):
        data_filled[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data_filled[~mask])
    # EWMA calculation
    result = np.empty_like(data_filled)
    result[0] = data_filled[0]
    for i in range(1, len(data_filled)):
        result[i] = alpha * data_filled[i] + (1 - alpha) * result[i - 1]
    return result

def butter_highpass_filter(data, cutoff_period, fs, order):
    """
    Butterworth high-pass filter.

    Parameters:
        data (np.ndarray): Input 1D array.
        cutoff_period (float): Cutoff period (in the same units as fs).
        fs (float): Sampling rate.
        order (int): Filter order.

    Returns:
        np.ndarray: Filtered data.
    """
    if np.all(np.isnan(data)) or np.all(data == 0):
        return np.full_like(data, np.nan)
    # Fill NaNs by linear interpolation
    data_filled = data.copy()
    mask = np.isnan(data_filled)
    if np.any(mask):
        data_filled[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data_filled[~mask])
    # Calculate normalized cutoff frequency
    nyquist = 0.5 * fs
    normal_cutoff = 1.0 / cutoff_period / nyquist
    # Ensure cutoff is in (0, 1)
    normal_cutoff = min(max(normal_cutoff, 1e-6), 0.999)
    try:
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        filtered_data = signal.filtfilt(b, a, data_filled)
        return filtered_data
    except Exception:
        return np.full_like(data, np.nan)