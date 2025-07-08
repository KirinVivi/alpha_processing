import numpy as np
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeSeriesProcessor:
    def aroonosc(self, array: np.ndarray, period: int = 4, backend="numpy", device="cpu") -> np.ndarray:
        """
        Calculate Aroon Oscillator for a given period.
        Formula: Aroon Up - Aroon Down
        Aroon Up = (period - index of last highest high) / period * 100
        Aroon Down = (period - index of last lowest low) / period * 100
        Args:
            array (np.ndarray): Input array with shape (n_samples, n_features)
            period (int): Period for Aroon calculation
            backend (str): Backend to use ('numpy' or 'torch')
            device (str): Device to use for torch backend ('cpu' or 'cuda')
        Returns:
            np.ndarray: Aroon Oscillator values with shape (n_samples, n_features)
        """
        logger.info(f"Input shape: {array.shape}, period: {period}, backend: {backend}")
        valid_mask = np.isfinite(array).all(axis=0)
        valid_arr = array[:, valid_mask]
        logger.info(f"Valid array shape after filtering: {valid_arr.shape}")

        if backend == "torch":
            x = torch.from_numpy(valid_arr).double().to(device)  # [T, F]
            T, F = x.shape
            if T < period:
                logger.warning("Time steps less than period, returning NaN array")
                return np.full_like(array, np.nan, dtype=np.float64)

            xw = x.unfold(0, period, 1)  # [T-period+1, F, period]
            idx_high = xw.argmax(dim=2)  # [T-period+1, F]
            idx_low = xw.argmin(dim=2)

            since_high = (period - 1) - idx_high
            since_low = (period - 1) - idx_low

            up = (period - since_high).double() / period * 100
            down = (period - since_low).double() / period * 100
            out = up - down  # [T-period+1, F]
            logger.info(f"PyTorch out shape: {out.shape}")

            pad = torch.full((period-1, F), float('nan'), device=device)
            res = torch.cat([pad, out], dim=0).cpu().numpy()  # [T, F]
            result = np.full_like(array, np.nan, dtype=np.float64)
            result[:, valid_mask] = res
            return result
        else:
            T, F = valid_arr.shape
            result = np.full_like(array, np.nan, dtype=np.float64)
            if T < period:
                logger.warning("Time steps less than period, returning NaN array")
                return result

            sw = np.lib.stride_tricks.sliding_window_view(valid_arr, period, axis=0)
            idx_high = sw.argmax(axis=2)  # [T-period+1, F]
            idx_low = sw.argmin(axis=2)

            since_high = (period - 1) - idx_high
            since_low = (period - 1) - idx_low

            up = (period - since_high).astype(np.float64) / period * 100
            down = (period - since_low).astype(np.float64) / period * 100
            out = up - down  # [T-period+1, F]
            logger.info(f"NumPy out shape: {out.shape}")

            result[period-1:, valid_mask] = out
            return result

# 测试代码
if __name__ == "__main__":
    processor = TimeSeriesProcessor()
    # 测试数据
    array = np.array([[10, 20], [15, 18], [12, 22], [17, 19], [14, 21]], dtype=float)
    period = 3

    # NumPy 后端
    numpy_result = processor.aroonosc(array, period=period, backend="numpy")
    print("NumPy 结果:\n", numpy_result)

    # PyTorch 后端
    torch_result = processor.aroonosc(array, period=period, backend="torch", device="cpu")
    print("PyTorch 结果:\n", torch_result)

    # 比较结果
    print("结果是否相同:", np.array_equal(numpy_result, torch_result, equal_nan=True))
    print("结果是否接近:", np.allclose(numpy_result, torch_result, rtol=1e-5, equal_nan=True))