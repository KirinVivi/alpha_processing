import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def filter_corr_features_pooled(
    feature_dfs: dict[str, pd.DataFrame],
    ic_series: pd.Series,
    correlation_threshold: float = 0.7
) -> list[str]:
    """
    針對面板數據，使用「池化相關性」方法，根據特徵間的相關性和IC值來篩選特徵。
    
    此函數假設輸入數據是一個字典，其中每個鍵是一個特徵名稱，
    每個值是一個形狀為 (n_dates, n_stocks) 的 DataFrame。

    Args:
        feature_dfs (dict[str, pd.DataFrame]): 一個字典，鍵是特徵名稱，
                                             值是 (n_dates, n_stocks) 的 DataFrame。
        ic_series (pd.Series): 一個 Series，索引是特徵名稱，值是對應的IC值。
        correlation_threshold (float): 相關性閾值，高於此值的特徵將被視為冗餘。

    Returns:
        list[str]: 一個列表，包含最終被保留下來的特徵的名稱。
    """
    logger.info("starting filtering alphas by calculation pooling corr")

    # --- 步驟 1: 將每個特徵的面板數據攤平成一個向量 ---
    logger.info("step 1/4: reconstruct data (needs some time)...")
    
    feature_vectors = {}
    # 使用 tqdm 顯示進度條
    for name, df in tqdm(feature_dfs.items()):
        # 將 (dates, stocks) 矩陣攤平成一個長向量
        feature_vectors[name] = df.values.flatten()

    # 從攤平的向量創建一個大的 DataFrame
    # 它的形狀是 (n_observations, n_features)
    try:
        combined_df = pd.DataFrame(feature_vectors)
        # 釋放不再需要的記憶體
        del feature_vectors
    except ValueError as e:
        logger.error(f"create and merge DataFrame failed: {e}")
        logger.error("please check the length of the df whether are equal")
        return []

    logger.info(f"data recontruction finished! Shape after merging: {combined_df.shape}")
    logger.warning("warning: take a long time to calculate the corr matrix of big data")

    # --- 步驟 2: 計算相關係數矩陣 ---
    logger.info("step 2/4: calculating the corr matrix...")
    correlation_matrix = combined_df.corr().abs()
    # 釋放不再需要的記憶體
    del combined_df

    # --- 步驟 3: 按 IC 絕對值降序排序 ---
    logger.info("step 3/4: sort by ic...")
    sorted_features = ic_series.abs().sort_values(ascending=False).index.tolist()

    # --- 步驟 4: 迭代篩選 ---
    logger.info("step 4/4: starting filtering iteration ...")
    features_to_keep = []
    features_to_discard = set()

    for feature in tqdm(sorted_features):
        if feature not in features_to_discard:
            # 1. 保留這個特徵
            features_to_keep.append(feature)
            
            # 2. 找到與它高度相關的所有其他特徵
            if feature in correlation_matrix:
                correlated_group = correlation_matrix.index[correlation_matrix[feature] > correlation_threshold].tolist()
                
                # 3. 將這些高度相關的特徵加入待剔除集合
                features_to_discard.update(correlated_group)
    
    logger.info(f"filtering finished! From {len(sorted_features)} alphas, remain {len(features_to_keep)} alphas.")
    return features_to_keep

# ### 使用範例
#
# # --- 準備模擬數據 ---
# # 模擬 100 天，20 支股票的數據
# dates = pd.to_datetime(pd.date_range('2023-01-01', periods=100))
# stocks = [f'stock_{i}' for i in range(20)]

# # 創建一個 (100, 20) 的收益率 DataFrame
# ret_df = pd.DataFrame(np.random.randn(100, 20), index=dates, columns=stocks)

# # 創建 5 個特徵，每個特徵都是一個 (100, 20) 的 DataFrame
# # 這是您的數據結構：一個字典，鍵是特徵名，值是 DataFrame
# feature_dfs_dict = {}
# feature_dfs_dict['A'] = pd.DataFrame(ret_df * 0.5 + np.random.randn(100, 20) * 0.1, index=dates, columns=stocks)
# feature_dfs_dict['B'] = pd.DataFrame(feature_dfs_dict['A'] * 0.9 + np.random.randn(100, 20) * 0.05, index=dates, columns=stocks) # B與A高度相關
# feature_dfs_dict['C'] = pd.DataFrame(ret_df * -0.4 + np.random.randn(100, 20) * 0.1, index=dates, columns=stocks)
# feature_dfs_dict['D'] = pd.DataFrame(np.random.randn(100, 20), index=dates, columns=stocks) # D是純噪聲
# feature_dfs_dict['E'] = pd.DataFrame(feature_dfs_dict['C'] * 0.85 + np.random.randn(100, 20) * 0.05, index=dates, columns=stocks) # E與C高度相關

# # 假設我們已經計算好了IC值
# ic_values = {
#     'A': 0.15,  # A 的 IC 最高
#     'B': 0.14,  # B 的 IC 略低於 A
#     'C': -0.12, # C 的 IC 絕對值也很高
#     'D': 0.01,  # D 的 IC 很低
#     'E': -0.11  # E 的 IC 略低於 C
# }
# ic_series = pd.Series(ic_values)

# # --- 執行篩選 ---
# final_features = filter_panel_features_pooled(
#     feature_dfs=feature_dfs_dict,
#     ic_series=ic_series,
#     correlation_threshold=0.8
# )

# logger.info("\n最終保留的特徵:")
# logger.info(final_features)
# 預期輸出: ['A', 'C', 'D']
