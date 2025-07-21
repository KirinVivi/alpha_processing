import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def filter_features_weighted_greedy_independent_set(
    feature_dfs: dict[str, pd.DataFrame],
    ic_series: pd.Series,
    correlation_threshold: float = 0.7
) -> tuple[list[str], pd.Series]:
    """
    使用「IC权重优化的贪心独立集」算法来筛选特征。

    此算法优先选择 IC 与度的“性价比”最高的特征。
    """
    logger.info("Starting filtering using IC-weighted greedy independent set method.")

    # --- 步骤 1 & 2: 构造相关性图 (与上一版图论方法相同) ---
    logger.info("Step 1/2: Building correlation graph...")
    try:
        feature_vectors = {name: df.values.flatten() for name, df in feature_dfs.items()}
        combined_df = pd.DataFrame(feature_vectors)
        correlation_matrix = combined_df.corr()
        del combined_df, feature_vectors
    except ValueError as e:
        logger.error(f"Failed to create DataFrame: {e}. Check shapes.")
        return [], pd.Series()
    
    features = correlation_matrix.columns.tolist()
    adj_matrix = (correlation_matrix.abs() > correlation_threshold)
    G = nx.from_pandas_adjacency(adj_matrix)
    
    # --- 步骤 3: 实现IC权重优化的贪心算法 ---
    logger.info("Step 2/2: Applying weighted greedy selection...")

    # 创建一个图的副本进行操作，或只记录节点状态，避免修改原图
    nodes_to_process = set(G.nodes())
    degrees = {node: G.degree(node) for node in nodes_to_process}
    
    features_to_keep = []

    pbar = tqdm(total=len(nodes_to_process), desc="Filtering Nodes")
    while nodes_to_process:
        # 计算所有剩余节点的“性价比”分数
        scores = {
            node: abs(ic_series.get(node, 0)) / (degrees[node] + 1)
            for node in nodes_to_process
        }
        
        # 找到分数最高的节点
        best_node = max(scores, key=scores.get)
        
        # 1. 保留这个“性价比”最高的节点
        features_to_keep.append(best_node)
        
        # 2. 从待处理集合中移除它和它的所有邻居
        nodes_to_remove = {best_node} | set(G.neighbors(best_node))
        
        # 高效地移除已处理的节点
        processed_nodes_in_this_step = nodes_to_process.intersection(nodes_to_remove)
        nodes_to_process.difference_update(processed_nodes_in_this_step)
        
        pbar.update(len(processed_nodes_in_this_step))

    pbar.close()

    final_kept_ic = ic_series[features_to_keep]

    logger.info(f"Filtering finished! From {len(features)} alphas, {len(features_to_keep)} remain.")
    
    return features_to_keep, final_kept_ic