import pandas as pd
import numpy as np
from tqdm import tqdm

def _calculate_enb(pnl_df: pd.DataFrame) -> float:
    """
    計算給定 PnL DataFrame 的有效投注數 (Effective Number of Bets)。
    """
    # 1. 計算 PnL 的相關係數矩陣
    correlation_matrix = pnl_df.corr()
    
    # 2. 計算特徵值
    eigenvalues = np.linalg.eigvalsh(correlation_matrix)
    
    # 3. 將特徵值歸一化，得到每個主成分的權重
    # 處理負的特徵值（可能因數值不穩定產生）
    eigenvalues[eigenvalues < 0] = 0
    weights = eigenvalues / eigenvalues.sum()
    
    # 4. 計算 Shannon 熵
    # 忽略權重為 0 的項，避免 log(0)
    weights = weights[weights > 1e-12]
    shannon_entropy = -np.sum(weights * np.log(weights))
    
    # 5. ENB = exp(H)
    return np.exp(shannon_entropy)

def select_features_by_max_entropy(
    pnl_dfs: dict[str, pd.DataFrame],
    num_to_select: int,
    initial_feature: str = None
) -> list[str]:
    """
    使用貪婪前向選擇和最大化 ENB 的方法，從一組 PnL 中篩選出最多樣化的特徵組合。

    Args:
        pnl_dfs (dict[str, pd.DataFrame]): 一個字典，鍵是特徵名稱，
                                           值是包含單一 PnL 列的 DataFrame。
        num_to_select (int): 最終想要保留的特徵數量。
        initial_feature (str, optional): 可選的初始特徵。如果未提供，
                                         將選擇波動率最低的 PnL 作為起點。

    Returns:
        list[str]: 一個列表，包含最終被保留下來的、最多樣化的特徵的名稱。
    """
    print("開始進行最大熵特徵篩選...")

    # --- 步驟 1: 合併所有 PnL DataFrame ---
    print("步驟 1/3: 合併 PnL 數據...")
    try:
        combined_pnl_df = pd.concat(
            {name: df.iloc[:, 0] for name, df in pnl_dfs.items()},
            axis=1
        )
    except Exception as e:
        print(f"合併 DataFrame 時出錯: {e}")
        return []

    all_features = combined_pnl_df.columns.tolist()
    if num_to_select >= len(all_features):
        print("要選擇的數量大於等於總特徵數，將返回所有特徵。")
        return all_features
        
    # --- 步驟 2: 確定第一個基石特徵 ---
    print("步驟 2/3: 選擇初始特徵...")
    if initial_feature and initial_feature in all_features:
        selected_features = [initial_feature]
        print(f"使用指定的初始特徵: {initial_feature}")
    else:
        # 如果未指定，選擇一個穩健的起點，例如波動率最低（最穩定）的那個
        pnl_volatility = combined_pnl_df.std()
        initial_feature = pnl_volatility.idxmin()
        selected_features = [initial_feature]
        print(f"未指定初始特徵，選擇波動率最低的特徵作為起點: {initial_feature}")

    remaining_features = [f for f in all_features if f not in selected_features]

    # --- 步驟 3: 迭代選擇剩餘的特徵 ---
    print("步驟 3/3: 開始貪婪前向選擇...")
    for _ in tqdm(range(num_to_select - 1), desc="選擇多樣化特徵"):
        best_next_feature = None
        max_enb = -1

        # 遍歷所有尚未被選擇的特徵
        for candidate_feature in remaining_features:
            # 嘗試將候選特徵加入組合
            current_selection = selected_features + [candidate_feature]
            
            # 計算加入後的 ENB
            current_enb = _calculate_enb(combined_pnl_df[current_selection])
            
            # 如果這個候選特徵能帶來更高的 ENB，就記住它
            if current_enb > max_enb:
                max_enb = current_enb
                best_next_feature = candidate_feature
        
        # 如果找到了最佳的下一個特徵
        if best_next_feature:
            selected_features.append(best_next_feature)
            remaining_features.remove(best_next_feature)
            # print(f"  - 已選擇 {len(selected_features)}/{num_to_select}: {best_next_feature}, 當前 ENB: {max_enb:.2f}")
        else:
            # 如果沒有更多可選的特徵，提前結束
            print("沒有更多可選的特徵來增加 ENB。")
            break

    print(f"\n篩選完成！最終選擇了 {len(selected_features)} 個特徵。")
    return selected_features

### 使用範例


# --- 準備模擬數據 ---
# 模擬 100 天的 PnL 數據
dates = pd.to_datetime(pd.date_range('2023-01-01', periods=100))

# 創建 5 個特徵的 PnL
pnl_dfs_dict = {}
# 因子 A: 基礎 PnL
pnl_dfs_dict['pnl_A'] = pd.DataFrame(np.random.randn(100).cumsum(), index=dates, columns=['pnl'])
# 因子 B: 與 A 高度相關
pnl_dfs_dict['pnl_B'] = pd.DataFrame(pnl_dfs_dict['pnl_A']['pnl'] * 0.9 + np.random.randn(100) * 0.1, columns=['pnl'])
# 因子 C: 與 A 幾乎無關
pnl_dfs_dict['pnl_C'] = pd.DataFrame(np.random.randn(100).cumsum(), index=dates, columns=['pnl'])
# 因子 D: 與 A 負相關
pnl_dfs_dict['pnl_D'] = pd.DataFrame(-pnl_dfs_dict['pnl_A']['pnl'] * 0.8 + np.random.randn(100) * 0.2, columns=['pnl'])
# 因子 E: 另一個獨立的 PnL
pnl_dfs_dict['pnl_E'] = pd.DataFrame(np.random.randn(100).cumsum(), index=dates, columns=['pnl'])

# --- 執行篩選 ---
# 我們希望從 5 個 PnL 中選出最多樣化的 3 個
# 假設我們已經知道 pnl_A 是最好的單個 PnL，將其作為起點
final_pnl_features = select_features_by_max_entropy(
    pnl_dfs=pnl_dfs_dict,
    num_to_select=3,
    initial_feature='pnl_A'
)

print("\n最終保留的 PnL 特徵:")
print(final_pnl_features)

# 預期輸出分析：
# 1. 初始選擇 pnl_A。
# 2. 接下來，算法會發現加入 pnl_C 或 pnl_E 能最大程度地增加多樣性（ENB），因為它們與 A 不相關。
# 3. 而加入 pnl_B 或 pnl_D 幾乎不會增加多樣性，因為它們與 A 高度相關（正或負）。
# 4. 因此，最終的結果很可能是 ['pnl_A', 'pnl_C', 'pnl_E'] 或類似的組合。
