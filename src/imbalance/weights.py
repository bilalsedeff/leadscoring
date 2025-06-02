"""
Üst düzey etkileşim & oran özellikleri
"""
import pandas as pd
from itertools import combinations

def add_pairwise_ratios(df: pd.DataFrame, num_cols, max_pairs=20):
    df = df.copy()
    for (a, b) in combinations(num_cols, 2):
        if len(df.columns) >= len(num_cols) + max_pairs:
            break
        ratio_name = f"{a}_DIV_{b}"
        # 0 bölünme koruması
        df[ratio_name] = df[a] / (df[b].replace(0, 1e-5))
    return df

def add_products(df: pd.DataFrame, num_cols, top_k=10):
    df = df.copy()
    for a, b in combinations(num_cols[:top_k], 2):
        df[f"{a}_X_{b}"] = df[a] * df[b]
    return df
