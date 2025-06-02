"""
Tip dönüştürücü – YAML dtypes tablosuna göre dataframe sütunlarını cast eder
"""
from omegaconf import OmegaConf
import pandas as pd
import numpy as np

def apply_type_map(df: pd.DataFrame, cfg_path="configs/data.yaml") -> pd.DataFrame:
    cfg = OmegaConf.load(cfg_path)
    dtype_map = cfg.dtypes
    for col, dtype in dtype_map.items():
        if col in df.columns:
            if dtype == "category":
                df[col] = df[col].astype("category")
            else:
                df[col] = df[col].astype(dtype)
    # Datetime cast
    for dt in cfg.datetime_cols:
        if dt in df.columns:
            df[dt] = pd.to_datetime(df[dt], errors="coerce")
    # Convert NULLs to NaNs
    df = df.replace({None: np.nan})
    # Convert 'NULL's to NaNs
    df = df.replace("NULL", np.nan)
    return df
