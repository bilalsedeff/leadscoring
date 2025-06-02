"""
Inference-pipeline  –  loads trained artefacts and returns
p_conv + segment label   (Low / Medium / High)

▪ CLI argparser  (--csv  |  --stdin  |  --json)
▪ account_id / lead_id otomatik geçerlenir – drop edilmeyecek
  ama model inputuna girmez (preprocessor already knows to drop)
▪ prediction sonuçları  outputs/<timestamp>/predict/  altına JSON
  ve (isteğe bağlı) CSV olarak yazılır.
"""

from __future__ import annotations
import argparse, json, sys, os, datetime as dt
from pathlib import Path
import pandas as pd
import joblib
from src.utils.logger import get_logger
from src.preprocessing.cleaning import BasicCleaner
from src.preprocessing.type_cast import apply_type_map
from src.features.engineering import add_temporal_features
from src.features.interactions import add_pairwise_ratios, add_products
from src.features.auto_selector import SmartFeatureSelector

LOG = get_logger("predict")


# ------------------------------------------------------------------ #
# artefact loader
# ------------------------------------------------------------------ #
def _load_artifacts(dir_path: Path):
    """
    Kayıtlı model ve preprocessor'ları yükler
    
    Args:
        dir_path: Modelin bulunduğu dizin
        
    Returns:
        (preprocessor, model, calibrator, binner, meta): Yüklenen model bileşenleri
    """
    LOG.info(f"Artefactlar yükleniyor: {dir_path}")
    
    # Preprocessor
    preproc_path = dir_path / "preproc.pkl"
    if not preproc_path.exists():
        raise FileNotFoundError(f"Preprocessor bulunamadı: {preproc_path}")
    
    # Model
    model_path = dir_path / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model bulunamadı: {model_path}")
    
    # Calibrator (isteğe bağlı)
    cal_path = dir_path / "calibrator.pkl"
    cal = None
    if cal_path.exists():
        cal = joblib.load(cal_path)
    
    # Binner (isteğe bağlı)
    bin_path = dir_path / "binner.pkl"
    binr = None
    if bin_path.exists():
        binr = joblib.load(bin_path)
    
    # Auto-selector (isteğe bağlı)
    selector_path = dir_path / "auto_selector.pkl"
    auto_selector = None
    if selector_path.exists():
        auto_selector = joblib.load(selector_path)
    
    # Metadata (isteğe bağlı)
    meta_path = dir_path / "metadata.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
    
    # Temizleyici (isteğe bağlı)
    if "cleaner" in meta:
        cleaner = meta["cleaner"]
    else:
        cleaner_path = dir_path / "cleaner.pkl"
        if cleaner_path.exists():
            cleaner = joblib.load(cleaner_path)
            meta["cleaner"] = cleaner
    
    # Meta bilgilere auto_selector'ı ekle
    if auto_selector:
        meta["auto_selector"] = auto_selector
    
    return joblib.load(preproc_path), joblib.load(model_path), cal, binr, meta


# ------------------------------------------------------------------ #
# predict-one DataFrame
# ------------------------------------------------------------------ #
def predict_df(df: pd.DataFrame, artefact_dir: Path) -> pd.DataFrame:
    pre, mdl, cal, binr, meta = _load_artifacts(artefact_dir)
    
    # Tahmin öncesi preprocessing adımları
    # 1. Temizleme
    if "cleaner" in meta:
        cleaner = meta["cleaner"]
        df_clean = cleaner.transform(df)
    else:
        # Eğer kaydedilmiş temizleyici yoksa, varsayılan temizleyici kullan
        LOG.info("Kaydedilmiş temizleyici bulunamadı, varsayılan temizleyici kullanılıyor")
        cleaner = BasicCleaner()
        df_clean = cleaner.transform(df)
    
    # 2. Veri tipi dönüşümü
    LOG.info("Veri tipi dönüşümleri uygulanıyor...")
    df_typed = apply_type_map(df_clean)
    
    # 3. Temporal özellikler ekle
    LOG.info("Temporal özellikler ekleniyor...")
    df_temp = add_temporal_features(df_typed)
    
    # 4. Özellik etkileşimleri ekle
    LOG.info("Özellik etkileşimleri ekleniyor...")
    df_inter = add_pairwise_ratios(df_temp)
    
    # 5. Akıllı özellik seçimi uygula (eğer meta içinde varsa)
    if "auto_selector" in meta:
        LOG.info("Akıllı özellik seçimi uygulanıyor...")
        auto_selector = meta["auto_selector"]
        df_features = auto_selector.transform(df_inter)
    else:
        df_features = df_inter
        LOG.info("Akıllı özellik seçimi bulunamadı, tüm özellikler kullanılıyor")
    
    # 6. Preprocessor uygula
    LOG.info("Preprocessor uygulanıyor...")
    try:
        # ID kolonlarını düşür
        id_cols = ["LeadId", "account_Id"]
        id_values = {}
        for col in id_cols:
            if col in df_features.columns:
                id_values[col] = df_features[col].copy()
                df_features = df_features.drop(columns=[col])
        
        # Target kolonu varsa düşür
        target_col = "Target_IsConverted"
        target_values = None
        if target_col in df_features.columns:
            target_values = df_features[target_col].copy()
            df_features = df_features.drop(columns=[target_col])
        
        # Veri tiplerini kontrol et ve düzelt
        for col in df_features.columns:
            if df_features[col].dtype == 'object':
                # Kategorik kolonları string'e çevir
                df_features[col] = df_features[col].astype(str)
        
        # Preprocessor uygula
        X_transformed = pre.transform(df_features)
        
        # Model tahminlerini al
        LOG.info("Model tahminleri yapılıyor...")
        probs = mdl.predict_proba(X_transformed)[:, 1]
        
        # Kalibre et (varsa)
        if cal:
            LOG.info("Olasılıklar kalibre ediliyor...")
            probs = cal.predict_proba(probs.reshape(-1, 1))[:, 1]
        
        # Segment etiketleri ekle (varsa)
        result = pd.DataFrame({"prediction_prob": probs})
        if binr:
            LOG.info("Segment etiketleri ekleniyor...")
            result["segment"] = binr.transform(probs)
        
        # ID kolonlarını geri ekle
        for col, values in id_values.items():
            result[col] = values.values
        
        # Target kolonu varsa, gerçek değerlerle karşılaştırma metrikleri ekle
        if target_values is not None:
            result["actual"] = target_values.values
            
            # Basit değerlendirme metrikleri
            from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
            try:
                result.attrs["roc_auc"] = roc_auc_score(target_values, probs)
                result.attrs["avg_precision"] = average_precision_score(target_values, probs)
                result.attrs["f1_score"] = f1_score(target_values, (probs >= 0.5).astype(int))
                LOG.info(f"Değerlendirme metrikleri: ROC-AUC={result.attrs['roc_auc']:.4f}, "
                        f"Avg. Precision={result.attrs['avg_precision']:.4f}, "
                        f"F1={result.attrs['f1_score']:.4f}")
            except Exception as e:
                LOG.warning(f"Değerlendirme metrikleri hesaplanamadı: {e}")
        
        return result
    
    except Exception as e:
        LOG.error(f"Tahmin sırasında hata: {e}")
        raise


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
def _parse_cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Lead-Scoring Inference")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--csv", help="input CSV path")
    g.add_argument("--stdin", action="store_true",
                   help="JSON line read from stdin")
    g.add_argument("--json", help="single JSON file")
    ap.add_argument("--artifacts", required=True, help="artefact directory")
    ap.add_argument("--out", help="output directory (default: ./outputs)")
    return ap.parse_args()


def _load_input(args) -> pd.DataFrame:
    if args.csv:
        return pd.read_csv(args.csv)
    if args.json:
        with open(args.json) as f:
            data = json.load(f)
        return pd.DataFrame(data if isinstance(data, list) else [data])
    if args.stdin:
        data = json.loads(sys.stdin.read())
        return pd.DataFrame([data])
    raise RuntimeError("No input")


def main():
    args = _parse_cli()
    artefact_dir = Path(args.artifacts)
    df_in = _load_input(args)

    res = predict_df(df_in, artefact_dir)

    # outputs/<timestamp>/predict/
    out_root = Path(args.out or "outputs")
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / ts / "predict"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "predictions.csv"
    out_json = out_dir / "predictions.json"
    res.to_csv(out_csv, index=False)
    res.to_json(out_json, orient="records", lines=False)

    LOG.info("✓ predictions saved to %s", out_dir)
    print(res.to_json(orient="records"))


if __name__ == "__main__":
    main()
