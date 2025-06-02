#!/usr/bin/env python
"""
Lead Scoring Pipeline CLI
=========================

Bu CLI uygulaması, Lead Scoring pipeline'ını çalıştırmak için kullanılır.
İstatistiksel analizler, model eğitimi, feature importance ve diğer 
tüm işlemleri yönetir.

Kullanım:
    python lead_scoring.py [KOMUT] [SEÇENEKLER]

Komutlar:
    setup         Gerekli paketleri kurar
    run           Pipeline'ı çalıştırır
    stats         İstatistiksel analizler yapar
    feature-imp   Feature importance hesaplar
    auto-select   Akıllı özellik seçimi yapar
    train         Model eğitir
    predict       Tahmin yapar
    dashboard     Streamlit dashboard'ı başlatır
    mlflow        MLflow UI'ı başlatır
    all           Tüm pipeline adımlarını çalıştırır
"""

import os
import sys
import click
import subprocess
import webbrowser
import time
import yaml
from pathlib import Path
import json
from datetime import datetime
import logging

# CLI kök dizinini belirle
CLI_ROOT = Path(__file__).resolve().parent
sys.path.append(str(CLI_ROOT))

# utils/paths modülünü içe aktar - CLI_ROOT yerine paths.py'deki PROJECT_ROOT'u kullan
from src.utils.paths import PROJECT_ROOT, get_experiment_dir, get_output_base_dir
from src.utils.logger import get_logger

# Logger
logger = get_logger("lead_scoring_cli")

# app_config.yaml'dan ayarları yükle
def _load_config():
    """app_config.yaml'dan yapılandırmayı yükler."""
    config_path = PROJECT_ROOT / "configs" / "app_config.yaml"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Yapılandırma dosyası yüklenemedi: {e}")
    
    # Varsayılan değerler
    return {
        "mlflow_port": 5000,
        "streamlit_port": 8501,
        "output_dir": "outputs",
        "tmp_dir": "tmp",
        "archive_days": 30,
        "log_level": "INFO"
    }

# Yapılandırmayı yükle
CONFIG = _load_config()

# Sabitler - app_config.yaml'dan alınır
MLFLOW_PORT = CONFIG.get("mlflow_port", 5000)
STREAMLIT_PORT = CONFIG.get("streamlit_port", 8501)
OUTPUT_DIR = CONFIG.get("output_dir", "outputs")

# Proje kök dizinini ve çıktı dizinini ortam değişkenlerine ekle
def _setup_environment(output_dir=None):
    """CLI komutları için ortam değişkenlerini ayarlar."""
    os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)
    
    if output_dir:
        os.environ["OUTPUT_DIR"] = str(output_dir)
    elif "OUTPUT_DIR" not in os.environ:
        # Varsayılan olarak app_config.yaml'dan alınan değeri kullan
        os.environ["OUTPUT_DIR"] = str(Path(OUTPUT_DIR))
    
    # Çıktı dizininin varlığını kontrol et ve oluştur
    Path(os.environ["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)
    
    return os.environ["OUTPUT_DIR"]

@click.group()
def cli():
    """Lead Scoring Pipeline CLI"""
    pass

@cli.command()
@click.option('--force', is_flag=True, help='Kurulu paketleri yeniden yükle')
def setup(force):
    """Gerekli paketleri kurar"""
    _setup_environment()
    requirements_file = PROJECT_ROOT / "requirements.txt"
    
    if not requirements_file.exists():
        click.echo("requirements.txt dosyası bulunamadı!")
        return
    
    click.echo("Gerekli paketler yükleniyor...")
    
    pip_cmd = [sys.executable, "-m", "pip", "install"]
    if force:
        pip_cmd.append("--force-reinstall")
    
    pip_cmd.extend(["-r", str(requirements_file)])
    
    try:
        subprocess.run(pip_cmd, check=True, env=os.environ)
        click.echo("Kurulum tamamlandı!")
    except subprocess.CalledProcessError as e:
        click.echo(f"Kurulum sırasında hata oluştu: {e}")
        sys.exit(1)

@cli.command()
@click.option('--split_method', type=click.Choice(['time', 'random']), default='time',
             help='Bölme metodu (zaman veya rastgele)')
@click.option('--target_col', default='Target_IsConverted', 
             help='Hedef değişken')
@click.option('--time_col', default=None, 
             help='Zaman kolonu')
@click.option('--train_cutoff', default=None, 
             help='Eğitim seti kesim noktası (YYYYMM)')
@click.option('--val_cutoff', default=None, 
             help='Validation seti kesim noktası (YYYYMM)')
@click.option('--test_cutoff', default=None, 
             help='Test seti kesim noktası (YYYYMM)')
@click.option('--output_dir', 
             help='Çıktı dizini')
@click.option('--drop_id_cols/--keep_id_cols', default=False,
             help='Kimlik kolonlarını düşür (LeadId, account_Id vb.)')
@click.option('--create_new_run', is_flag=True, help='Yeni bir run klasörü oluştur')
def run(split_method, target_col, time_col, train_cutoff, val_cutoff, test_cutoff, output_dir, drop_id_cols, create_new_run):
    """Veri hazırlama, bölme ve temel özellikleri oluşturur."""
    # Yeni run dizini oluştur
    if create_new_run:
        run_dir = Path(f"outputs/run_{int(time.time())}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Log dizini oluştur
        logs_dir = run_dir / "logs"
        os.makedirs(logs_dir, exist_ok=True)
        
        # Log yönlendirmesi
        log_file = logs_dir / "run.log"
        log_handler = logging.FileHandler(log_file)
        log_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s'))
        logging.getLogger().addHandler(log_handler)
        
        # Split dizini
        split_dir = None
        if not output_dir:
            split_dir = run_dir / "split"
            output_dir = split_dir
    else:
        run_dir = None
        split_dir = None
    
    click.echo(f"Veri bölme ve hazırlama işlemi başlatılıyor...")
    if run_dir:
        click.echo(f"Çıktı dizini: {run_dir}")
    
    # Split pipeline çağır
    cmd = ["python", "-m", "src.pipelines.split"]
    
    if train_cutoff:
        cmd.extend(["--train-cutoff", train_cutoff])
    if val_cutoff:
        cmd.extend(["--val-cutoff", val_cutoff])
    if test_cutoff:
        cmd.extend(["--test-cutoff", test_cutoff])
    if time_col:
        cmd.extend(["--time-col", time_col])
    
    cmd.extend(["--group-col", "account_Id"])
    
    if output_dir:
        cmd.extend(["--output-dir", str(output_dir)])
    
    cmd.extend(["--force-balance"])
    
    try:
        subprocess.run(cmd, check=True, env=os.environ)
    except subprocess.CalledProcessError as e:
        click.echo(f"Veri hazırlama ve split işleminde hata: {e}")
        sys.exit(1)
    
    click.echo("Veri hazırlama ve split işlemi tamamlandı.")
    
    # Metadata dosyasını güncelle
    if create_new_run:
        try:
            metadata = {
                "run_id": int(time.time()),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "split_method": split_method,
                "train_cutoff": train_cutoff,
                "val_cutoff": val_cutoff,
                "test_cutoff": test_cutoff,
                "time_col": time_col,
                "target_col": target_col,
                "drop_id_cols": drop_id_cols,
                "create_new_run": create_new_run,
                "output_dir": str(run_dir) if run_dir else None
            }
            
            with open(run_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            click.echo(f"Metadata oluşturulurken hata: {e}")
    
    return run_dir

def find_latest_run_dir():
    """
    En son oluşturulmuş run dizinini bulur.
    
    Returns:
        Path: En son run dizini
    """
    # outputs/run_* dizinlerini listele
    run_dirs = sorted(Path("outputs").glob("run_*"), reverse=True)
    
    if not run_dirs:
        click.echo("Herhangi bir run dizini bulunamadı!")
        return None
    
    return run_dirs[0]

@cli.command()
@click.option('-t', '--test_type', type=click.Choice(['chi_square', 't_test', 'anova', 'conversion_rate', 'correlation', 'all']),
              help='İstatistiksel test tipi')
@click.option('-c', '--categorical_cols', help='Kategorik değişkenler (virgülle ayrılmış)')
@click.option('-n', '--numeric_cols', help='Sayısal değişkenler (virgülle ayrılmış)')
@click.option('-g', '--group_col', help='Grup değişkeni (t-test ve ANOVA için)')
@click.option('--target_col', help='Hedef değişken')
@click.option('--corr_method', type=click.Choice(['pearson', 'spearman', 'kendall']), default='pearson',
              help='Korelasyon yöntemi')
@click.option('--output_dir', help='Çıktıların kaydedileceği dizin')
@click.option('--run_id', help='Kullanılacak run ID (latest: en son run)')
@click.option('--use_train_only', is_flag=True, help='Sadece eğitim verisini kullan')
@click.option('--view', is_flag=True, help='Sonuçları Streamlit ile görüntüle')
@click.option('--auto_split/--no_auto_split', default=True, help='Split bulunamazsa otomatik olarak split oluştur')
def stats(test_type, categorical_cols, numeric_cols, group_col, target_col, corr_method, output_dir, run_id, use_train_only, view, auto_split):
    """İstatistiksel analizler yapar"""
    # Eğer output_dir belirtilmemişse ve run_id varsa, ilgili run dizini altında çalış
    if not output_dir and run_id:
        if run_id.lower() == 'latest':
            latest_run = find_latest_run_dir()
            if latest_run:
                run_dir = latest_run
                click.echo(f"En son run dizini kullanılıyor: {run_dir}")
            else:
                click.echo("Herhangi bir run dizini bulunamadı!")
                return
        else:
            run_dir = Path(f"outputs/run_{run_id}")
            if not run_dir.exists():
                click.echo(f"Run dizini bulunamadı: {run_dir}")
                return
        
        # İstatistiksel testler dizini
        stats_dir = run_dir / "statistical_tests"
        output_dir = stats_dir
    
    # Eğer output_dir hala belirtilmemişse, mevcut experiment dizini altında statistical_tests kullan
    if not output_dir:
        from src.utils.paths import get_experiment_dir
        exp_dir = get_experiment_dir()
        output_dir = exp_dir / "statistical_tests"
    
    # Çıktı dizinini oluştur
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Test verilerini yükle - birden fazla olası konumu kontrol et
    data_path = None
    potential_paths = []
    
    # 1. Run ID ile belirtilen veya en son run dizinindeki split
    if run_id:
        if run_id.lower() == 'latest':
            latest_run = find_latest_run_dir()
            if latest_run:
                potential_paths.append(latest_run / "split" / "train.csv")
        else:
            potential_paths.append(Path(f"outputs/run_{run_id}/split/train.csv"))
    
    # 2. Outputs dizinindeki en son split dizini
    split_dirs = sorted(Path("outputs").glob("*/split"), reverse=True)
    if split_dirs:
        potential_paths.append(split_dirs[0] / "train.csv")
    
    # 3. outputs/split dizini
    potential_paths.append(Path("outputs/split/train.csv"))
    
    # 4. data/processed dizini
    potential_paths.append(Path("data/processed/train.csv"))
    
    # Potansiyel yolları kontrol et
    for path in potential_paths:
        if path.exists():
            data_path = path
            click.echo(f"Veri yolu bulundu: {data_path}")
            break
    
    # Eğer hiçbir veri bulunamazsa ve auto_split aktifse, split işlemini çalıştır
    if data_path is None:
        if auto_split:
            click.echo("Split verisi bulunamadı. Otomatik split işlemi başlatılıyor...")
            try:
                # Split komutunu çalıştır
                from src.utils.paths import reset_experiment_dir
                reset_experiment_dir()  # Yeni bir run dizini oluştur
                
                # Run komutunu doğrudan çağırmak yerine subprocess ile çağır
                try:
                    split_cmd = [sys.executable, "-m", "src.pipelines.split"]
                    # Kesme noktalarını ayarla (varsayılan değerler kullanılacak)
                    split_cmd.extend(["--train-cutoff", "2024-06-30", "--val-cutoff", "2024-11-30", "--test-cutoff", "2025-04-30"])
                    # Split işlemini çalıştır
                    subprocess.run(split_cmd, check=True, env=os.environ)
                    
                    # Yeni oluşturulan split verilerini kontrol et
                    if Path("outputs/split/train.csv").exists():
                        data_path = Path("outputs/split/train.csv")
                        click.echo(f"Split işlemi başarılı. Veri yolu: {data_path}")
                    else:
                        # Alternatif dizinleri kontrol et
                        from src.utils.paths import get_experiment_dir
                        split_dir = get_experiment_dir() / "split"
                        if (split_dir / "train.csv").exists():
                            data_path = split_dir / "train.csv"
                            click.echo(f"Split işlemi başarılı. Veri yolu: {data_path}")
                        else:
                            click.echo("Split işlemi tamamlandı ancak train.csv bulunamadı.")
                            return
                except Exception as e:
                    click.echo(f"Split işlemi sırasında hata: {e}")
                    return
            except Exception as e:
                click.echo(f"Split işlemi sırasında hata: {e}")
                return
        else:
            click.echo("Herhangi bir split verisi bulunamadı! Lütfen önce split işlemi yapın veya --auto_split kullanın.")
            return
    
    click.echo(f"Veri yolu: {data_path}")
    
    # Timestamp ekle (çıktı klasörü benzersiz olsun)
    timestamp = int(time.time())
    unique_dir = output_dir / f"{test_type}_{timestamp}_{'train_only' if use_train_only else 'all_data'}"
    os.makedirs(unique_dir, exist_ok=True)
    
    # İstatistiksel analiz modülünü çağır
    cmd = ["python", "-m", "src.pipelines.statistical_tests"]
    
    # Test tipini ayarla
    if test_type:
        cmd.append(f"--test_type={test_type}")
    
    # Verileri ekle - data_path değil train_path kullanılmalı
    cmd.append(f"--train_path={data_path}")
    
    # Kategorik değişkenleri ekle
    if categorical_cols:
        cmd.append(f"--categorical_cols={categorical_cols}")
    
    # Sayısal değişkenleri ekle
    if numeric_cols:
        cmd.append(f"--numeric_cols={numeric_cols}")
    
    # Grup değişkenini ekle
    if group_col:
        cmd.append(f"--group_col={group_col}")
    
    # Hedef değişkeni ekle
    if target_col:
        cmd.append(f"--target_col={target_col}")
    
    # Korelasyon yöntemini ekle
    if corr_method:
        cmd.append(f"--corr_method={corr_method}")
    
    # Çıktı dizinini ekle
    cmd.append(f"--output_subdir={unique_dir.name}")
    
    # Sadece eğitim verisini kullan
    if use_train_only:
        cmd.append("--use_train_only")
    else:
        cmd.append("--use_all_data")
    
    # Görüntüle
    if view:
        cmd.append("--view")
    
    # Komutu çalıştır
    click.echo(f"İstatistiksel analizler başlatılıyor... Test tipi: {test_type}")
    subprocess.run(cmd, check=True, env=os.environ)
    
    click.echo(f"İstatistiksel analizler tamamlandı. Çıktılar: {unique_dir}")
    
    return unique_dir

@cli.command()
@click.option('--method', '-m', 
              type=click.Choice(['shap', 'permutation', 'both']), 
              default='shap', 
              help='Feature importance hesaplama metodu')
@click.option('--cv/--no_cv', default=True, help='Cross-validation kullan')
@click.option('--n_folds', default=5, help='Cross-validation fold sayısı')
@click.option('--top_k', type=int, help='Seçilecek feature sayısı')
@click.option('--stability_threshold', type=float, default=0.6, 
             help='Kararlılık eşiği (0-1 arası)')
@click.option('--output_dir', help='Çıktıların kaydedileceği dizin')
@click.option('--view', is_flag=True, help='Sonuçları Streamlit ile görüntüle')
@click.option('--preprocess/--no_preprocess', default=True, 
             help='Veriyi ön işleme uygula')
@click.option('--filter_features/--no_filter_features', default=True, 
             help='Özellik seçiminde akıllı filtreleme kullan')
@click.option('--interactive/--no_interactive', default=False, 
             help='Etkileşimli mod (figürleri ekranda göster)')
def feature_imp(method, cv, n_folds, top_k, stability_threshold, output_dir, view,
               preprocess, filter_features, interactive):
    """Feature importance hesaplar"""
    # Mevcut run klasörünü kullan
    from src.utils.paths import get_experiment_dir
    output_dir = _setup_environment(output_dir)
    
    cmd = [sys.executable, "-m", "src.pipelines.feature_importance", 
           f"--method={method}"]
    
    if not cv:
        cmd.append("--no-cv")
    
    cmd.append(f"--n-folds={n_folds}")
    
    if top_k:
        cmd.append(f"--top-k={top_k}")
    
    cmd.append(f"--stability-threshold={stability_threshold}")
    
    # Yeni parametreleri ekle
    if not preprocess:
        cmd.append("--no-preprocess")
        
    if not filter_features:
        cmd.append("--no-filter-features")
        
    if interactive:
        cmd.append("--interactive")
    
    # output_dir parametresini script'e geç - mevcut run klasörü kullanılacak
    exp_dir = get_experiment_dir()
    cmd.append(f"--output-dir={exp_dir}/feature_importance")
    
    click.echo(f"Feature importance hesaplanıyor... (Metod: {method})")
    click.echo(f"Parametreler: CV={cv}, Fold={n_folds}, Stability={stability_threshold}, Preprocess={preprocess}, Filter={filter_features}")
    click.echo(f"Çıktı dizini: {exp_dir}/feature_importance")
    
    try:
        subprocess.run(cmd, check=True, env=os.environ)
        click.echo("Feature importance başarıyla hesaplandı!")
        
        if view:
            _start_streamlit()
    except subprocess.CalledProcessError as e:
        click.echo(f"Feature importance hesaplanırken hata oluştu: {e}")
        sys.exit(1)

@cli.command()
@click.option('--model_type', '-m', 
              type=click.Choice(['1', '2', '3', '4']), 
              default='4', 
              help='Model tipi: 1=Baseline, 2=LightGBM, 3=Source-based, 4=Ensemble')
@click.option('--experiment_name', help='MLflow experiment adı')
@click.option('--output_dir', help='Çıktıların kaydedileceği dizin')
@click.option('--view', is_flag=True, help='Sonuçları MLflow ile görüntüle')
def train(model_type, experiment_name, output_dir, view):
    """Model eğitir"""
    # Mevcut run klasörünü kullan
    from src.utils.paths import get_experiment_dir
    output_dir = _setup_environment(output_dir)
    
    if not experiment_name:
        experiment_name = f"model_{model_type}_{int(time.time())}"
    
    # paths.py'deki get_experiment_dir() fonksiyonu kullanılacak
    exp_dir = get_experiment_dir()
    
    cmd = ["python", "-m", "src.pipelines.train", 
           f"+model={model_type}", 
           f"++experiment_name={experiment_name}"]
    
    click.echo(f"Model eğitimi başlatılıyor... (Model Tipi: {model_type}, Experiment: {experiment_name})")
    click.echo(f"Çıktı dizini: {exp_dir}")
    
    try:
        subprocess.run(cmd, check=True, env=os.environ)
        click.echo(f"Model eğitimi başarıyla tamamlandı! (Experiment: {experiment_name})")
        
        if view:
            _start_mlflow()
    except subprocess.CalledProcessError as e:
        click.echo(f"Model eğitimi sırasında hata oluştu: {e}")
        sys.exit(1)

@cli.command()
@click.option('--input_file', '-i', required=True, help='Tahmin edilecek veri dosyası (CSV)')
@click.option('--model_path', '-m', required=True, help='Model dizini')
@click.option('--output_file', '-o', help='Tahmin sonuçlarının kaydedileceği dosya')
def predict(input_file, model_path, output_file):
    """Model ile tahmin yapar"""
    if not os.path.exists(input_file):
        click.echo(f"Girdi dosyası bulunamadı: {input_file}")
        sys.exit(1)
    
    if not os.path.exists(model_path):
        click.echo(f"Model dizini bulunamadı: {model_path}")
        sys.exit(1)
    
    if not output_file:
        output_file = f"predictions_{int(time.time())}.csv"
    
    # Tahmin yapma komutu
    cmd = [sys.executable, "-c", 
          f"import pandas as pd; from src.pipelines.predict import score; df = pd.read_csv('{input_file}'); result = score(df, '{model_path}'); result.to_csv('{output_file}', index=False); print(f'Tahminler {output_file} dosyasına kaydedildi.')"]
    
    click.echo(f"Tahmin yapılıyor...")
    try:
        subprocess.run(cmd, check=True, env=os.environ)
        click.echo(f"Tahmin başarıyla tamamlandı! Sonuçlar: {output_file}")
    except subprocess.CalledProcessError as e:
        click.echo(f"Tahmin sırasında hata oluştu: {e}")
        sys.exit(1)

@cli.command()
@click.option('--port', default=None, help='Streamlit port numarası')
@click.option('--no_browser', is_flag=True, help='Tarayıcı açma')
def dashboard(port, no_browser):
    """Streamlit dashboard'ı başlatır"""
    if port is None:
        port = STREAMLIT_PORT
    
    cmd = ["streamlit", "run", "streamlit_app/app.py", "--server.port", str(port)]
    
    click.echo(f"Streamlit dashboard başlatılıyor (Port: {port})...")
    
    process = subprocess.Popen(cmd, env=os.environ)
    
    time.sleep(2)  # Dashboard'ın başlaması için bekle
    
    if not no_browser:
        webbrowser.open(f"http://localhost:{port}")
        click.echo(f"Dashboard tarayıcıda açıldı: http://localhost:{port}")
    
    click.echo(f"Dashboard çalışıyor: http://localhost:{port}")
    click.echo("Durdurmak için Ctrl+C tuşlarına basın.")
    
    try:
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        click.echo("Dashboard durduruldu.")

@cli.command()
@click.option('--port', default=None, help='MLflow port numarası')
@click.option('--no_browser', is_flag=True, help='Tarayıcı açma')
def mlflow(port, no_browser):
    """MLflow UI'ı başlatır"""
    if port is None:
        port = MLFLOW_PORT
        
    cmd = ["mlflow", "ui", "--port", str(port)]
    
    click.echo(f"MLflow UI başlatılıyor (Port: {port})...")
    
    process = subprocess.Popen(cmd, env=os.environ)
    
    time.sleep(2)  # UI'ın başlaması için bekle
    
    if not no_browser:
        webbrowser.open(f"http://localhost:{port}")
        click.echo(f"MLflow UI tarayıcıda açıldı: http://localhost:{port}")
    
    click.echo(f"MLflow UI çalışıyor: http://localhost:{port}")
    click.echo("Durdurmak için Ctrl+C tuşlarına basın.")
    
    try:
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        click.echo("MLflow UI durduruldu.")

@cli.command()
@click.option('--experiment_name', help='Experiment adı')
@click.option('--model_type', '-m', default='4', 
             help='Model tipi: 1=Baseline, 2=LightGBM, 3=Source-based, 4=Ensemble')
@click.option('--skip_stats', is_flag=True, help='İstatistiksel analizleri atla')
@click.option('--skip_feature_imp', is_flag=True, help='Feature importance adımını atla')
@click.option('--skip_auto_select', is_flag=True, help='Akıllı özellik seçimini atla')
@click.option('--dashboard', is_flag=True, help='İşlem tamamlandığında dashboard başlat')
@click.option('--create_new_run', is_flag=True, default=True, help='Yeni bir run klasörü oluştur')
def all(experiment_name, model_type, skip_stats, skip_feature_imp, skip_auto_select, dashboard, create_new_run):
    """Tüm pipeline adımlarını çalıştırır"""
    # Yeni run klasörü isteniyorsa reset et
    from src.utils.paths import reset_experiment_dir, get_experiment_dir
    
    # Varsayılan olarak her zaman yeni bir run oluştur
    reset_experiment_dir()
    
    # Timestamp oluştur
    timestamp = int(time.time())
    
    # Experiment adı ayarla
    if not experiment_name:
        experiment_name = f"full_run_{timestamp}"
    
    # Ortam değişkenlerini ayarla
    _setup_environment()
    
    # Experiment dizinini al (yeni oluşturulmuş olmalı)
    exp_dir = get_experiment_dir()
    
    click.echo("=== FULL PIPELINE RUN ===")
    click.echo(f"Experiment: {experiment_name}")
    click.echo(f"Çıktı dizini: {exp_dir}")
    
    try:
        # 1. Veri hazırlama ve bölme
        click.echo("\n[1/5] Veri hazırlama ve bölme...")
        subprocess.run([sys.executable, "-m", "src.pipelines.split", 
                       "--train-cutoff", "2024-06-30", 
                       "--val-cutoff", "2024-11-30", 
                       "--test-cutoff", "2025-04-30", 
                       "--output-dir", str(exp_dir)], 
                       check=True, env=os.environ)
        click.echo("✓ Veri hazırlama ve bölme tamamlandı.")
        
        # 2. İstatistiksel analizler (isteğe bağlı)
        if not skip_stats:
            click.echo("\n[2/5] İstatistiksel analizler...")
            # stats() fonksiyonunu doğrudan çağırmak yerine subprocess kullan
            stats_cmd = [sys.executable, "-m", "src.pipelines.statistical_tests", 
                        "--test_type", "all", 
                        "--use_train_only",
                        "--exclude_id_cols",
                        "--exclude_date_cols",
                        "--exclude_high_cardinality",
                        "--train_path", f"{exp_dir}/train.csv",
                        "--output_subdir", f"stats_{timestamp}"]
            subprocess.run(stats_cmd, check=True, env=os.environ)
            click.echo("✓ İstatistiksel analizler tamamlandı.")
        else:
            click.echo("\n[2/5] İstatistiksel analizler atlandı.")
        
        # 3. Feature importance (isteğe bağlı)
        if not skip_feature_imp:
            click.echo("\n[3/5] Feature importance hesaplanıyor...")
            # feature_imp() fonksiyonunu doğrudan çağırmak yerine subprocess kullan
            train_path = f"{exp_dir}/train.csv"
            feat_cmd = [sys.executable, "-m", "src.pipelines.feature_importance", 
                       "--method", "shap", 
                       f"--train-path={train_path}",
                       f"--output-dir={exp_dir}/feature_importance"]
            subprocess.run(feat_cmd, check=True, env=os.environ)
            click.echo("✓ Feature importance hesaplaması tamamlandı.")
        else:
            click.echo("\n[3/5] Feature importance adımı atlandı.")
        
        # 4. Akıllı özellik seçimi (isteğe bağlı)
        if not skip_auto_select:
            click.echo("\n[4/5] Akıllı özellik seçimi yapılıyor...")
            auto_select_cmd = [sys.executable, "-m", "src.features.auto_selector", 
                             f"--input-file={exp_dir}/train.csv",
                             f"--output-dir={exp_dir}/auto_select"]
            subprocess.run(auto_select_cmd, check=True, env=os.environ)
            click.echo("✓ Akıllı özellik seçimi tamamlandı.")
        else:
            click.echo("\n[4/5] Akıllı özellik seçimi atlandı.")
        
        # 5. Model eğitimi
        click.echo(f"\n[5/5] Model eğitimi başlatılıyor... (Model: {model_type})")
        train_cmd = [sys.executable, "-m", "src.pipelines.train", 
                    f"+model={model_type}", 
                    f"++experiment_name={experiment_name}",
                    f"++split_dir={exp_dir}"]
        subprocess.run(train_cmd, check=True, env=os.environ)
        click.echo("✓ Model eğitimi tamamlandı.")
        
        click.echo("\n=== FULL PIPELINE TAMAMLANDI ===")
        click.echo(f"Experiment adı: {experiment_name}")
        click.echo(f"Çıktı dizini: {exp_dir}")
        
        # Dashboard başlat (isteğe bağlı)
        if dashboard:
            click.echo("\nDashboard başlatılıyor...")
            _start_services()
        
    except subprocess.CalledProcessError as e:
        click.echo(f"\nPipeline çalıştırılırken hata oluştu: {e}")
        sys.exit(1)

@cli.command()
@click.option('--input_file', '-i', default='outputs/split/train.csv', 
             help='Girdi CSV dosyası (varsayılan: outputs/split/train.csv)')
@click.option('--output_dir', '-o', help='Çıktı dizini (varsayılan: outputs/auto_select)')
@click.option('--missing_thresh', default=0.3, type=float, 
             help='Eksik değer oranı eşiği (0-1 arası)')
@click.option('--duplicate/--no_duplicate', default=True, 
             help='Duplikat kolonları ele')
@click.option('--near_zero_var', default=0.01, type=float, 
             help='Düşük varyans eşiği (0-1 arası)')
@click.option('--outlier_method', type=click.Choice(['iqr', 'zscore']), default='iqr', 
             help='Outlier tespit metodu')
@click.option('--outlier_thresh', default=0.5, type=float, 
             help='Outlier oranı eşiği (0-1 arası)')
@click.option('--correlation_thresh', default=0.95, type=float, 
             help='Yüksek korelasyon eşiği (0-1 arası)')
@click.option('--target_correlation_min', default=0.02, type=float, 
             help='Minimum hedef korelasyonu (0-1 arası)')
@click.option('--pca/--no_pca', default=False, 
             help='PCA uygula')
@click.option('--pca_components', default=5, type=int, 
             help='PCA bileşen sayısı (PCA aktifse)')
@click.option('--update_config/--no_update_config', default=True, 
             help='Experiment konfigürasyonunu güncelle')
def auto_select(input_file, output_dir, missing_thresh, duplicate, near_zero_var, 
               outlier_method, outlier_thresh, correlation_thresh, 
               target_correlation_min, pca, pca_components, update_config):
    """Akıllı özellik seçimi yapar"""
    if not os.path.exists(input_file):
        click.echo(f"Girdi dosyası bulunamadı: {input_file}")
        sys.exit(1)
    
    # Mevcut run klasörünü kullan
    from src.utils.paths import get_experiment_dir
    _setup_environment()
    
    # Çıktı dizini
    if not output_dir:
        exp_dir = get_experiment_dir()
        output_dir = f"{exp_dir}/auto_select"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # PCA bileşenlerini ayarla
    pca_param = str(pca_components) if pca else "None"
    
    # Komutu hazırla
    cmd = [
        sys.executable, "-c",
        f"""
import pandas as pd
from src.features.auto_selector import SmartFeatureSelector, generate_feature_report
import joblib
from pathlib import Path
import os
from src.utils.config_updater import update_experiment_config

# Veriyi yükle
print(f"Veriler okunuyor: {input_file}")
df = pd.read_csv("{input_file}")

# Target kolonu kontrol et
target_col = "Target_IsConverted"
y = None
if target_col in df.columns:
    print(f"Target kolonu bulundu: {target_col}")
    y = df[target_col].copy()
    X = df.drop(columns=[target_col])
else:
    print(f"Target kolonu bulunamadı: {target_col}")
    X = df.copy()

# SmartFeatureSelector oluştur
selector = SmartFeatureSelector(
    missing_thresh={missing_thresh},
    duplicate={str(duplicate).lower()},
    near_zero_var_thresh={near_zero_var},
    outlier_method="{outlier_method}",
    outlier_thresh={outlier_thresh},
    correlation_thresh={correlation_thresh},
    target_correlation_min={target_correlation_min},
    pca_components={pca_param},
    verbose=True
)

# Fit et
print("Akıllı özellik seçimi uygulanıyor...")
selector.fit(X, y)

# Sonuçları kaydet
print(f"Sonuçlar kaydediliyor: {output_dir}")
selected_features_path = Path("{output_dir}") / "selected_features.txt"
with open(selected_features_path, "w") as f:
    for feature in selector.kept_columns_:
        f.write(f"{feature}\\n")

dropped_features_path = Path("{output_dir}") / "dropped_features.txt"
with open(dropped_features_path, "w") as f:
    for feature in selector.to_drop_:
        f.write(f"{feature}\\n")

# Rapor oluştur
report_path = Path("{output_dir}") / "feature_selection_report.html"
generate_feature_report(selector, str(report_path))

# Selector'ı kaydet
joblib.dump(selector, Path("{output_dir}") / "auto_selector.pkl")

# Dönüştürülmüş veri
X_selected = selector.transform(X)
if y is not None:
    X_selected[target_col] = y

# Dönüştürülmüş veriyi kaydet
X_selected.to_csv(Path("{output_dir}") / "transformed_data.csv", index=False)

# Sayısal ve kategorik kolonları belirle
num_cols = X_selected.select_dtypes(include='number').columns.tolist()
cat_cols = X_selected.select_dtypes(exclude='number').columns.tolist()

# Target kolonu listelerden çıkar
if target_col in num_cols:
    num_cols.remove(target_col)
elif target_col in cat_cols:
    cat_cols.remove(target_col)

# ID kolonlarını çıkar
id_cols = ["LeadId", "account_Id"]
for id_col in id_cols:
    if id_col in num_cols:
        num_cols.remove(id_col)
    elif id_col in cat_cols:
        cat_cols.remove(id_col)

# İstatistikleri yazdır
print(f"\\nÖzellik Seçimi Özeti:")
print(f"  Toplam özellik sayısı: {{len(selector.columns_)}}")
print(f"  Seçilen özellik sayısı: {{len(selector.kept_columns_)}}")
print(f"  Elenen özellik sayısı: {{len(selector.to_drop_)}}")
print(f"  Sayısal özellik sayısı: {{len(num_cols)}}")
print(f"  Kategorik özellik sayısı: {{len(cat_cols)}}")

# Konfigürasyonu güncelle
if {str(update_config).lower()}:
    try:
        update_experiment_config(num_cols=num_cols, cat_cols=cat_cols)
        print("\\nExperiment konfigürasyonu güncellendi.")
    except Exception as e:
        print(f"\\nExperiment konfigürasyonu güncellenemedi: {{e}}")
        # Yedek çözüm: Özellik listesini ayrı bir dosyaya yaz
        backup_path = Path("{output_dir}") / "manual_update_needed.txt"
        with open(backup_path, "w") as f:
            f.write("# num_cols:\\n")
            for col in num_cols:
                f.write(f"- {{col}}\\n")
            f.write("\\n# cat_cols:\\n")
            for col in cat_cols:
                f.write(f"- {{col}}\\n")
        print(f"Manuel güncelleme için yedek liste oluşturuldu: {{backup_path}}")

print(f"\\nAkıllı özellik seçimi tamamlandı. Sonuçlar: {output_dir}")
print(f"  - HTML rapor: {{report_path}}")
print(f"  - Seçilen özellikler: {{selected_features_path}}")
print(f"  - Elenen özellikler: {{dropped_features_path}}")
print(f"  - Dönüştürülmüş veri: {{Path('{output_dir}') / 'transformed_data.csv'}}")
print(f"  - SmartFeatureSelector: {{Path('{output_dir}') / 'auto_selector.pkl'}}")
"""
    ]
    
    click.echo(f"Akıllı özellik seçimi başlatılıyor...")
    click.echo(f"Çıktı dizini: {output_dir}")
    try:
        subprocess.run(cmd, check=True, env=os.environ)
        click.echo(f"Akıllı özellik seçimi tamamlandı! Çıktılar: {output_dir}")
        
        # Tarayıcıda raporu göster
        report_path = os.path.join(output_dir, "feature_selection_report.html")
        if os.path.exists(report_path):
            if click.confirm("Özellik seçim raporunu tarayıcıda açmak ister misiniz?", default=True):
                click.echo(f"Rapor açılıyor: {report_path}")
                webbrowser.open(f"file://{os.path.abspath(report_path)}")
    except subprocess.CalledProcessError as e:
        click.echo(f"Akıllı özellik seçimi sırasında hata oluştu: {e}")
        sys.exit(1)

def _find_available_port(start_port, max_attempts=10):
    """Belirtilen başlangıç noktasından itibaren kullanılabilir bir port bulur."""
    port = start_port
    for _ in range(max_attempts):
        if not _is_port_in_use(port):
            return port
        port += 1
    return start_port + max_attempts  # Yine de bulunamazsa, varsayılan + max_attempts döndür

def _start_streamlit():
    """Streamlit servisini başlatır"""
    streamlit_port = CONFIG.get("streamlit_port", STREAMLIT_PORT)
    
    # Port kontrolü
    if _is_port_in_use(streamlit_port):
        logger.warning(f"Port {streamlit_port} zaten kullanımda, alternatif port aranıyor...")
        streamlit_port = _find_available_port(streamlit_port)
    
    click.echo("Streamlit Dashboard başlatılıyor...")
    process = subprocess.Popen(["streamlit", "run", "streamlit_app/app.py", 
                               "--server.port", str(streamlit_port)], env=os.environ)
    
    time.sleep(2)  # Dashboard'ın başlaması için bekle
    
    webbrowser.open(f"http://localhost:{streamlit_port}")
    click.echo(f"Streamlit dashboard başlatıldı: http://localhost:{streamlit_port}")
    
    return process

def _start_mlflow():
    """MLflow servisini başlatır"""
    mlflow_port = CONFIG.get("mlflow_port", MLFLOW_PORT)
    
    # Port kontrolü
    if _is_port_in_use(mlflow_port):
        logger.warning(f"Port {mlflow_port} zaten kullanımda, alternatif port aranıyor...")
        mlflow_port = _find_available_port(mlflow_port)
    
    click.echo("MLflow UI başlatılıyor...")
    process = subprocess.Popen(["mlflow", "ui", "--port", str(mlflow_port)], env=os.environ)
    
    time.sleep(2)  # UI'ın başlaması için bekle
    
    webbrowser.open(f"http://localhost:{mlflow_port}")
    click.echo(f"MLflow UI başlatıldı: http://localhost:{mlflow_port}")
    
    return process

def _start_services():
    """Hem Streamlit hem de MLflow servislerini başlatır"""
    streamlit_port = CONFIG.get("streamlit_port", STREAMLIT_PORT)
    mlflow_port = CONFIG.get("mlflow_port", MLFLOW_PORT)
    
    # Port kontrolü ve alternatif bulma
    if _is_port_in_use(streamlit_port):
        logger.warning(f"Port {streamlit_port} zaten kullanımda, alternatif port aranıyor...")
        streamlit_port = _find_available_port(streamlit_port)
    
    if _is_port_in_use(mlflow_port):
        logger.warning(f"Port {mlflow_port} zaten kullanımda, alternatif port aranıyor...")
        mlflow_port = _find_available_port(mlflow_port)
    
    streamlit_process = subprocess.Popen(["streamlit", "run", "streamlit_app/app.py", 
                                      "--server.port", str(streamlit_port)], env=os.environ)
    
    time.sleep(2)  # Dashboard'ın başlaması için bekle
    webbrowser.open(f"http://localhost:{streamlit_port}")
    
    mlflow_process = subprocess.Popen(["mlflow", "ui", "--port", str(mlflow_port)], env=os.environ)
    
    time.sleep(2)  # UI'ın başlaması için bekle
    webbrowser.open(f"http://localhost:{mlflow_port}")
    
    click.echo(f"Streamlit dashboard başlatıldı: http://localhost:{streamlit_port}")
    click.echo(f"MLflow UI başlatıldı: http://localhost:{mlflow_port}")
    click.echo("\nServisler çalışıyor. Durdurmak için Ctrl+C tuşlarına basın.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        streamlit_process.terminate()
        mlflow_process.terminate()
        click.echo("Servisler durduruldu.")

# Port kontrolü için yardımcı fonksiyon
def _is_port_in_use(port):
    """
    Belirtilen portun kullanımda olup olmadığını kontrol eder.
    
    Args:
        port: Kontrol edilecek port numarası
        
    Returns:
        bool: Port kullanımdaysa True, değilse False
    """
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

if __name__ == "__main__":
    cli() 