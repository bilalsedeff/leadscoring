import click, questionary, os, webbrowser, subprocess, platform, sys
from rich.console import Console
from rich.table import Table
from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
import pandas as pd
import matplotlib.pyplot as plt
import time
import logging
from pathlib import Path
import yaml
from omegaconf import OmegaConf

# Logger yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("lead_scoring.log"), logging.StreamHandler()]
)
logger = logging.getLogger("lead_scoring_cli")

console = Console()

# === Konfigürasyon Yükleme ===
def load_config():
    """Uygulama konfigürasyonunu yükler."""
    config_path = Path("configs/app_config.yaml")
    if not config_path.exists():
        # Varsayılan konfigürasyon
        config = {
            "mlflow_port": 5000,
            "streamlit_port": 8501,
            "output_dir": "outputs",
            "tmp_dir": "tmp"
        }
        # Dizin yapısını oluştur
        os.makedirs(config["output_dir"], exist_ok=True)
        os.makedirs(config["tmp_dir"], exist_ok=True)
        
        # Konfigürasyonu kaydet
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        logger.info(f"Varsayılan konfigürasyon oluşturuldu: {config_path}")
    else:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Konfigürasyon yüklendi: {config_path}")
    
    return config

# Konfigürasyonu yükle
CONFIG = load_config()

# Sabitler
MLFLOW_PORT = CONFIG["mlflow_port"]
STREAMLIT_PORT = CONFIG["streamlit_port"]
OUTPUT_DIR = CONFIG["output_dir"]

# Dizin oluştur
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Çapraz platform komut yürütme
def run_command(command, check=True, shell=True, background=False):
    """
    Komutu çalıştırır ve sonuçları döndürür.
    
    Args:
        command: Çalıştırılacak komut
        check: Hata durumunda exception fırlatılsın mı?
        shell: Shell kullanılsın mı?
        background: Arka planda çalıştırılsın mı?
        
    Returns:
        subprocess.CompletedProcess veya subprocess.Popen nesnesi
    """
    try:
        if background:
            if platform.system() == "Windows":
                # Windows'ta DETACHED_PROCESS ile arka planda çalıştır
                proc = subprocess.Popen(
                    command,
                    shell=shell,
                    creationflags=subprocess.DETACHED_PROCESS
                )
            else:
                # Unix sistemlerde stdout/stderr'i pipe'la
                proc = subprocess.Popen(
                    command,
                    shell=shell,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            return proc
        else:
            # Ön planda çalıştır ve bitirmeyi bekle
            result = subprocess.run(
                command,
                shell=shell,
                check=check,
                text=True,
                capture_output=True
            )
            return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Komut çalıştırma hatası: {e}")
        console.print(f"[bold red]Hata: {e}[/]")
        console.print(f"[red]Çıktı: {e.stdout}[/]")
        console.print(f"[red]Hata Çıktısı: {e.stderr}[/]")
        raise
    except Exception as e:
        logger.error(f"Beklenmeyen hata: {e}")
        console.print(f"[bold red]Beklenmeyen hata: {e}[/]")
        raise

@click.group()
def cli():
    """Lead Scoring Interactive CLI"""

@cli.command()
def menu():
    """Lead Scoring işlem menüsünü başlatır"""
    # Ana menü
    while True:
        choice = questionary.select(
            "Ne yapmak istiyorsunuz?",
            choices=[
                "1️⃣  Veri Hazırlama & Split",
                "2️⃣  Feature Importance → seçim",
                "3️⃣  Akıllı Özellik Seçimi",
                "4️⃣  İstatistiksel Analiz",
                "5️⃣  Model Eğitimi",
                "6️⃣  Full Run (tüm adımlar)",
                "7️⃣  Sonuçları Görüntüle",
                "8️⃣  Servisleri Başlat (MLflow & Streamlit)",
                "9️⃣  Çıktıları Temizle",
                "🔟  Konfigürasyonu Güncelle",
                "🚪  Çık"
            ]).ask()

        if choice.startswith("1"):
            _split_step()
        elif choice.startswith("2"):
            _feat_imp_step()
        elif choice.startswith("3"):
            _auto_select_step()
        elif choice.startswith("4"):
            _stat_test_step()
        elif choice.startswith("5"):
            _train_step()
        elif choice.startswith("6"):
            _full_run()
        elif choice.startswith("7"):
            _view_results()
        elif choice.startswith("8"):
            _start_services()
        elif choice.startswith("9"):
            _cleanup_outputs()
        elif choice.startswith("🔟"):
            _update_config()
        else:
            console.print("[bold red]Görüşürüz![/]")
            break

def _split_step():
    """Veri hazırlama ve split işlemlerini gerçekleştirir"""
    console.rule("[bold blue]Veri Hazırlama & Split Başlıyor[/]")
    
    # Yeni run klasörü oluşturup oluşturmayacağımızı sor
    create_new_run = questionary.confirm(
        "Yeni bir run klasörü oluşturmak istiyor musunuz?",
        default=False
    ).ask()
    
    # Eğer yeni run klasörü isteniyorsa çevre değişkenlerini güncelle
    if create_new_run:
        # Yeni timestamp oluştur
        timestamp = int(time.time())
        # OUTPUT_DIR değişkenini güncelle
        global OUTPUT_DIR
        OUTPUT_DIR = os.path.join(os.path.dirname(OUTPUT_DIR), f"run_{timestamp}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # paths.py içindeki get_experiment_dir fonksiyonunu da güncellememiz gerekiyor
        from src.utils.paths import update_experiment_dir
        update_experiment_dir(OUTPUT_DIR)
        
        console.print(f"[green]Yeni run klasörü oluşturuldu: {OUTPUT_DIR}[/]")
    
    # Output klasörünü oluştur
    split_dir = f"{OUTPUT_DIR}/split"
    os.makedirs(split_dir, exist_ok=True)
    
    # Split işlemini çalıştır
    console.print("[yellow]Veriler hazırlanıyor ve split işlemi yapılıyor...[/]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[yellow]Veri hazırlanıyor...", total=None)
        
        # Split parametrelerini yapılandırmadan al
        config_path = Path("configs/split.yaml")
        if not config_path.exists():
            console.print("[red]Split yapılandırması bulunamadı. Varsayılan değerler kullanılacak.[/]")
            config = {}
        else:
            config = OmegaConf.load(config_path)
        
        # Split parametrelerini sor
        train_cutoff = questionary.text(
            "Eğitim veri seti kesim tarihi (YYYY-MM-DD formatı):",
            default=str(config.get("train_cutoff", "2024-06-30"))  # 2024 Haziran sonu
        ).ask()
        
        val_cutoff = questionary.text(
            "Validasyon veri seti kesim tarihi (YYYY-MM-DD formatı):",
            default=str(config.get("val_cutoff", "2024-11-30"))  # 2024 Kasım sonu
        ).ask()
        
        test_cutoff = questionary.text(
            "Test veri seti kesim tarihi (YYYY-MM-DD formatı):",
            default=str(config.get("test_cutoff", "2025-04-30"))  # 2025 Nisan sonu
        ).ask()
        
        # Split yapmadan önce parametreleri doğrula
        try:
            # Tarihleri doğrula (YYYY-MM-DD formatında olmalı)
            pd.to_datetime(train_cutoff)
            pd.to_datetime(val_cutoff)
            pd.to_datetime(test_cutoff)
            
            # Kesim tarihlerinin mantıklı olduğunu kontrol et
            if not (pd.to_datetime(train_cutoff) < pd.to_datetime(val_cutoff) < pd.to_datetime(test_cutoff)):
                console.print("[red]Hata: Kesim tarihleri sıralı olmalı: train_cutoff < val_cutoff < test_cutoff[/]")
                return
                
        except ValueError as e:
            console.print(f"[red]Hata: Tarih formatı geçersiz. YYYY-MM-DD formatında olmalı: {e}[/]")
            return
        
        # Grup kolonunu sor
        group_col = questionary.text(
            "Grup kolonu (account_Id vb., boş bırakabilirsiniz):",
            default=config.get("group_col", "account_Id")
        ).ask() or None
        
        # Zaman kolonunu sor
        time_col = questionary.text(
            "Zaman kolonu:",
            default=config.get("time_col", "YearMonth")
        ).ask()
        
        # Random seed sor
        random_seed = questionary.text(
            "Random seed (rastgele numara üretimi için):",
            default=str(config.get("random_seed", "42"))
        ).ask()
        
        try:
            random_seed = int(random_seed)
        except ValueError:
            random_seed = 42
            console.print("[yellow]Uyarı: Random seed bir sayı değil. Varsayılan 42 kullanılacak.[/]")
        
        # Split işlemini çalıştır
        try:
            from src.pipelines.split import run_split
            
            # Split işlemi için ihtiyaç duyulan parametreleri hazırla
            split_params = {
                "train_cutoff": train_cutoff,
                "val_cutoff": val_cutoff,
                "test_cutoff": test_cutoff,
                "time_col": time_col,
                "group_col": group_col,
                "random_seed": random_seed,
                "output_dir": split_dir,
                "force_balance": True  # Tüm setlerde veri olmasını zorla
            }
            
            # Split işlemini çalıştır
            result = run_split(**split_params)
            
            progress.update(task, description="[green]Veri hazırlama tamamlandı!")
            
            # Sonuçları göster
            console.print("\n[bold green]Veri Hazırlama & Split Tamamlandı![/]")
            console.print(f"Eğitim veri seti: {result['train_shape'][0]} satır, {result['train_shape'][1]} kolon")
            console.print(f"Validasyon veri seti: {result['val_shape'][0]} satır, {result['val_shape'][1]} kolon")
            console.print(f"Test veri seti: {result['test_shape'][0]} satır, {result['test_shape'][1]} kolon")
            
            # Test setinin boş olup olmadığını kontrol et
            if result['test_shape'][0] == 0:
                console.print("[bold red]UYARI: Test veri seti boş! Lütfen kesim tarihlerini kontrol edin.[/]")
            
            console.print(f"\nVeri setleri kaydedildi: {split_dir}")
            
            # Yapılandırmayı güncelle ve kaydet
            config = OmegaConf.create({
                "train_cutoff": train_cutoff,
                "val_cutoff": val_cutoff,
                "test_cutoff": test_cutoff,
                "time_col": time_col,
                "group_col": group_col,
                "random_seed": random_seed
            })
            
            os.makedirs("configs", exist_ok=True)
            OmegaConf.save(config, config_path)
            console.print(f"Split yapılandırması güncellendi: {config_path}")
                
        except Exception as e:
            console.print(f"[bold red]Hata: {str(e)}[/]")
            import traceback
            console.print(traceback.format_exc())
    
    return True

def _feat_imp_step():
    """Feature importance hesaplar ve görselleştirir"""
    console.rule("[bold blue]Feature Importance Analizi[/]")
    
    # Output klasörlerini oluştur
    feat_dir = f"{OUTPUT_DIR}/feature_importance"
    split_dir = f"{OUTPUT_DIR}/split"
    os.makedirs(feat_dir, exist_ok=True)
    
    # Train veri seti kontrolü
    train_path = os.path.join(split_dir, "train.csv")
    if not os.path.exists(train_path):
        console.print("[bold red]Eğitim verisi (train.csv) bulunamadı![/]")
        console.print("[yellow]Önce veri hazırlama ve split adımını çalıştırmalısınız.[/]")
        
        if questionary.confirm("Veri hazırlama ve split adımını şimdi çalıştırmak ister misiniz?", default=True).ask():
            _split_step()
        else:
            console.print("[red]İşlem iptal edildi.[/]")
            return
    
    # Feature importance hesapla
    console.print("[yellow]Feature importance hesaplanıyor (SADECE TRAIN VERİSİ ÜZERİNDE)...[/]")
    
    # İlerleme göstergesi ile komutu çalıştır
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Feature importance hesaplanıyor...", total=None)
        try:
            result = run_command(f"python -m src.pipelines.feature_importance --train-path={train_path}")
            progress.update(task, completed=True)
            
            if result.returncode != 0:
                console.print("[bold red]Feature importance hesaplama hatası![/]")
                console.print(f"[red]Çıktı: {result.stdout}[/]")
                console.print(f"[red]Hata: {result.stderr}[/]")
                return
                
        except Exception as e:
            progress.update(task, completed=True)
            logger.exception("Feature importance hesaplamada hata")
            console.print(f"[bold red]Hata: {str(e)}[/]")
            return
    
    # Sonuçları oku
    imp_path = f"{feat_dir}/importance.csv"
    if os.path.exists(imp_path):
        try:
            imp = pd.read_csv(imp_path)
            
            # Tablo oluştur
            table = Table(title="Feature Importance (En Yüksek 20)")
            table.add_column("Sıra", style="cyan")
            table.add_column("Feature", style="green")
            table.add_column("Önem Skoru", style="magenta")
            
            # Top 20 feature'ları göster
            for i, (feature, score) in enumerate(zip(imp.iloc[:20, 0], imp.iloc[:20, 1]), 1):
                table.add_row(str(i), feature, f"{score:.4f}")
            
            console.print(table)
            
            # Görselleştirme
            try:
                plt.figure(figsize=(10, 8))
                plt.barh(imp.iloc[:20, 0], imp.iloc[:20, 1])
                plt.xlabel('Önem Skoru')
                plt.ylabel('Feature')
                plt.title('Feature Importance (Top 20) - Sadece Eğitim Verisi')
                plt.tight_layout()
                plt.savefig(f"{feat_dir}/importance_plot.png")
                console.print(f"[green]Görselleştirme kaydedildi: {feat_dir}/importance_plot.png[/]")
            except Exception as e:
                logger.warning(f"Görselleştirme oluşturulurken hata: {e}")
                console.print("[yellow]Görselleştirme oluşturulurken hata oluştu, ancak işleme devam edilecek.[/]")
            
            # Kullanıcıdan kaç feature seçileceğini sor
            k = questionary.text(
                "İlk kaç özelliği modele dahil etmek istiyorsunuz? (5-100 arası önerilir)",
                default="20"
            ).ask()
            
            try:
                k = int(k)
                if k <= 0:
                    raise ValueError("Pozitif bir sayı girilmelidir")
                    
                # Feature seçimini kaydet
                with open(f"{feat_dir}/selected_features.txt", "w") as f:
                    for i in range(min(k, len(imp))):
                        f.write(f"{imp.iloc[i, 0]}\n")
                
                console.print(f"[green]Seçilen {k} feature kaydedildi: {feat_dir}/selected_features.txt[/]")
                
                # Çevre değişkeni olarak ayarla
                os.environ["FEAT_TOP_K"] = str(k)
                
                # experiment.yaml dosyasını güncelle
                try:
                    from src.utils.config_updater import update_from_feature_importance
                    if update_from_feature_importance(k):
                        console.print("[green]Experiment yapılandırması başarıyla güncellendi.[/]")
                        console.print("[cyan]Seçilen özellikler artık experiment.yaml dosyasında![/]")
                        console.print("[bold yellow]ÖNEMLİ: Feature importance analizi sadece eğitim verisi üzerinde yapıldı (data leakage engellendi)![/]")
                    else:
                        console.print("[bold red]Experiment yapılandırması güncellenemedi![/]")
                except Exception as e:
                    logger.exception("Experiment yapılandırması güncellenirken hata")
                    console.print(f"[bold red]Yapılandırma güncellenirken hata: {str(e)}[/]")
                
            except ValueError as ve:
                console.print(f"[bold red]Geçersiz sayı: {str(ve)}! Varsayılan olarak 20 kullanılacak.[/]")
                os.environ["FEAT_TOP_K"] = "20"
                
                # Varsayılan değerle experiment.yaml güncelle
                try:
                    from src.utils.config_updater import update_from_feature_importance
                    update_from_feature_importance(20)
                    console.print("[green]Experiment yapılandırması varsayılan 20 feature ile güncellendi.[/]")
                except Exception as e:
                    logger.exception("Varsayılan değerle güncelleme yapılırken hata")
                    console.print(f"[bold red]Varsayılan değerle güncelleme yapılırken hata: {str(e)}[/]")
        
        except Exception as e:
            logger.exception("Feature importance sonuçları işlenirken hata")
            console.print(f"[bold red]Sonuçlar işlenirken hata: {str(e)}[/]")
    else:
        console.print("[bold red]Feature importance hesaplanamadı veya dosya bulunamadı![/]")
        console.print(f"[red]Beklenen dosya: {imp_path}[/]")

def _auto_select_step():
    """Akıllı özellik seçimi yapar"""
    console.rule("[bold blue]Akıllı Özellik Seçimi[/]")
    
    # Output klasörlerini oluştur
    auto_select_dir = f"{OUTPUT_DIR}/auto_select"
    split_dir = f"{OUTPUT_DIR}/split"
    os.makedirs(auto_select_dir, exist_ok=True)
    
    # Train veri seti kontrolü
    train_path = os.path.join(split_dir, "train.csv")
    if not os.path.exists(train_path):
        console.print("[bold red]Eğitim verisi (train.csv) bulunamadı![/]")
        console.print("[yellow]Önce veri hazırlama ve split adımını çalıştırmalısınız.[/]")
        
        if questionary.confirm("Veri hazırlama ve split adımını şimdi çalıştırmak ister misiniz?", default=True).ask():
            _split_step()
        else:
            console.print("[red]İşlem iptal edildi.[/]")
            return
    
    # Parametreleri sor
    missing_thresh = questionary.text(
        "Eksik değer oranı eşiği (0-1 arası):",
        default="0.3"
    ).ask()
    
    duplicate = questionary.confirm(
        "Duplikat kolonları ele?",
        default=True
    ).ask()
    
    near_zero_var = questionary.text(
        "Düşük varyans eşiği (0-1 arası):",
        default="0.01"
    ).ask()
    
    outlier_method = questionary.select(
        "Outlier tespit metodu:",
        choices=["iqr", "zscore"]
    ).ask()
    
    outlier_thresh = questionary.text(
        "Outlier oranı eşiği (0-1 arası):",
        default="0.5"
    ).ask()
    
    use_pca = questionary.confirm(
        "PCA uygula?",
        default=False
    ).ask()
    
    pca_components = None
    if use_pca:
        pca_components = questionary.text(
            "PCA bileşen sayısı:",
            default="5"
        ).ask()
    
    update_config = questionary.confirm(
        "Experiment konfigürasyonunu güncelle?",
        default=True
    ).ask()
    
    # Komut oluştur
    cmd = ["python", "lead_scoring.py", "auto-select", 
          f"--input-file={train_path}",
          f"--output-dir={auto_select_dir}",
          f"--missing-thresh={missing_thresh}",
          f"--{'duplicate' if duplicate else 'no-duplicate'}",
          f"--near-zero-var={near_zero_var}",
          f"--outlier-method={outlier_method}",
          f"--outlier-thresh={outlier_thresh}"]
    
    if use_pca:
        cmd.append("--pca")
        cmd.append(f"--pca-components={pca_components}")
    else:
        cmd.append("--no-pca")
    
    if update_config:
        cmd.append("--update-config")
    else:
        cmd.append("--no-update-config")
    
    # Akıllı özellik seçimi işlemini çalıştır
    console.print("[yellow]Akıllı özellik seçimi uygulanıyor...[/]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Akıllı özellik seçimi çalışıyor...", total=None)
        try:
            result = run_command(" ".join(cmd))
            progress.update(task, completed=True)
            
            if result.returncode != 0:
                console.print("[bold red]Akıllı özellik seçimi hatası![/]")
                console.print(f"[red]Çıktı: {result.stdout}[/]")
                console.print(f"[red]Hata: {result.stderr}[/]")
                return
                
        except Exception as e:
            progress.update(task, completed=True)
            logger.exception("Akıllı özellik seçiminde hata")
            console.print(f"[bold red]Hata: {str(e)}[/]")
            return
    
    console.print("[green]Akıllı özellik seçimi tamamlandı![/]")
    
    # Raporu tarayıcıda aç
    report_path = os.path.join(auto_select_dir, "feature_selection_report.html")
    if os.path.exists(report_path):
        if questionary.confirm("Özellik seçim raporunu tarayıcıda açmak ister misiniz?", default=True).ask():
            webbrowser.open(f"file://{os.path.abspath(report_path)}")

def _stat_test_step():
    """İstatistiksel testleri gerçekleştirir"""
    console.rule("[bold blue]İstatistiksel Testler Başlıyor[/]")
    
    # İstatistiksel test tipini sor
    test_type = questionary.select(
        "Hangi istatistiksel testi yapmak istiyorsunuz?",
        choices=[
            "chi_square", 
            "t_test", 
            "anova", 
            "conversion_rate", 
            "correlation", 
            "all"
        ],
        default="all"
    ).ask()
    
    # Kullanılacak veri setini sor
    train_only = questionary.confirm(
        "Sadece eğitim veri setini mi kullanmak istiyorsunuz? (Hayır seçilirse tüm veri kullanılır - data leak riski!)",
        default=True
    ).ask()
    
    # Yeni alt klasör oluştur
    timestamp = int(time.time())
    output_subdir = f"{test_type}_{timestamp}_{'train_only' if train_only else 'all_data'}"
    
    # İstatistiksel test işlemlerini çalıştır
    console.print("[yellow]İstatistiksel testler çalıştırılıyor...[/]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[yellow]İstatistiksel testler yürütülüyor...", total=None)
        
        try:
            from src.pipelines.statistical_tests import main as run_statistical_tests
            import click
            
            # Click komutunu manuel olarak çalıştır
            ctx = click.Context(run_statistical_tests)
            result = run_statistical_tests.callback(
                test_type=test_type,
                categorical_cols=None,
                numeric_cols=None,
                group_col="Source_Final__c",  # Kaynak bazlı gruplamayı varsayılan olarak kullan
                target_col="Target_IsConverted",
                corr_method="pearson",
                min_corr=0.05,
                output_subdir=output_subdir,
                train_path=f"{OUTPUT_DIR}/split/train.csv",
                use_train_only=train_only,
                exclude_id_cols=True,
                train_cutoff=None,  # Mevcut yapılandırmadan alınacak
                val_cutoff=None     # Mevcut yapılandırmadan alınacak
            )
            
            progress.update(task, description="[green]İstatistiksel testler tamamlandı!")
                    
            # Sonuçları göster
            console.print("\n[bold green]İstatistiksel Testler Tamamlandı![/]")
            console.print(f"Sonuçlar şuraya kaydedildi: {result}")
            
        except Exception as e:
            console.print(f"[bold red]Hata: {str(e)}[/]")
            import traceback
            console.print(traceback.format_exc())
    
    return True

def _train_step():
    """Model eğitim işlemini gerçekleştirir"""
    console.rule("[bold blue]Model Eğitimi[/]")
    
    # Output klasörünü oluştur
    model_dir = f"{OUTPUT_DIR}/models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Model tipini seç
    model_type = questionary.select(
        "Hangi modeli eğitmek istiyorsunuz?",
        choices=[
            "1. Baseline (Logistic Regression)",
            "2. LightGBM",
            "3. Source Bazlı Model (Her kaynak için ayrı)",
            "4. Ensemble (Tüm modeller)",
        ]
    ).ask()
    
    # Hiperparametre optimizasyonu yap mı?
    use_optuna = questionary.confirm(
        "Hiperparametre optimizasyonu yapmak ister misiniz? (Optuna)",
        default=False
    ).ask()
    
    # Model tipine göre parametreleri belirle
    model_param = model_type[0]
    experiment_name = f"model_{model_param}_{int(time.time())}"
    
    # Komut oluştur
    command = f"python -m src.pipelines.train +model={model_param} +experiment_name={experiment_name}"
    
    # Optuna kullanılacaksa ekstra parametreler ekle
    if use_optuna:
        # Optimizasyon ayarlarını sor
        n_trials = questionary.text(
            "Kaç deneme yapılsın? (10-100 arası önerilir)",
            default="50"
        ).ask()
        
        timeout = questionary.text(
            "Maksimum çalışma süresi (saniye)? (3600=1 saat)",
            default="3600"
        ).ask()
        
        metric = questionary.select(
            "Optimizasyon metriği?",
            choices=["roc_auc", "f1", "precision", "recall", "average_precision"]
        ).ask()
        
        command += f" +use_optuna=true +n_trials={n_trials} +timeout={timeout} +metric={metric}"
    
    console.print(f"[yellow]Model eğitimi başlıyor (Model Tipi: {model_type}, Optuna: {use_optuna})...[/]")
    
    # İlerleme göstergesiyle komutu çalıştır
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Model eğitimi çalışıyor...", total=None)
        try:
            result = run_command(command)
            progress.update(task, completed=True)
            
            if result.returncode != 0:
                console.print("[bold red]Model eğitim hatası![/]")
                console.print(f"[red]Çıktı: {result.stdout}[/]")
                console.print(f"[red]Hata: {result.stderr}[/]")
                return
                
            console.print(":tada: [bold green]Model eğitimi tamamlandı[/]")
            console.print(f"[cyan]Model ve sonuçlar: {model_dir} dizininde[/]")
            
        except Exception as e:
            progress.update(task, completed=True)
            logger.exception("Model eğitimi çalıştırılırken hata")
            console.print(f"[bold red]Hata: {str(e)}[/]")
            return
    
    # Eğitilmiş modelin sonuçlarını göster
    try:
        from src.utils.model_metrics import get_latest_metrics
        metrics_data = get_latest_metrics(experiment_name)
        
        if metrics_data:
            table = Table(title=f"Model Performans Metrikleri - {experiment_name}")
            table.add_column("Metrik", style="cyan")
            table.add_column("Değer", style="green")
            
            for metric_name, metric_value in metrics_data.items():
                table.add_row(metric_name, f"{metric_value:.4f}" if isinstance(metric_value, float) else str(metric_value))
            
            console.print(table)
    except Exception as e:
        logger.warning(f"Model metrikleri görüntülenirken hata: {e}")
        console.print("[yellow]Model metrikleri görüntülenemiyor, ancak model eğitimi tamamlandı.[/]")
    
    # MLflow'da gösterme seçeneği
    if questionary.confirm("Sonuçları MLflow'da görmek ister misiniz?").ask():
        _start_mlflow()

def _full_run():
    """Tüm pipeline adımlarını çalıştırır"""
    console.rule("[bold blue]FULL RUN - Tüm Adımlar[/]")
    
    # Yeni run klasörü oluşturup oluşturmayacağımızı sor
    create_new_run = questionary.confirm(
        "Yeni bir run klasörü oluşturmak istiyor musunuz?",
        default=True  # Full run için varsayılanı True yapıyoruz
    ).ask()
    
    # Experiment adını sor
    experiment_name = questionary.text(
        "Experiment için bir isim girin:",
        default=f"full_run_{int(time.time())}"
    ).ask()
    
    # Progress göster
    steps = [
        ("Veri Hazırlama", f"python -m src.pipelines.split {' --create-new-run' if create_new_run else ''}"),
        ("Feature Importance", f"python -m src.pipelines.feature_importance --train-path={OUTPUT_DIR}/split/train.csv")
    ]
    
    results_ok = True
    
    for step_name, command in steps:
        console.print(f"[bold yellow]{step_name} adımı çalıştırılıyor...[/]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"[cyan]{step_name} çalışıyor...", total=None)
            try:
                result = run_command(command)
                progress.update(task, completed=True)
                
                if result.returncode != 0:
                    console.print(f"[bold red]{step_name} adımı başarısız oldu![/]")
                    console.print(f"[red]Çıktı: {result.stdout}[/]")
                    console.print(f"[red]Hata: {result.stderr}[/]")
                    
                    # Kullanıcıya sorma
                    if not questionary.confirm("Hata oluştu. Devam etmek istiyor musunuz?", default=False).ask():
                        console.print("[bold red]İşlem kullanıcı tarafından durduruldu.[/]")
                        return
                    
                    results_ok = False
                else:
                    console.print(f"[green]{step_name} tamamlandı ✓[/]")
                    
            except Exception as e:
                progress.update(task, completed=True)
                logger.exception(f"{step_name} adımı çalıştırılırken hata")
                console.print(f"[bold red]Hata: {str(e)}[/]")
                
                # Kullanıcıya sorma
                if not questionary.confirm("Hata oluştu. Devam etmek istiyor musunuz?", default=False).ask():
                    console.print("[bold red]İşlem kullanıcı tarafından durduruldu.[/]")
                    return
                
                results_ok = False
    
    # Feature importance sonrası akıllı özellik seçimi adımını ekle
    if questionary.confirm("Akıllı özellik seçimi adımını çalıştırmak ister misiniz?", default=True).ask():
        console.print("[bold yellow]Akıllı özellik seçimi adımı çalıştırılıyor...[/]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Akıllı özellik seçimi çalışıyor...", total=None)
            try:
                # Mevcut aktif run klasörünü kullan
                from src.utils.paths import get_experiment_dir
                exp_dir = get_experiment_dir()
                
                auto_select_cmd = [
                    "python", "lead_scoring.py", "auto-select",
                    f"--input-file={OUTPUT_DIR}/split/train.csv",
                    f"--output-dir={exp_dir}/auto_select"
                ]
                result = run_command(" ".join(auto_select_cmd))
                progress.update(task, completed=True)
                
                if result.returncode != 0:
                    console.print("[bold red]Akıllı özellik seçimi adımı başarısız oldu![/]")
                    console.print(f"[red]Çıktı: {result.stdout}[/]")
                    console.print(f"[red]Hata: {result.stderr}[/]")
                    
                    # Kullanıcıya sorma
                    if not questionary.confirm("Hata oluştu. Devam etmek istiyor musunuz?", default=False).ask():
                        console.print("[bold red]İşlem kullanıcı tarafından durduruldu.[/]")
                        return
                else:
                    console.print("[green]Akıllı özellik seçimi tamamlandı ✓[/]")
                    
            except Exception as e:
                progress.update(task, completed=True)
                logger.exception("Akıllı özellik seçiminde hata")
                console.print(f"[bold red]Hata: {str(e)}[/]")
                
                # Kullanıcıya sorma
                if not questionary.confirm("Hata oluştu. Devam etmek istiyor musunuz?", default=False).ask():
                    console.print("[bold red]İşlem kullanıcı tarafından durduruldu.[/]")
                    return
    
    # Experiment adını da güncelle
    try:
        from src.utils.config_updater import update_experiment_config
        update_experiment_config(experiment_name=experiment_name)
        console.print(f"[green]Experiment adı '{experiment_name}' olarak güncellendi.[/]")
    except Exception as e:
        logger.exception("Experiment adı güncellenirken hata")
        console.print(f"[bold red]Experiment adı güncellenirken hata: {str(e)}[/]")
        if not questionary.confirm("Experiment adı güncellenirken hata oluştu. Devam etmek istiyor musunuz?", default=False).ask():
            console.print("[bold red]İşlem kullanıcı tarafından durduruldu.[/]")
            return
    
    # Kalan adımları çalıştır
    remaining_steps = [
        ("İstatistiksel Analiz", "python -m src.pipelines.stats --type=4"),
        ("Model Eğitimi", f"python -m src.pipelines.train")  # experiment_name parametresi kaldırıldı, yaml'dan alınacak
    ]
    
    for step_name, command in remaining_steps:
        console.print(f"[bold yellow]{step_name} adımı çalıştırılıyor...[/]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"[cyan]{step_name} çalışıyor...", total=None)
            try:
                result = run_command(command)
                progress.update(task, completed=True)
                
                if result.returncode != 0:
                    console.print(f"[bold red]{step_name} adımı başarısız oldu![/]")
                    console.print(f"[red]Çıktı: {result.stdout}[/]")
                    console.print(f"[red]Hata: {result.stderr}[/]")
                    
                    # Kullanıcıya sorma
                    if not questionary.confirm("Hata oluştu. Devam etmek istiyor musunuz?", default=False).ask():
                        console.print("[bold red]İşlem kullanıcı tarafından durduruldu.[/]")
                        return
                    
                    results_ok = False
                else:
                    console.print(f"[green]{step_name} tamamlandı ✓[/]")
                    
            except Exception as e:
                progress.update(task, completed=True)
                logger.exception(f"{step_name} adımı çalıştırılırken hata")
                console.print(f"[bold red]Hata: {str(e)}[/]")
                
                # Kullanıcıya sorma
                if not questionary.confirm("Hata oluştu. Devam etmek istiyor musunuz?", default=False).ask():
                    console.print("[bold red]İşlem kullanıcı tarafından durduruldu.[/]")
                    return
                
                results_ok = False
    
    if results_ok:
        console.print(":tada: [bold green]Tüm adımlar başarıyla tamamlandı![/]")
    else:
        console.print("[yellow]İşlem tamamlandı, ancak bazı adımlarda hatalar oluştu.[/]")
        
    console.print(f"[cyan]Experiment adı: {experiment_name}[/]")
    
    # Servisleri başlatma
    if questionary.confirm("MLflow & Streamlit servislerini başlatmak ister misiniz?").ask():
        _start_services()

def _view_results():
    """Mevcut sonuçları görüntüler"""
    console.rule("[bold blue]Sonuçları Görüntüle[/]")
    
    # Output klasörlerini listele
    output_dirs = [d for d in os.listdir(OUTPUT_DIR) 
                  if os.path.isdir(os.path.join(OUTPUT_DIR, d))]
    
    if not output_dirs:
        console.print("[bold red]Hiç sonuç bulunamadı![/]")
        return
    
    # Sonuçları sırala (en yeniler üstte)
    try:
        # Timestamp olanları sırala
        timestamp_dirs = []
        other_dirs = []
        
        for d in output_dirs:
            if d.startswith("run_") and d[4:].isdigit():
                # run_1234567890 formatındaki klasörler
                timestamp_dirs.append((int(d[4:]), d))
            else:
                other_dirs.append(d)
        
        # Timestamp'e göre sırala (yeniden eskiye)
        timestamp_dirs.sort(reverse=True)
        
        # Listede ilk 5 yeni çalıştırma, sonra diğer klasörler, en son da sabit klasörler
        sorted_dirs = [d[1] for d in timestamp_dirs[:5]]
        if len(timestamp_dirs) > 5:
            sorted_dirs.append("... Daha Eski Çalıştırmalar ...")
            sorted_dirs.extend([d[1] for d in timestamp_dirs[5:]])
        sorted_dirs.extend(other_dirs)
        
    except Exception as e:
        logger.warning(f"Sonuç klasörleri sıralanırken hata: {e}")
        sorted_dirs = output_dirs
    
    # Ek filtreleme seçenekleri
    filter_options = [
        "Tüm Sonuçlar",
        "Son 5 Çalıştırma",
        "Sadece Başarılı Çalıştırmalar",
        "Özel Filtreleme..."
    ]
    
    filter_choice = questionary.select(
        "Sonuçları nasıl filtrelemek istiyorsunuz?",
        choices=filter_options
    ).ask()
    
    filtered_dirs = []
    
    if filter_choice == "Tüm Sonuçlar":
        filtered_dirs = sorted_dirs
    elif filter_choice == "Son 5 Çalıştırma":
        filtered_dirs = [d for d in sorted_dirs if not d.startswith("...")][:5]
    elif filter_choice == "Sadece Başarılı Çalıştırmalar":
        # Başarılı çalıştırmaları belirle (burada sadece model klasörü olanları başarılı kabul ediyoruz)
        filtered_dirs = []
        for d in sorted_dirs:
            if d.startswith("..."):
                continue
            model_dir = os.path.join(OUTPUT_DIR, d, "models")
            if os.path.isdir(model_dir) and os.listdir(model_dir):
                filtered_dirs.append(d)
    elif filter_choice == "Özel Filtreleme...":
        keyword = questionary.text(
            "Filtrelemek için bir anahtar kelime girin:"
        ).ask()
        filtered_dirs = [d for d in sorted_dirs if keyword.lower() in d.lower()]
    
    if not filtered_dirs:
        console.print("[bold red]Filtreleme kriterlerine uyan sonuç bulunamadı![/]")
        return
    
    # Filtrelenmiş sonuçları göster ve birini seç
    result_type = questionary.select(
        "Hangi sonuçları görüntülemek istiyorsunuz?",
        choices=filtered_dirs + ["Listeyi Yeniden Filtrele"]
    ).ask()
    
    if result_type == "Listeyi Yeniden Filtrele":
        return _view_results()
    
    if result_type.startswith("..."):
        console.print("[yellow]Lütfen spesifik bir klasör seçin.[/]")
        return _view_results()
    
    # Seçilen klasördeki içeriği göster
    dir_path = os.path.join(OUTPUT_DIR, result_type)
    
    # Alt klasörleri listele
    subdirs = [d for d in os.listdir(dir_path) 
              if os.path.isdir(os.path.join(dir_path, d))]
    
    if subdirs:
        table = Table(title=f"{result_type} Klasöründeki Alt Klasörler")
        table.add_column("Klasör", style="cyan")
        table.add_column("Dosya Sayısı", style="green")
        table.add_column("Toplam Boyut (MB)", style="magenta")
        
        for d in subdirs:
            subdir_path = os.path.join(dir_path, d)
            files = [f for f in os.listdir(subdir_path) 
                    if os.path.isfile(os.path.join(subdir_path, f))]
            total_size_mb = sum(os.path.getsize(os.path.join(subdir_path, f)) 
                               for f in files) / (1024 * 1024) if files else 0
            table.add_row(d, str(len(files)), f"{total_size_mb:.2f}")
        
        console.print(table)
        
        # Alt klasör seçme seçeneği
        subdir_choice = questionary.select(
            "Bir alt klasörün içeriğini görüntülemek ister misiniz?",
            choices=["Hayır"] + subdirs
        ).ask()
        
        if subdir_choice != "Hayır":
            subdir_path = os.path.join(dir_path, subdir_choice)
            files = [f for f in os.listdir(subdir_path) 
                    if os.path.isfile(os.path.join(subdir_path, f))]
            
            if files:
                table = Table(title=f"{result_type}/{subdir_choice} Klasöründeki Dosyalar")
                table.add_column("Dosya Adı", style="cyan")
                table.add_column("Boyut (KB)", style="green")
                table.add_column("Son Değişiklik", style="yellow")
                
                for f in sorted(files):
                    file_path = os.path.join(subdir_path, f)
                    size_kb = os.path.getsize(file_path) / 1024
                    mod_time = time.strftime('%Y-%m-%d %H:%M:%S', 
                                           time.localtime(os.path.getmtime(file_path)))
                    table.add_row(f, f"{size_kb:.2f}", mod_time)
                
                console.print(table)
            else:
                console.print("[yellow]Bu klasörde dosya bulunmuyor.[/]")
    
    # Klasördeki dosyaları listele
    files = [f for f in os.listdir(dir_path) 
            if os.path.isfile(os.path.join(dir_path, f))]
    
    if files:
        table = Table(title=f"{result_type} Klasöründeki Dosyalar")
        table.add_column("Dosya Adı", style="cyan")
        table.add_column("Boyut (KB)", style="green")
        table.add_column("Son Değişiklik", style="yellow")
        
        for f in sorted(files):
            file_path = os.path.join(dir_path, f)
            size_kb = os.path.getsize(file_path) / 1024
            mod_time = time.strftime('%Y-%m-%d %H:%M:%S', 
                                   time.localtime(os.path.getmtime(file_path)))
            table.add_row(f, f"{size_kb:.2f}", mod_time)
        
        console.print(table)
    
    # Çıktı temizleme seçeneği
    if questionary.confirm("Bu çalıştırmanın çıktılarını temizlemek ister misiniz?", default=False).ask():
        if questionary.confirm(f"[bold red]DİKKAT: {result_type} klasörü ve tüm içeriği silinecek! Emin misiniz?[/]", default=False).ask():
            try:
                import shutil
                shutil.rmtree(dir_path)
                console.print(f"[green]'{result_type}' klasörü başarıyla silindi.[/]")
            except Exception as e:
                logger.exception(f"Klasör silinirken hata: {e}")
                console.print(f"[bold red]Hata: Klasör silinemedi! {str(e)}[/]")
    
    # Servisleri başlatma
    if questionary.confirm("Sonuçları Streamlit ve MLflow'da görmek ister misiniz?").ask():
        _start_services()

def _start_services():
    """MLflow ve Streamlit servislerini başlatır"""
    console.rule("[bold blue]Servisler Başlatılıyor[/]")
    
    services = questionary.checkbox(
        "Hangi servisleri başlatmak istiyorsunuz?",
        choices=[
            questionary.Choice("MLflow UI", checked=True),
            questionary.Choice("Streamlit Dashboard", checked=True)
        ]
    ).ask()
    
    mlflow_process = None
    streamlit_process = None
    
    for service in services:
        if service == "MLflow UI":
            console.print("[yellow]MLflow UI başlatılıyor...[/]")
            
            # Port kontrolü
            if _is_port_in_use(MLFLOW_PORT):
                console.print(f"[bold yellow]Uyarı: {MLFLOW_PORT} portu zaten kullanımda![/]")
                
                new_port = questionary.text(
                    f"Başka bir port numarası girin (varsayılan: {MLFLOW_PORT + 1}):",
                    default=str(MLFLOW_PORT + 1)
                ).ask()
                
                try:
                    new_port = int(new_port)
                    if _is_port_in_use(new_port):
                        console.print(f"[bold red]Hata: {new_port} portu da kullanımda! MLflow başlatılamıyor.[/]")
                        continue
                    mlflow_port = new_port
                except ValueError:
                    console.print("[bold red]Geçersiz port numarası! MLflow başlatılamıyor.[/]")
                    continue
            else:
                mlflow_port = MLFLOW_PORT
            
            try:
                # MLflow'u başlat
                mlflow_process = run_command(
                    f"mlflow ui --port {mlflow_port}", 
                    background=True
                )
                
                console.print(f"[green]MLflow UI başlatıldı ✓[/]")
                console.print(f"[bold cyan]📊 MLflow UI → http://localhost:{mlflow_port}[/]")
                
                if questionary.confirm("Tarayıcıda MLflow UI'ı açmak ister misiniz?").ask():
                    webbrowser.open(f"http://localhost:{mlflow_port}")
                    
            except Exception as e:
                logger.exception("MLflow başlatılırken hata")
                console.print(f"[bold red]MLflow başlatılırken hata: {str(e)}[/]")
        
        if service == "Streamlit Dashboard":
            console.print("[yellow]Streamlit Dashboard başlatılıyor...[/]")
            
            # Port kontrolü
            if _is_port_in_use(STREAMLIT_PORT):
                console.print(f"[bold yellow]Uyarı: {STREAMLIT_PORT} portu zaten kullanımda![/]")
                
                new_port = questionary.text(
                    f"Başka bir port numarası girin (varsayılan: {STREAMLIT_PORT + 1}):",
                    default=str(STREAMLIT_PORT + 1)
                ).ask()
                
                try:
                    new_port = int(new_port)
                    if _is_port_in_use(new_port):
                        console.print(f"[bold red]Hata: {new_port} portu da kullanımda! Streamlit başlatılamıyor.[/]")
                        continue
                    streamlit_port = new_port
                except ValueError:
                    console.print("[bold red]Geçersiz port numarası! Streamlit başlatılamıyor.[/]")
                    continue
            else:
                streamlit_port = STREAMLIT_PORT
            
            try:
                # Streamlit'i başlat
                streamlit_process = run_command(
                    f"streamlit run streamlit_app/app.py --server.port {streamlit_port}", 
                    background=True
                )
                
                console.print(f"[green]Streamlit Dashboard başlatıldı ✓[/]")
                console.print(f"[bold cyan]📈 Streamlit dashboard → http://localhost:{streamlit_port}[/]")
                
                if questionary.confirm("Tarayıcıda Streamlit Dashboard'u açmak ister misiniz?").ask():
                    webbrowser.open(f"http://localhost:{streamlit_port}")
                    
            except Exception as e:
                logger.exception("Streamlit başlatılırken hata")
                console.print(f"[bold red]Streamlit başlatılırken hata: {str(e)}[/]")
    
    # Servisleri açık tut
    if mlflow_process or streamlit_process:
        console.print("[yellow]Servisler arka planda çalışıyor. Kapatmak için CTRL+C tuşlarına basın.[/]")
        try:
            # Kullanıcı çıkana kadar bekle
            while True:
                if questionary.confirm("Servisleri kapatmak istiyor musunuz?").ask():
                    break
                time.sleep(5)
        except KeyboardInterrupt:
            pass
        finally:
            # Servisleri kapat
            if mlflow_process:
                mlflow_process.terminate()
            if streamlit_process:
                streamlit_process.terminate()
            console.print("[red]Servisler kapatıldı.[/]")

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

def _start_mlflow():
    """MLflow servisini başlatır"""
    console.print("[yellow]MLflow UI başlatılıyor...[/]")
    
    # Port kontrolü
    if _is_port_in_use(MLFLOW_PORT):
        console.print(f"[bold yellow]Uyarı: {MLFLOW_PORT} portu zaten kullanımda![/]")
        
        new_port = questionary.text(
            f"Başka bir port numarası girin (varsayılan: {MLFLOW_PORT + 1}):",
            default=str(MLFLOW_PORT + 1)
        ).ask()
        
        try:
            new_port = int(new_port)
            if new_port < 1024 or new_port > 65535:
                console.print("[bold red]Port numarası 1024-65535 arasında olmalıdır![/]")
                return
            
            config["mlflow_port"] = new_port
            
        except ValueError:
            console.print("[bold red]Geçersiz port numarası![/]")
            return
            
    # Yapılandırmayı kaydet
    try:
        with open("configs/app_config.yaml", "w") as f:
            yaml.dump(CONFIG, f)
        
        console.print("[green]Yapılandırma başarıyla güncellendi.[/]")
        console.print("[yellow]Değişikliklerin geçerli olması için uygulamayı yeniden başlatın.[/]")
        
    except Exception as e:
        logger.exception("Yapılandırma güncellenirken hata")
        console.print(f"[bold red]Hata: {str(e)}[/]")

def _start_streamlit():
    """Streamlit servisini başlatır"""
    console.print("[yellow]Streamlit Dashboard başlatılıyor...[/]")
    
    # Port kontrolü
    if _is_port_in_use(STREAMLIT_PORT):
        console.print(f"[bold yellow]Uyarı: {STREAMLIT_PORT} portu zaten kullanımda![/]")
        
        new_port = questionary.text(
            f"Başka bir port numarası girin (varsayılan: {STREAMLIT_PORT + 1}):",
            default=str(STREAMLIT_PORT + 1)
        ).ask()
        
        try:
            new_port = int(new_port)
            if new_port < 1024 or new_port > 65535:
                console.print("[bold red]Port numarası 1024-65535 arasında olmalıdır![/]")
                return
            
            config["streamlit_port"] = new_port
            
        except ValueError:
            console.print("[bold red]Geçersiz port numarası! Streamlit başlatılamıyor.[/]")
            return
    else:
        streamlit_port = STREAMLIT_PORT
        
    try:
        # Streamlit'i başlat
        streamlit_process = run_command(
            f"streamlit run streamlit_app/app.py --server.port {streamlit_port}", 
            background=True
        )
        
        console.print(f"[green]Streamlit Dashboard başlatıldı ✓[/]")
        console.print(f"[bold cyan]📈 Streamlit dashboard → http://localhost:{streamlit_port}[/]")
        
        webbrowser.open(f"http://localhost:{streamlit_port}")
        
        # Servis başarıyla başlatıldı, kullanıcıya beklemesi gerektiğini bildir
        console.print("[yellow]Streamlit Dashboard arka planda çalışıyor. Ana menüye dönmek için herhangi bir tuşa basın...[/]")
        input()
    except Exception as e:
        logger.exception("Streamlit başlatılırken hata")
        console.print(f"[bold red]Streamlit başlatılırken hata: {str(e)}[/]")

def _cleanup_outputs():
    """Eski çıktıları temizler"""
    console.rule("[bold blue]Çıktı Temizleme[/]")
    
    # Output klasörlerini listele
    output_dirs = [d for d in os.listdir(OUTPUT_DIR) 
                  if os.path.isdir(os.path.join(OUTPUT_DIR, d))]
    
    if not output_dirs:
        console.print("[bold red]Hiç sonuç bulunamadı![/]")
        return
    
    # Temizleme seçenekleri
    cleanup_options = [
        "Tüm çıktıları temizle",
        "Belirli bir tarihten öncekileri temizle",
        "Başarısız çalıştırmaları temizle",
        "Belirli bir klasörü temizle",
        "İptal"
    ]
    
    cleanup_choice = questionary.select(
        "Hangi çıktıları temizlemek istiyorsunuz?",
        choices=cleanup_options
    ).ask()
    
    if cleanup_choice == "İptal":
        return
    
    if cleanup_choice == "Tüm çıktıları temizle":
        if questionary.confirm(
            f"[bold red]DİKKAT: {OUTPUT_DIR} altındaki TÜM klasörler silinecek! Emin misiniz?[/]", 
            default=False
        ).ask():
            try:
                import shutil
                for d in output_dirs:
                    dir_path = os.path.join(OUTPUT_DIR, d)
                    shutil.rmtree(dir_path)
                console.print("[green]Tüm çıktılar başarıyla temizlendi.[/]")
            except Exception as e:
                logger.exception("Çıktılar temizlenirken hata")
                console.print(f"[bold red]Hata: {str(e)}[/]")
    
    elif cleanup_choice == "Belirli bir tarihten öncekileri temizle":
        # Timestamp formatındaki klasörleri bul
        timestamp_dirs = []
        for d in output_dirs:
            if d.startswith("run_") and d[4:].isdigit():
                timestamp_dirs.append((int(d[4:]), d))
        
        if not timestamp_dirs:
            console.print("[bold red]Tarih bazlı klasör bulunamadı![/]")
            return
        
        # Tarihleri sırala ve göster
        timestamp_dirs.sort(reverse=True)
        
        date_table = Table(title="Mevcut Çalıştırma Tarihleri")
        date_table.add_column("Sıra", style="cyan")
        date_table.add_column("Klasör", style="green")
        date_table.add_column("Tarih", style="magenta")
        
        for i, (ts, dir_name) in enumerate(timestamp_dirs, 1):
            date_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))
            date_table.add_row(str(i), dir_name, date_str)
        
        console.print(date_table)
        
        # Kaç gün öncekileri temizlemek istediğini sor
        days = questionary.text(
            "Kaç gün önceki çalıştırmaları temizlemek istiyorsunuz?",
            default="30"
        ).ask()
        
        try:
            days = int(days)
            if days <= 0:
                raise ValueError("Pozitif bir sayı girilmelidir")
                
            # Eşik tarihi hesapla
            threshold_ts = time.time() - (days * 24 * 60 * 60)
            
            # Silinecek klasörleri belirle
            to_delete = []
            for ts, dir_name in timestamp_dirs:
                if ts < threshold_ts:
                    to_delete.append(dir_name)
            
            if not to_delete:
                console.print(f"[yellow]{days} gün öncesine ait çalıştırma bulunamadı.[/]")
                return
            
            # Onay al
            if questionary.confirm(
                f"[bold yellow]{len(to_delete)} adet klasör silinecek. Onaylıyor musunuz?[/]", 
                default=False
            ).ask():
                import shutil
                for dir_name in to_delete:
                    dir_path = os.path.join(OUTPUT_DIR, dir_name)
                    shutil.rmtree(dir_path)
                console.print(f"[green]{len(to_delete)} adet klasör başarıyla silindi.[/]")
            
        except ValueError as ve:
            console.print(f"[bold red]Geçersiz sayı: {str(ve)}![/]")
            return
    
    elif cleanup_choice == "Başarısız çalıştırmaları temizle":
        # Başarısız çalıştırmaları tespit et (model klasörü olmayanlar)
        failed_runs = []
        for d in output_dirs:
            model_dir = os.path.join(OUTPUT_DIR, d, "models")
            if not os.path.isdir(model_dir) or not os.listdir(model_dir):
                failed_runs.append(d)
        
        if not failed_runs:
            console.print("[yellow]Başarısız çalıştırma bulunamadı.[/]")
            return
        
        # Listeyi göster
        console.print(f"[yellow]Başarısız olarak tespit edilen {len(failed_runs)} çalıştırma:[/]")
        for run in failed_runs:
            console.print(f"- {run}")
        
        # Onay al
        if questionary.confirm(
            f"[bold yellow]Bu {len(failed_runs)} çalıştırmayı silmek istiyor musunuz?[/]", 
            default=False
        ).ask():
            try:
                import shutil
                for run in failed_runs:
                    dir_path = os.path.join(OUTPUT_DIR, run)
                    shutil.rmtree(dir_path)
                console.print(f"[green]{len(failed_runs)} başarısız çalıştırma temizlendi.[/]")
            except Exception as e:
                logger.exception("Başarısız çalıştırmalar temizlenirken hata")
                console.print(f"[bold red]Hata: {str(e)}[/]")
    
    elif cleanup_choice == "Belirli bir klasörü temizle":
        # Klasör seçtir
        dir_choice = questionary.select(
            "Hangi klasörü temizlemek istiyorsunuz?",
            choices=output_dirs
        ).ask()
        
        # Onay al
        if questionary.confirm(
            f"[bold red]DİKKAT: {dir_choice} klasörü silinecek! Emin misiniz?[/]", 
            default=False
        ).ask():
            try:
                import shutil
                dir_path = os.path.join(OUTPUT_DIR, dir_choice)
                shutil.rmtree(dir_path)
                console.print(f"[green]'{dir_choice}' klasörü başarıyla silindi.[/]")
            except Exception as e:
                logger.exception(f"Klasör silinirken hata: {e}")
                console.print(f"[bold red]Hata: Klasör silinemedi! {str(e)}[/]")

def _update_config():
    """Uygulama konfigürasyonunu günceller"""
    console.rule("[bold blue]Konfigürasyon Güncelleme[/]")
    
    # Mevcut yapılandırmayı göster
    console.print("[yellow]Mevcut yapılandırma:[/]")
    console.print(f"MLflow port: [cyan]{MLFLOW_PORT}[/]")
    console.print(f"Streamlit port: [cyan]{STREAMLIT_PORT}[/]")
    console.print(f"Çıktı dizini: [cyan]{OUTPUT_DIR}[/]")
    console.print(f"Geçici dizin: [cyan]{CONFIG.get('tmp_dir', 'tmp')}[/]")
    
    # Güncelleme seçenekleri
    update_options = [
        "MLflow port numarasını güncelle",
        "Streamlit port numarasını güncelle",
        "Çıktı dizinini güncelle",
        "Varsayılanlara döndür",
        "İptal"
    ]
    
    update_choice = questionary.select(
        "Neyi güncellemek istiyorsunuz?",
        choices=update_options
    ).ask()
    
    if update_choice == "İptal":
        return
    
    config_path = Path("configs/app_config.yaml")
    config = CONFIG.copy()
    
    if update_choice == "MLflow port numarasını güncelle":
        new_port = questionary.text(
            "Yeni MLflow port numarası:",
            default=str(MLFLOW_PORT)
        ).ask()
        
        try:
            new_port = int(new_port)
            if new_port < 1024 or new_port > 65535:
                console.print("[bold red]Port numarası 1024-65535 arasında olmalıdır![/]")
                return
            
            config["mlflow_port"] = new_port
            
        except ValueError:
            console.print("[bold red]Geçersiz port numarası![/]")
            return
            
    elif update_choice == "Streamlit port numarasını güncelle":
        new_port = questionary.text(
            "Yeni Streamlit port numarası:",
            default=str(STREAMLIT_PORT)
        ).ask()
        
        try:
            new_port = int(new_port)
            if new_port < 1024 or new_port > 65535:
                console.print("[bold red]Port numarası 1024-65535 arasında olmalıdır![/]")
                return
            
            config["streamlit_port"] = new_port
            
        except ValueError:
            console.print("[bold red]Geçersiz port numarası![/]")
            return
            
    elif update_choice == "Çıktı dizinini güncelle":
        new_dir = questionary.text(
            "Yeni çıktı dizini:",
            default=OUTPUT_DIR
        ).ask()
        
        if not new_dir:
            console.print("[bold red]Geçersiz dizin adı![/]")
            return
        
        config["output_dir"] = new_dir
        
    elif update_choice == "Varsayılanlara döndür":
        config = {
            "mlflow_port": 5000,
            "streamlit_port": 8501,
            "output_dir": "outputs",
            "tmp_dir": "tmp"
        }
    
    # Yapılandırmayı kaydet
    try:
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        console.print("[green]Yapılandırma başarıyla güncellendi.[/]")
        console.print("[yellow]Değişikliklerin geçerli olması için uygulamayı yeniden başlatın.[/]")
        
    except Exception as e:
        logger.exception("Yapılandırma güncellenirken hata")
        console.print(f"[bold red]Hata: {str(e)}[/]")

if __name__ == "__main__":
    cli()
