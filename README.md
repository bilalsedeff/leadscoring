# Lead Scoring Pipeline

## Son Güncellemeler

### 02.06.2025 - Split Otomatizasyonu ve Hydra İyileştirmeleri
- İstatistiksel testler ve diğer komutlar için otomatik split oluşturma özelliği eklendi
- Split verisi bulunamadığında otomatik split işlemi çalıştırılabilir hale geldi (`--auto_split` parametresi)
- Train.py'da Hydra yapılandırması düzeltildi ve YAML dosyalarından parametre okuma iyileştirildi
- Rolling window hesaplamaları güçlendirildi ve hata yakalama mekanizmaları geliştirildi
- Engineering.py'da temporal özellik oluşturma daha sağlam hale getirildi

### 01.06.2025 - İstatistiksel Testlerde İyileştirmeler
- Eğitim verisi bulunamadığında otomatik veri yükleme ve kesme (cutoff) değerlerini kullanma özelliği eklendi
- Unit testler eklendi ve test kapsamı genişletildi
- Parametre kontrolü ve hata yakalama geliştirilerek daha sağlam bir yapı oluşturuldu
- İstatistiksel test komutlarına `--train_cutoff` ve `--val_cutoff` parametreleri eklendi

### 31.05.2025 - Özellik Önemi Görselleştirmesi İyileştirmeleri
- Daha anlamlı kararlılık ölçümleri için threshold ve top-k bazlı metrikler eklendi
- Boxplot ve violin plot görselleştirmeleri eklendi
- SHAP beeswarm, dependence ve force plotları iyileştirildi
- Bellek kullanımı ve işlem süresi optimizasyonları yapıldı

### 30.05.2025 - Hesap Kesişim Sorunu Düzeltildi
- Train, validation ve test setlerinde hesap (account_Id) kesişimi sorunu düzeltildi
- TimeGroupSplitter minimum zaman değerine göre sıralama yapılarak kesin atama sağlandı
- Grup bütünlüğü korunurken veri sızıntısını önleyen otomatik kontrol mekanizması eklendi
- T-test'in otomatik olarak ANOVA'ya yönlendirilmesi için ikiden fazla kategori durumu yönetildi

## Proje Hakkında
Lead Scoring projesi, müşteri adaylarının (lead) dönüşüm olasılığını tahmin etmek için geliştirilmiş CLI tabanlı, MLflow ve Streamlit görselleştirmeli bir machine learning projesidir. Conversion_Datamart.csv dosyasındaki veriler üzerinde çalışarak, potansiyel müşterileri düşük, orta ve yüksek potansiyelli olarak sınıflandırır.

## Temel Özellikler
- Zaman bazlı train/test split (hesap bütünlüğünü korur)
- Otomatik istatistiksel analizler ve öznitelik seçimi
- Kaynak bazlı modelleme (Web, App, Ecommerce, Social Media)
- Model kalibrasyonu ve segment etiketleme
- Kapsamlı metrik değerlendirmesi ve görselleştirme
- Zengin CLI arayüzü
- MLflow model takibi
- Streamlit interaktif dashboard

## Kurulum

```bash
# Gerekli paketleri yükle
pip install -r requirements.txt

# veya lead_scoring.py ile kur
python lead_scoring.py setup
```

## Hızlı Başlangıç

```bash
# Veri hazırlama ve bölme
python lead_scoring.py run

# İstatistiksel analizler
python lead_scoring.py stats --test_type=chi_square
python lead_scoring.py stats --test_type=correlation

# Özellik önemi analizi
python lead_scoring.py feature-imp --method=shap

# Model eğitimi
python lead_scoring.py train --model_type=2  # LightGBM

# Dashboard başlatma
python lead_scoring.py dashboard
```

## Detaylı CLI Komutları

### Veri Hazırlama ve Bölme
```bash
# Temel komut - yeni bir run klasörü oluşturur
python lead_scoring.py run

# Gelişmiş parametreler ile
python lead_scoring.py run --split_method=time --target_col=Target_IsConverted --train_cutoff=2024-06-30 --val_cutoff=2024-11-30 --drop_id_cols
```

### İstatistiksel Analizler
```bash
# Ki-kare testi
python lead_scoring.py stats --test_type=chi_square

# T-testi (kaynak türüne göre)
python lead_scoring.py stats --test_type=t_test --group_col=Source_Final__c

# ANOVA (kaynak türüne göre)
python lead_scoring.py stats --test_type=anova --group_col=Source_Final__c

# Korelasyon analizi
python lead_scoring.py stats --test_type=correlation

# Dönüşüm oranı karşılaştırması
python lead_scoring.py stats --test_type=conversion_rate

# Tüm testler
python lead_scoring.py stats --test_type=all

# Cutoff değerleri belirterek ham veriden analiz
python lead_scoring.py stats --test_type=all --train_cutoff=72024 --val_cutoff=122024
```

### Özellik Önemi ve Seçimi
```bash
# SHAP bazlı özellik önemi
python lead_scoring.py feature-imp --method=shap

# Permutation bazlı özellik önemi
python lead_scoring.py feature-imp --method=permutation

# Akıllı özellik seçimi
python lead_scoring.py auto-select
```

### Model Eğitimi
```bash
# Baseline (LogisticRegression)
python lead_scoring.py train --model_type=1

# LightGBM
python lead_scoring.py train --model_type=2

# Source-specific (her kaynak için ayrı model)
python lead_scoring.py train --model_type=3

# Ensemble (varsayılan)
python lead_scoring.py train --model_type=4
```

### Tahmin ve Değerlendirme
```bash
# Test seti üzerinde tahmin
python lead_scoring.py predict --input_file=data/test.csv --model_path=outputs/run_XXX/models/model_name
```

### Dashboard ve Görselleştirme
```bash
# Streamlit dashboard
python lead_scoring.py dashboard

# MLflow UI
python lead_scoring.py mlflow
```

### Tam Pipeline
```bash
# Tüm adımları sırayla çalıştırma
python lead_scoring.py all

# Yeni bir run klasöründe tüm adımları çalıştırma
python lead_scoring.py all --create_new_run
```

## Dosya Yapısı

```
pipeline/
  ├── configs/                 # Yapılandırma dosyaları
  ├── data/                    # Veri dosyaları
  │   ├── raw/                 # Ham veri
  │   ├── interim/             # Ara işlemiş veri
  │   └── processed/           # İşlenmiş veri
  ├── notebooks/               # Jupyter notebooks
  ├── outputs/                 # Çıktı dosyaları
  │   ├── run_TIMESTAMP/       # Her çalıştırma için zaman damgalı klasör
  │   │   ├── feature_importance/
  │   │   ├── models/
  │   │   ├── statistical_tests/
  │   │   └── logs/
  │   └── split/               # Bölünmüş veri setleri
  ├── src/                     # Kaynak kod
  │   ├── calibration/         # Model kalibrasyonu
  │   ├── cli/                 # CLI uygulaması
  │   ├── evaluation/          # Metrik ve değerlendirme
  │   ├── features/            # Özellik mühendisliği
  │   ├── imbalance/           # Dengesizlik yönetimi
  │   ├── ingestion/           # Veri yükleme
  │   ├── models/              # Model tanımları
  │   ├── pipelines/           # Pipeline tanımları
  │   ├── preprocessing/       # Veri önişleme
  │   ├── registry/            # Model kayıt
  │   ├── tests/               # Birim testler
  │   └── utils/               # Yardımcı fonksiyonlar
  ├── streamlit_app/           # Streamlit uygulaması
  ├── lead_scoring.py          # Ana giriş noktası
  └── requirements.txt         # Bağımlılıklar
```

## Katkıda Bulunma

1. Bu repo'yu fork edin
2. Feature branch'ınızı oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some amazing feature'`)
4. Branch'ınıza push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın 