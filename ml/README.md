# SaaS Customer Churn Intelligence – Python Veri Altyapısı ve EDA

Bu repository, B2B SaaS şirketleri için müşteri kaybı (*Customer Churn*) tahmini ve karar destek altyapısı sunan projenin **Python Veri İşleme, Doğrulama ve Keşifsel Veri Analizi (EDA)** katmanını içerir.

---

## 1. Mimari ve Klasör Yapısı

```
ml/
├── README.md                           # Proje dokümantasyonu ve iş akışı
├── requirements.txt                    # Python bağımlılıkları
├── data/
│   ├── raw/                            # Orijinal ham CSV veri setleri (Değiştirilmez)
│   │   ├── accounts.csv
│   │   ├── invoices.csv
│   │   ├── payments.csv
│   │   ├── plans.csv
│   │   └── subscriptions.csv
│   └── processed/
│       ├── tables/                     # Temizlenmiş ve tipleri normalize edilmiş tekil tablolar
│       │   ├── accounts.csv
│       │   ├── invoices.csv
│       │   ├── payments.csv
│       │   ├── plans.csv
│       │   └── subscriptions.csv
│       └── merged/                     # Analiz amaçlı birleştirilmiş veri seti (1 satır = 1 hesap/abone)
│           └── account_level_eda_dataset.csv
├── src/
│   ├── data/
│   │   ├── preprocess.py               # Veri ön işleme, tekilleştirme, tip ve format standardizasyonu
│   │   ├── validation.py               # Bütünlük, PK/FK, kural ve kapsam testleri
│   │   └── build_dataset.py            # Çoğalmayı önleyen 2 aşamalı birleştirme pipeline'ı
│   ├── eda/
│   │   └── eda.py                      # İstatistiksel analiz ve 12 adet görselleştirme üretici
│   ├── features/                       # (Gelecek geliştirme) Özellik mühendisliği
│   ├── models/                         # (Gelecek geliştirme) ML modelleri
│   └── evaluation/                     # (Gelecek geliştirme) Model değerlendirme
├── reports/
│   ├── cleaning/                       # Temizlik ve doğrulama JSON ve Markdown raporları
│   │   ├── cleaning_report.md
│   │   ├── cleaning_summary.json
│   │   ├── validation_report.md
│   │   ├── validation_report.json
│   │   └── dataset_build_summary.json
│   └── eda/                            # EDA özet raporları ve yüksek çözünürlüklü grafikler
│       ├── eda_summary_report.md
│       ├── eda_metrics.json
│       └── figures/
│           ├── 01_churn_distribution.png
│           ├── 02_churn_by_plan.png
│           ├── 03_churn_by_segment.png
│           ├── 04_churn_by_region.png
│           ├── 05_churn_by_acquisition_channel.png
│           ├── 06_mrr_distribution_by_churn.png
│           ├── 07_tenure_distribution_by_churn.png
│           ├── 08_missing_values_matrix.png
│           ├── 09_numeric_distributions.png
│           ├── 10_monthly_churn_trend.png
│           ├── 11_payment_failures_vs_churn_cohort.png
│           └── 12_invoice_coverage_breakdown.png
├── artifacts/                          # İleride eğitilecek modeller ve scaler nesneleri için
└── tests/                              # Pytest test paketi
    ├── test_preprocess.py
    ├── test_validation.py
    └── test_build_dataset.py
```

---

## 2. Kurulum ve Ortam Hazırlığı

Python 3.10+ sürümü önerilir. Sanal ortam oluşturup gerekli paketleri yükleyin:

```bash
# Sanal ortam oluşturma ve etkinleştirme
python3 -m venv .venv
source .venv/bin/activate

# Gerekli bağımlılıkları yükleme
pip install -r ml/requirements.txt
```

---

## 3. Çalıştırma Sırası ve İş Akışı

Pipeline adımları birbirini tamamlayacak şekilde tasarlanmıştır. Lütfen aşağıdaki sırayı takip edin:

### Adım 1: Veri Ön İşleme (Preprocessing)
Ham CSV dosyalarını okur, mükerrer kayıtları güvenle eler, boşluk ve büyük/küçük harf standartlarını uygular, tarihleri ISO formatına (`YYYY-MM-DD`) dönüştürür ve temizlenmiş tabloları kaydeder.

```bash
python ml/src/data/preprocess.py
# veya modül olarak:
python -m ml.src.data.preprocess
```
*Çıktılar*: `ml/data/processed/tables/*.csv`, `ml/reports/cleaning/cleaning_report.md`

### Adım 2: Veri Bütünlüğü Doğrulaması (Validation)
Birincil ve yabancı anahtar ilişkilerini (PK/FK), eksik değer politikalarını, kategorik alanları, kronolojik tarih tutarlılığını ve sayısal sınırları 30 ayrı kontrol ile doğrular.

```bash
python ml/src/data/validation.py
# veya modül olarak:
python -m ml.src.data.validation
```
*Çıktılar*: `ml/reports/cleaning/validation_report.md`, `ml/reports/cleaning/validation_report.json`

### Adım 3: Birleştirilmiş Analiz Veri Setinin Oluşturulması (Dataset Building)
Ödemeleri fatura bazında, faturaları da abonelik bazında 2 aşamalı olarak özetler. Satır çoğalmasını (fanout) önleyerek **1 Satır = 1 Hesap / Abonelik** tane seviyesinde 2,000 satırlık analiz veri setini üretir.

```bash
python ml/src/data/build_dataset.py
# veya modül olarak:
python -m ml.src.data.build_dataset
```
*Çıktılar*: `ml/data/processed/merged/account_level_eda_dataset.csv`, `ml/reports/cleaning/dataset_build_summary.json`

### Adım 4: Keşifsel Veri Analizi (EDA)
Kapsamlı istatistiksel analizler çalıştırır ve 12 adet görsel grafik ile detaylı markdown özet raporu oluşturur.

```bash
python ml/src/eda/eda.py
# veya modül olarak:
python -m ml.src.eda.eda
```
*Çıktılar*: `ml/reports/eda/figures/*.png`, `ml/reports/eda/eda_summary_report.md`

### Adım 5: Otomatik Testlerin Çalıştırılması (Testing)
Veri işleme, bütünlük ve birleştirme kurallarının doğruluğunu `pytest` ile test eder.

```bash
pytest ml/tests -v
```

---

## 4. Temizlik ve Eksik Değer Politikası

- **Mükerrer Kayıtlar**: Her ham tabloda tam 1 adet tam mükerrer satır tespit edilmiş ve `keep='first'` prensibiyle güvenle ayıklanmıştır.
- **`accounts.region` (479 eksik)**: Coğrafi bölge verisi yapay olarak doldurulmamış, analitik tarafsızlığı korumak adına `NaN` olarak bırakılmıştır. Yapılan analizde eksik bölgeye sahip hesapların churn oranı (%21.09), bilinen bölgelerle neredeyse farksızdır (MCAR yapısı).
- **`subscriptions.ended_at` (1,590 eksik)**: Bu eksiklikler veri hatası değildir; aktif aboneliklerin henüz iptal edilmediğini ifade eder. 410 iptal edilmiş aboneliğin tamamında bitiş tarihi eksiksiz mevcuttur.
- **Eksik Tarihler**: Tablolarda 1'er adet bulunan eksik tarihler tahmine dayalı olarak doldurulmamıştır.

---

## 5. Kritik Veri Sınırlılığı ve Analiz Notu

> [!WARNING]
> **Fatura ve Ödeme Kapsam Kısıtı (Invoice Coverage Disparity)**:
> `invoices.csv` içerisindeki 2,000 fatura kaydı ve `payments.csv` içerisindeki 2,000 ödeme denemesi **yalnızca ilk 120 abonelik (`S000001` - `S000120`)** ile ilişkilidir. Geriye kalan 1,880 abonelikte (%94.0) hiçbir fatura ve ödeme geçmişi bulunmamaktadır.
> Bu nedenle:
> - `account_level_eda_dataset.csv` dosyası **keşifsel analiz (EDA) amaçlıdır**; henüz doğrudan model eğitimine hazır kabul edilmemelidir.
> - İleride ML özellikleri üretilirken ödeme metriklerinin tüm popülasyonu temsil etmediği göz önünde bulundurulmalıdır.

---

## 6. Kapsam Dışı Bırakılan Unsurlar (Sonraki Aşamalar)

Bu fazda kasıtlı olarak kapsam dışı bırakılan ve sonraki adımlarda geliştirilecek bileşenler:
- Makine öğrenmesi modelleri (Logistic Regression, Random Forest, XGBoost vb.)
- Model hiperparametre optimizasyonu ve çapraz doğrulama
- Model tahmin API'si
- C# / ASP.NET Core Web API backend servisi
- Next.js kullanıcı arayüzü (frontend)
- SQL Server veritabanı şeması ve entegrasyonu

---

## 7. Orijinal Veri Seti Lisansı ve Kaynak Gösterimi

Bu veri seti **Creative Commons Attribution 4.0 International (CC BY 4.0)** lisansı altında lisanslanmıştır.

- **Lisans Metni**: [https://creativecommons.org/licenses/by/4.0/legalcode](https://creativecommons.org/licenses/by/4.0/legalcode)
- **Önerilen Kaynak Gösterimi (Attribution)**:
  > Analytics Engineering, "SaaS Subscriptions & Customer Churn Dataset v1.0.0". https://www.analyticsengineering.com/datasets/saas-subscriptions
