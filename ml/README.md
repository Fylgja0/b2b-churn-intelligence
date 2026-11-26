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
│       ├── merged/                     # Analiz amaçlı birleştirilmiş veri seti (1 satır = 1 hesap/abone)
│       │   └── account_level_eda_dataset.csv
│       └── modeling/                   # Point-in-time güvenli modelleme veri seti (T0=2024-12-31)
│           └── churn_modeling_dataset.csv
├── src/
│   ├── data/
│   │   ├── preprocess.py               # Veri ön işleme, tekilleştirme, tip ve format standardizasyonu
│   │   ├── validation.py               # Güçlendirilmiş bütünlük, PK/FK, kural, tarih ve kapsam testleri (39 kontrol)
│   │   └── build_dataset.py            # Çoğalmayı önleyen 2 aşamalı birleştirme pipeline'ı
│   ├── eda/
│   │   └── eda.py                      # İstatistiksel analiz ve 12 adet görselleştirme üretici
│   ├── features/
│   │   └── build_features.py           # Zaman noktası tabanlı sızıntısız özellik mühendisliği ve kohort oluşturucu
│   ├── models/                         # Ortak ML modelleme altyapısı (Eğitim henüz başlatılmadı)
│   │   ├── __init__.py
│   │   ├── experiment_config.py        # feature_manifest.json tabanlı senaryo konfigürasyonu
│   │   ├── preprocessing.py            # Sızıntısız scikit-learn ColumnTransformer & Pipeline
│   │   ├── model_registry.py           # Genişletilebilir baseline model kayıt defteri
│   │   ├── cross_validation.py         # Deterministik StratifiedKFold ve OOF tahmin motoru
│   │   └── train.py                    # Tek ortak deney çalıştırıcı ve CLI
│   └── evaluation/                     # Merkezi metrik, görselleştirme ve senaryo karşılaştırma
│       ├── __init__.py
│       ├── metrics.py                  # Önceliklendirilmiş ikili sınıflandırma metrikleri
│       ├── plots.py                    # PR/ROC eğrileri ve senaryo karşılaştırma grafikleri
│       └── compare_experiments.py      # region_included vs region_excluded karşılaştırıcı
├── reports/
│   ├── cleaning/                       # Temizlik ve doğrulama JSON ve Markdown raporları
│   │   ├── cleaning_report.md
│   │   ├── cleaning_summary.json
│   │   ├── validation_report.md
│   │   ├── validation_report.json
│   │   └── dataset_build_summary.json
│   ├── eda/                            # EDA özet raporları ve yüksek çözünürlüklü grafikler
│   │   ├── eda_summary_report.md
│   │   ├── eda_metrics.json
│   │   └── figures/
│   │       ├── 01_churn_distribution.png
│   │       ├── 02_churn_by_plan.png
│   │       ├── 03_churn_by_segment.png
│   │       ├── 04_churn_by_region.png
│   │       ├── 05_churn_by_acquisition_channel.png
│   │       ├── 06_mrr_distribution_by_churn.png
│   │       ├── 07_tenure_distribution_by_churn.png
│   │       ├── 08_missing_values_matrix.png
│   │       ├── 09_numeric_distributions.png
│   │       ├── 10_monthly_churn_trend.png
│   │       ├── 11_payment_failures_vs_churn_cohort.png
│   │       └── 12_invoice_coverage_breakdown.png
│   └── modeling/                       # Modelleme kohortu, feature manifest ve deney rapor alanları
│       ├── modeling_dataset_report.md
│       ├── modeling_dataset_summary.json
│       ├── feature_manifest.json
│       └── experiments/                # (Eğitim çalıştırıldığında üretilecek çıktılar için hazır alan)
│           ├── region_included/
│           ├── region_excluded/
│           └── comparison/
├── artifacts/
│   └── models/                         # (Eğitim çalıştırıldığında kaydedilecek .joblib modeller)
│       ├── region_included/
│       └── region_excluded/
└── tests/                              # Pytest test paketi (100 adet pozitif ve negatif test)
    ├── test_preprocess.py              # Ham veri varlığı, temizlik, tekilleştirme ve tip testleri (5 test)
    ├── test_validation.py              # 39 doğrulama kontrolü, dinamik kardinalite ve 18 negatif test (28 test)
    ├── test_build_dataset.py           # 2 aşamalı birleştirme ve tane seviyesi testleri (4 test)
    ├── test_eda.py                     # Ki-kare, Cochran kuralı, Cochran ihlali ve dejenere matris testleri (6 test)
    ├── test_features.py                # Kohort, hedef pencereleri, leakage koruması ve zamansal testler (22 test)
    ├── test_vize_dashboard.py          # Vize dashboard ve bulgu raporu tutarlılık testleri (6 test)
    ├── test_models.py                  # Model registry, senaryo ayrımı, leakage ve CV/OOF testleri (16 test)
    └── test_evaluation.py              # Metrik doğruluğu, sınıf dağılımı, grafik ve karşılaştırma testleri (13 test)
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
Tablolar arası ilişkileri ve veri bütünlüğünü 39 ayrı kontrol ile doğrular:
- **Foreign Key Kontrolleri**: Hem `NULL` yabancı anahtarları hem de eşleşmeyen (yetim) kayıtları ayrı ayrı denetler.
- **Gerçek Takvim ve Tarih Eşleşme Doğrulaması**: Yalnızca regex biçimine değil, gerçek takvim geçerliliğine (`datetime.strptime`) bakar (`2025-02-31` gibi hataları yakalar). Ham ve temiz tarihin aynı takvim gününü temsil ettiğini doğrular; başlangıçta eksik olanları, geçersiz olanları ve eşleşmeyenleri ayrı sayar.
- **Kategorik Politikalar ve Eksik Sütun Denetimi**: `plans.plan_name` ("Starter", "Growth", "Scale", "Enterprise") dahil tüm kategorik alanları denetler. Zorunlu kategorik alanlarda (`subscriptions.status`, `accounts.acquisition_channel`, `plans.plan_name`) `NULL` değerleri veya tablodan sütunun tamamen silinmesini hata olarak işaretler; izinli alanlarda (`accounts.region`) ise `NULL` varlığına kontrollü onay verir.
- **Kronolojik Tutarlılık**: Abonelik ve fatura tarihlerinin mantıksal sıralamasını denetler (`started_at <= ended_at`).
- **Kapsam ve Kohort Doğrulaması (Dinamik 1:1 Kardinalite)**: 1:1 hesap-abonelik ilişkisini sabit 2.000 satır koşulu olmaksızın, dinamik küme teorisi ve sayaçlar (`missing_accounts`, `unexpected_accounts`, `duplicate_account_refs`, `duplicate_accounts_count`, `null_account_pks_count`, `row_count_diff`) ile değerlendirir. `mismatch_count` formülü mükerrer sayımlardan arındırılmış gerçek hata sayısını yansıtır.

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
Kapsamlı tanımlayıcı istatistikler ve bağımsızlık analizleri çalıştırır; Scipy kütüphanesi ile $\chi^2$ test istatistiği, serbestlik derecesi, p-değeri (`correction=False`) ve Cochran beklenen hücre frekansı varsayımlarını dinamik olarak hesaplar. 12 adet görsel grafik ile detaylı markdown özet raporu oluşturur.

```bash
python ml/src/eda/eda.py
# veya modül olarak:
python -m ml.src.eda.eda
```
*Çıktılar*: `ml/reports/eda/figures/*.png`, `ml/reports/eda/eda_summary_report.md`, `ml/reports/eda/eda_metrics.json`

### Adım 5: Zaman Noktası Tabanlı Özellik Mühendisliği (Feature Engineering)
Tahminin yapıldığı an olan $T_0$ (`2024-12-31`) itibarıyla aktif olan 1.242 aboneliği kohorta dahil eder, hedef sızıntısını (target leakage) engellemek amacıyla doğrudan sızıntı alanlarını (`ended_at`, `status`, `churn`, `tenure_days`) hariç tutarak 90, 180 ve 365 günlük ileriye dönük churn hedeflerini (`target_churn_90d`, `target_churn_180d`, `target_churn_365d`) üretir; zamansal güvenilirliği doğrulanmış veya açık iş varsayımlarıyla sabit kabul edilmiş öznitelikleri türetir ve taşınabilir göreli yollarla detaylı dokümantasyon manifestini oluşturur.

```bash
python ml/src/features/build_features.py
# veya modül olarak:
python -m ml.src.features.build_features
```
*Çıktılar*: `ml/data/processed/modeling/churn_modeling_dataset.csv`, `ml/reports/modeling/modeling_dataset_report.md`, `ml/reports/modeling/modeling_dataset_summary.json`, `ml/reports/modeling/feature_manifest.json`

### Adım 6: Otomatik Testlerin Çalıştırılması (Testing)
Veri işleme, bütünlük ve birleştirme kurallarının yanı sıra EDA varsayım kontrolleri, zaman noktası tabanlı özellik mühendisliği, vize dashboard tutarlılığı ve yeni ML modelleme / değerlendirme altyapısını `pytest` ile test eder. Test paketi pozitif ve negatif kontroller, dinamik kardinalite testleri, Cochran varsayım denetimleri, kohort sınır koşulları, leakage izolasyon testleri, senaryo konfigürasyonu, CV/OOF determinizmi ve metrik doğrulamasını içerir (**Toplam 100 test**).

```bash
pytest ml/tests -v
```

---

## 4. Temizlik ve Eksik Değer Politikası

- **Mükerrer Kayıtlar**: Her ham tabloda tam 1 adet tam mükerrer satır tespit edilmiş ve `keep='first'` prensibiyle güvenle ayıklanmıştır.
- **`accounts.region` (479 eksik)**: Coğrafi bölge verisi yapay olarak doldurulmamış, analitik tarafsızlığı korumak adına `NaN` olarak bırakılmıştır. Eksik bölgeye sahip hesapların churn oranı (%21.09), bilinen bölgelerle (APAC: %21.60, LATAM: %20.36, EMEA: %19.04) sayısal olarak benzerdir. Gözlemlenebilir değişkenlerle yapılan Ki-kare testlerinde anlamlı bir ilişki bulunamamış olmakla birlikte, gözlemlenemeyen etkenlerin eksikliği tetikleyip tetiklemediği bilinemeyeceğinden **eksikliğin kesin olarak MCAR olduğu iddia edilemez**; eksiklik mekanizması mevcut veriler ışığında **belirsiz** olarak değerlendirilmelidir.
- **`subscriptions.ended_at` (1,590 eksik)**: Bu eksiklikler veri hatası değildir; aktif aboneliklerin henüz iptal edilmediğini ifade eder. 410 iptal edilmiş aboneliğin tamamında bitiş tarihi eksiksiz mevcuttur.
- **Eksik Tarihler**: Tablolarda 1'er adet bulunan başlangıçta eksik tarihler tahmine dayalı olarak doldurulmamış, analitik şeffaflık adına `NaT` olarak korunmuştur.

---

## 5. Kritik Analitik Uyarılar ve Modelleme Sınırları

### 5.1. Fatura ve Ödeme Kapsam Kısıtı (Invoice Coverage Disparity)
> [!WARNING]
> `invoices.csv` içerisindeki 2,000 fatura kaydı ve `payments.csv` içerisindeki 2,000 ödeme denemesi **yalnızca ilk 120 abonelik (`S000001` - `S000120`)** ile ilişkilidir. Geriye kalan 1,880 abonelikte (%94.0) hiçbir fatura ve ödeme geçmişi bulunmamaktadır.
> - `account_level_eda_dataset.csv` dosyası **yalnızca keşifsel analiz (EDA) amaçlıdır**; doğrudan model eğitimine hazır kabul edilmemelidir.
> - İleride ML modelleri geliştirilirken ödeme ve fatura öznitelikleri tüm popülasyona genellenemez.

### 5.2. Tenure Değişkeni ve Hedef Sızıntısı (Target Leakage) Riski
> [!CAUTION]
> Veri setindeki `tenure_days` değişkeni:
> - İptal edilmiş müşterilerde: `ended_at - started_at`
> - Aktif müşterilerde: `2025-12-31 - started_at`
> olarak hesaplanmıştır (farklı referans noktaları kullanılmıştır).
> 
> Bu hesaplama tanımlayıcı analiz (EDA) için uygundur; ancak **tahmin modelinde doğrudan girdi özniteliği olarak kullanılırsa Hedef Sızıntısına (Target Leakage) yol açar**, çünkü `ended_at` bilgisi gelecekte gerçekleşen churn olayının sonucudur. Model geliştirmeden önce bir tahmin referans zamanı ($T_0$) sabitlenmeli ve tenure yalnızca $T_0$ anına kadar geçen süre olarak türetilmelidir.

### 5.3. Aylık Churn Grafiği: Hacim vs Oran Ayrımı
> [!NOTE]
> `10_monthly_churn_trend.png` grafiği aylık churn oranını değil, aylık **iptal edilen abonelik sayısını (hacmini)** gösterir. Her ay risk altındaki müşteri tabanı tanımlanmadan bir oran türetmek yanıltıcı olacağından ve sentetik veride mevsimsellik veya nedensellik hakkında yeterli kanıt bulunmadığından kesin çıkarımlardan kaçınılmalıdır.

---

## 6. Makine Öğrenmesi Modelleme Altyapısı

> [!IMPORTANT]
> **Durum Notu – Model Eğitimi Henüz Başlatılmadı**:
> Modelleme, çapraz doğrulama, değerlendirme ve senaryo karşılaştırma altyapısı kurulmuş ve sentetik veri tabanlı otomatik testlerle doğrulanmıştır. **Gerçek veri seti (`churn_modeling_dataset.csv`) üzerinde baseline model eğitim deneyleri henüz çalıştırılmamıştır** ve bir sonraki aşamada ayrıca başlatılacaktır.

### 6.1. Tek Ortak Mimari ve Manifest Kaynağı
- Tüm özellik, metadata, hedef ve hariç tutulan (leakage/unverified) sütun tanımları tek kaynak olan `ml/reports/modeling/feature_manifest.json` dosyasından dinamik olarak okunur (`ml/src/models/experiment_config.py`).
- Metadata alanları (`subscription_id`, `account_id`) ve hedef/sızıntı alanları hiçbir koşulda model girdisi olarak kullanılmaz; ID sütunları yalnızca out-of-fold (OOF) tahmin izlenebilirliği için çıktı tablolarına eklenir.
- Üç hedef değişkenin (`target_churn_90d`, `target_churn_180d`, `target_churn_365d`) tamamı ayrı ayrı modellenebilir; ana modelleme hedefi **`target_churn_365d`** olarak belirlenmiştir.

### 6.2. İki Deney Senaryosu (Bölge Özelliği Etki Karşılaştırması)
İki senaryo için kopyalanmış kod yerine tek ortak altyapı üzerinden senaryo konfigürasyonu kullanılır:
- **Senaryo A (`region_included`)**: `region` kategorik özellik olarak modele dahil edilir; 321 eksik kayıt ham CSV üzerinde değiştirilmez, pipeline içinde `Unknown` kategorisine dönüştürülür. Aynı eksiklik bilgisinin iki kez verilmesini önlemek için `region_missing_flag` bu senaryoda **hariç tutulur** (11 özellik: 4 kategorik, 7 sayısal).
- **Senaryo B (`region_excluded`)**: `region` ve `region_missing_flag` özellikleri model girdilerinden tamamen çıkarılır; hiçbir müşteri kaydı silinmez (10 özellik: 3 kategorik, 7 sayısal).
- **Adil Karşılaştırma İlkesi**: Her iki senaryoda da aynı 1.242 gözlem, aynı hedefler, aynı CV fold indeksleri, aynı algoritmalar/hiperparametreler ve aynı değerlendirme metrikleri kullanılır.

### 6.3. Baseline Model Registry (`ml/src/models/model_registry.py`)
İlk baseline aşamasında SMOTE, class weighting veya hiperparametre optimizasyonu uygulanmadan aşağıdaki 4 algoritma desteklenir (yeni algoritmalar `ModelRegistry.register(...)` ile kolayca eklenebilir):
1. **`dummy`** (*Referans Model*): `DummyClassifier(strategy="prior")`
2. **`logistic_regression`**: `LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)` (varsayılan sınıf ağırlığı)
3. **`random_forest`**: `RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)`
4. **`gradient_boosting`**: `GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)`

### 6.4. Sızıntısız Veri Ön İşleme (`ml/src/models/preprocessing.py`)
- Tüm dönüşümler scikit-learn `Pipeline` ve `ColumnTransformer(remainder="drop")` içinde tanımlanır ve yalnızca ilgili eğitim fold'unda fit edilir.
- **Kategorik özellikler**: `SimpleImputer(strategy="constant", fill_value="Unknown")` + `OneHotEncoder(handle_unknown="ignore")`.
- **Sayısal özellikler**: `SimpleImputer(strategy="median")`; `StandardScaler` yalnızca ölçekleme gerektiren modellerde (`logistic_regression`) uygulanır, ağaç tabanlı modellerde uygulanmaz.

### 6.5. Çapraz Doğrulama ve OOF Tahminleri (`ml/src/models/cross_validation.py`)
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` kullanılır. Mevcut veri setinde her müşteri için tek abonelik gözlemi bulunduğundan başlangıçta `GroupKFold` kullanılmamıştır.
- Her hedef değişken için üretilen fold indeksleri iki bölge senaryosunda da birebir aynıdır.
- Özellikle `target_churn_90d` hedefinde yalnızca 56 pozitif örnek bulunduğundan, her fold için eğitim/doğrulama sınıf dağılımları ve pozitif sayıları raporlanır.
- Her gözlem kendi doğrulama fold'undan tam olarak bir kez out-of-fold (OOF) tahmini alır (`subscription_id`, `account_id`, `target`, `model`, `y_true`, `y_probability`, `y_pred_0_5`, `fold`).

### 6.6. Değerlendirme Metrikleri ve Karşılaştırma (`ml/src/evaluation/`)
Öncelik sırası:
1. **Average Precision (AP)** (`average_precision_score` — **birincil karşılaştırma metriği**)
2. **ROC-AUC**
3. **Recall** (`0.5` karar eşiğinde — optimize edilmemiş baseline eşiği)
4. **Precision** (`0.5` karar eşiğinde — optimize edilmemiş baseline eşiği)
5. **F1-score** (`0.5` karar eşiğinde — optimize edilmemiş baseline eşiği)
6. **Brier Score** (olasılık kalibrasyonu; düşük olması daha iyidir)

Accuracy, sınıf dengesizliği nedeniyle model seçim kriteri olarak kullanılmaz. Sonraki aşamada baseline deneylerini çalıştırmak için kullanılacak CLI komutları:

```bash
# Tek senaryo çalıştırma
python -m ml.src.models.train --scenario region_included
python -m ml.src.models.train --scenario region_excluded

# Her iki senaryoyu birlikte çalıştırıp karşılaştırma raporu üretme
python -m ml.src.models.train --scenario all --compare
```

---

## 7. Kapsam Dışı Bırakılan Unsurlar (Sonraki Aşamalar)

Bu fazda kasıtlı olarak kapsam dışı bırakılan ve sonraki adımlarda yürütülecek çalışmalar:
- Gerçek veri seti üzerinde baseline model eğitim deneylerinin çalıştırılması ve analiz edilmesi
- Hiperparametre optimizasyonu, SMOTE, class weighting ve karar eşiği (threshold) optimizasyonu
- Ek gradient boosting kütüphaneleri (XGBoost, LightGBM, CatBoost)
- Model tahmin API'si
- C# / ASP.NET Core Web API backend servisi
- Next.js kullanıcı arayüzü (frontend)
- SQL Server veritabanı şeması ve entegrasyonu

---

## 8. Orijinal Veri Seti Lisansı ve Kaynak Gösterimi

Bu veri seti **Creative Commons Attribution 4.0 International (CC BY 4.0)** lisansı altında lisanslanmıştır.

- **Lisans Metni**: [https://creativecommons.org/licenses/by/4.0/legalcode](https://creativecommons.org/licenses/by/4.0/legalcode)
- **Önerilen Kaynak Gösterimi (Attribution)**:
  > Analytics Engineering, "SaaS Subscriptions & Customer Churn Dataset v1.0.0". https://www.analyticsengineering.com/datasets/saas-subscriptions
