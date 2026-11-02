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
│   │   ├── validation.py               # Güçlendirilmiş bütünlük, PK/FK, kural, tarih ve kapsam testleri (38 kontrol)
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
└── tests/                              # Pytest test paketi (42 adet pozitif ve negatif test)
    ├── test_preprocess.py              # Ham veri varlığı, temizlik, tekilleştirme ve tip testleri (5 test)
    ├── test_validation.py              # 39 doğrulama kontrolü, dinamik kardinalite ve 18 negatif test (28 test)
    ├── test_build_dataset.py           # 2 aşamalı birleştirme ve tane seviyesi testleri (4 test)
    └── test_eda.py                     # Ki-kare, Cochran kuralı, Cochran ihlali ve dejenere matris testleri (5 test)
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

### Adım 5: Otomatik Testlerin Çalıştırılması (Testing)
Veri işleme, bütünlük ve birleştirme kurallarının doğruluğunu `pytest` ile test eder. Test paketi hem beklenen veri üzerindeki pozitif kontrolleri hem de bozulmuş veri senaryolarını yakalayan **18 negatif testi**, dinamik kardinalite testlerini ve Cochran varsayım ihlali senaryosunu içerir (**Toplam 42 test**).

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
