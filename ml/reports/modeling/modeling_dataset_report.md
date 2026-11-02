# SaaS Churn Intelligence – Zaman Noktası Tabanlı Modelleme Veri Seti Raporu

## 1. Modelleme Senaryosu ve Yönetici Özeti

Bu çalışma, B2B SaaS müşteri kaybı (*Customer Churn*) tahmin modellerinin eğitilmesi için gerekli olan **zaman noktasına duyarlı (Point-in-Time Safe)** modelleme veri setinin oluşturulmasını ve metodolojik dokümantasyonunu kapsamaktadır. Makine öğrenmesi modellerinde en sık karşılaşılan metodolojik tuzaklardan biri olan **Hedef Sızıntısı (Target Leakage)** riskini azaltmak amacıyla, öznitelikler tahmin anında ($T_0$) bilinebilecek veya iş mantığı altında sabit kaldığı varsayılan bilgilerden türetilmiştir.

- **Gözlem Tarihi ($T_0$ / Observation Date)**: `2024-12-31` (Günün sonu)
- **Beyan Edilen Veri Kapsamı Bitiş Tarihi**: `2025-12-31`
- **Modelleme Kohortu Büyüklüğü**: 1,242 aktif abonelik
- **Üretilen Veri Seti**: `ml/data/processed/modeling/churn_modeling_dataset.csv`
- **Özellik Manifestosu**: `ml/reports/modeling/feature_manifest.json`

---

## 2. Zaman Noktası ($T_0$), Tahmin Ufukları ve Veri Kapsamı

Modelleme kohortu, 31 Aralık 2024 gecesi itibarıyla aktif olan müşterilerden oluşturulmuştur. Aynı gözlem anı için üç farklı karar penceresini temsil eden ileriye dönük ikili churn hedef değişkenleri üretilmiştir:

| Hedef Değişken | Tahmin Ufku | Başlangıç Tarihi | Bitiş Tarihi (Dahil) | Mantıksal Koşul |
|---|:---:|:---:|:---:|---|
| `target_churn_90d` | 90 gün | 2024-12-31 | 2025-03-31 | `2024-12-31 < ended_at <= 2025-03-31` |
| `target_churn_180d` | 180 gün | 2024-12-31 | 2025-06-29 | `2024-12-31 < ended_at <= 2025-06-29` |
| `target_churn_365d` | 365 gün | 2024-12-31 | 2025-12-31 | `2024-12-31 < ended_at <= 2025-12-31` |

> [!NOTE]
> **Pencere Bitiş Tarihleri**: 2025 yılı artık yıl değildir (Şubat 28 gündür). 90 gün bitişi 31 Mart 2025, 180 gün bitişi 29 Haziran 2025 ve 365 gün bitişi 31 Aralık 2025'tir. Pencerelerin bitiş günlerinde gerçekleşen churn olayları hedef pozitif olarak sayılmıştır.

### 2.1. Beyan Edilen Veri Kapsamı ile Satır Bazlı Gözlem Ayrımı

- **Veri Kaynağı**: *Analytics Engineering – SaaS Subscriptions & Customer Churn Dataset v1.0.0*
- **Beyan Edilen Kapsam**: `2023-01-01` ile `2025-12-31` arası.
- **Kapsam Dayanağı (`coverage_basis`)**: `declared_by_source` (Veri kaynağı tarafından beyan edilmiş takvim aralığı).
- **Bağımsız Doğrulama (`independently_verified`)**: `false` (Dış bağımsız denetim logu veya transaction watermark bulunmamaktadır).
- **Satır Düzeyinde Gözlemlenen Maksimum Başlangıç Tarihi (`observed_max_started_at`)**: `2025-10-25`
- **Satır Düzeyinde Gözlemlenen Maksimum Bitiş Tarihi (`observed_max_ended_at`)**: `2025-12-30`

> [!IMPORTANT]
> **Gözlem Bütünlüğü ve Negatif Hedef Etiketleme Varsayımı**:
> - Kaynak CSV dosyalarında bağımsız extraction timestamp veya veri akışı bütünlük watermark'ı yer almamaktadır.
> - Tablodaki maksimum `ended_at` değerinin `2025-12-30` olması, veri kesim tarihi olan `2025-12-31` gününde iptal gerçekleşmemiş olmasından kaynaklanabilir; tek başına bir eksiklik kanıtı değildir.
> - Ancak `ended_at` değeri `NULL` olan (veya hedef penceresinden sonra sonlanan) aboneliklerin negatif (0) olarak etiketlenmesi, veri kaynağının 2025 yılı boyunca tüm iptal olaylarını eksiksiz yakaladığı (**sağdan sansürleme olmadığı**) varsayımına dayanmaktadır.

---

## 3. Kohort Seçim Kriterleri ve Elenen Kayıtlar

Bir aboneliğin modelleme kohortuna dahil edilmesi için temel koşul, $T_0$ anında aktif olmasıdır:
$$\text{started\_at} \le T_0 \quad \land \quad (\text{ended\_at is NULL} \quad \lor \quad \text{ended\_at} > T_0)$$

### Kohort Dağılım Tablosu
| Kategori | Kayıt Sayısı | Oran (%) | Dahil Edilme Durumu |
|---|---:|---:|:---:|
| **Modelleme Kohortuna Dahil** | **1,242** | **%62.10** | **Dahil Edildi ✅** |
| $T_0$ Tarihinde veya Öncesinde Sona Ermiş | 186 | %9.30 | Hariç Tutuldu ❌ |
| $T_0$ Sonrasında Başlamış | 571 | %28.55 | Hariç Tutuldu ❌ |
| Başlangıç Tarihi Eksik (`started_at is NULL`) | 1 | %0.05 | Hariç Tutuldu ❌ |
| **Toplam Temizlenmiş Abonelik** | **2,000** | **%100.00** | - |

### Status ve ended_at Tutarlılık Denetimi
- `status == 'active'` iken `ended_at` dolu kayıt sayısı: **0**
- `status == 'cancelled'` iken `ended_at` boş kayıt sayısı: **0**
- Kohorttaki 1.242 aboneliğin 182'sinde nihai `status == 'cancelled'` ve `ended_at > 2024-12-31` görülmektedir. Bu bir tutarsızlık değildir; 2025 yılı içerisinde gerçekleşen ve tahmin edilmek istenen müşteri kaybı olaylarını temsil eder.

---

## 4. Churn Hedef Değişkenlerinin Dağılımı ve Sınıf Dengesizliği (Class Imbalance)

Modelleme kohortu üzerinde hesaplanan hedef değişken dağılımları:

| Hedef Değişken | Pozitif (Churn = 1) | Negatif (Aktif = 0) | Toplam | Pozitif Oran (%) | Dengesizlik Oranı |
|---|---:|---:|---:|---:|:---:|
| `target_churn_90d` | 56 | 1,186 | 1,242 | **%4.51** | ~1:21 |
| `target_churn_180d` | 108 | 1,134 | 1,242 | **%8.70** | ~1:10 |
| `target_churn_365d` | 182 | 1,060 | 1,242 | **%14.65** | ~1:5 |

> [!IMPORTANT]
> **Kritik Metodolojik Uyarı – Sınıf Dengesizliği**:
> - Özellikle 90 günlük hedefte pozitif sınıf oranı yalnızca **%4,51** seviyesindedir (21 negatif örneğe 1 pozitif örnek).
> - İleride model eğitimi ve değerlendirmesi yapılırken **Accuracy (Doğruluk)** metriği kesinlikle yanıltıcı olacaktır; çünkü tüm tahminleri 0 veren naif bir model %95,49 doğruluk elde eder.
> - Gelecek modelleme aşamasında **PR-AUC (Precision-Recall AUC)**, **ROC-AUC**, **Recall@k** ve **F-beta** metrikleri temel alınmalıdır.
> - Hiyerarşik Tutarlılık Doğrulaması: Bütün satırlarda `target_churn_90d <= target_churn_180d <= target_churn_365d` kuralının sağlandığı matematiksel olarak kanıtlanmıştır.

---

## 5. Point-in-Time Özellikler (Feature Engineering)

Modelleme veri setine dahil edilen tüm öznitelikler $T_0$ anı itibarıyla dondurulmuştur:

| Özellik Adı | Tip | Kaynak | Zamansal Güvenilirlik | Açıklama |
|---|:---:|:---:|:---:|---|
| `region` | `string` | `accounts` | Assumed | Müşteri coğrafi bölgesi (APAC, EMEA, LATAM). Eksikler NULL korunmuştur. |
| `region_missing_flag` | `integer` | `accounts` | Assumed | Bölge verisi eksik mi? (1 = Eksik, 0 = Mevcut). |
| `segment` | `string` | `accounts` | Assumed | Şirket ölçek segmenti (SMB, Mid-Market, Enterprise). |
| `acquisition_channel` | `string` | `accounts` | Verified | İlk temas edinim kanalı (organic, paid, partner, referral, outbound). |
| `account_age_days_at_t0` | `integer` | `accounts` | Verified | $T_0$ ile hesap açılış tarihi (`created_at`) arasındaki gün sayısı ($T_0 - \text{created\_at}$). |
| `account_created_at_missing_flag` | `integer` | `accounts` | Verified | Hesap açılış tarihi eksiklik bayrağı (1 = Eksik, 0 = Mevcut). |
| `subscription_tenure_days_at_t0` | `integer` | `subscriptions` | Verified | $T_0$ anındaki abonelik kıdemi ($T_0 - \text{started\_at}$). Hedef sızıntısından arındırılmıştır. |
| `subscription_start_month` | `integer` | `subscriptions` | Verified | Abonelik başlangıç ayı (1 - 12). Mevsimsellik göstergesi. |
| `subscription_start_quarter` | `integer` | `subscriptions` | Verified | Abonelik başlangıç çeyreği (1 - 4). |
| `plan_name` | `string` | `plans` | Assumed | Abone olunan plan adı (Starter, Growth, Scale, Enterprise). Değişmediği varsayılmıştır. |
| `plan_monthly_price` | `float` | `plans` | Assumed | Planın aylık liste fiyatı (`monthly_price`). Fiyatın geçmişte değişmediği varsayılmıştır. |
| `plan_included_seats` | `integer` | `plans` | Assumed | Plan dahilindeki koltuk sayısı (`included_seats`). Paket kapsamının değişmediği varsayılmıştır. |

---

## 6. Zamansal Güvenilirlik Sınıflandırması (Temporal Reliability)

Kaynak dokümantasyon ve veri mimarisi incelemeleri sonucunda özelliklerin zaman içindeki davranışları üç grupta sınıflandırılmıştır:

1. **Doğrulanmış (Verified)**: Zamansal olarak sabit olduğu kesin olan veya doğrudan $T_0$ ile geçmiş tarihler arasındaki farktan hesaplanan özellikler:
   - `acquisition_channel`: Müşterinin ilk edinim kanalı, tanımı gereği zamanla değişmeyen başlangıç olayıdır.
   - `account_age_days_at_t0`, `account_created_at_missing_flag`: $T_0$ ile `created_at` arasındaki matematiksel gün farkı.
   - `subscription_tenure_days_at_t0`: $T_0$ ile `started_at` arasındaki matematiksel gün farkı.
   - `subscription_start_month`, `subscription_start_quarter`: Başlangıç tarihinin takvim bileşenleri.
2. **Varsayıma Dayalı (Assumed)**: Kaynak veride geçmiş logu veya versiyonlama tutulmayan ancak iş mantığı gereği abonelik boyunca değişmediği varsayılan özellikler:
   - `region`, `region_missing_flag`: Müşteri şirket merkezinin abonelik süresince değişmediği varsayılmıştır.
   - `segment`: Şirket ölçeğinin (SMB, Mid-Market, Enterprise) $T_0$ öncesinde değişmediği varsayılmıştır.
   - `plan_name`, `plan_monthly_price`, `plan_included_seats`: Kaynak veride plan versiyonlaması veya geçmiş fiyatlandırma kayıtları bulunmamaktadır. Güncel plan kataloğundaki liste fiyatı ve dahil koltuk adedinin abonelik dönemi boyunca sabit kaldığı varsayılmıştır; fiyatların veya paket kapsamının geçmişte değişmiş olma riski metodolojik bir sınırlılık olarak değerlendirilmelidir.
3. **Doğrulanamayan (Unverified)**: $T_0$ tarihindeki değerini geçmişe dönük olarak garanti edemediğimiz alanlar (`seats`, `mrr`, `additional_seats`). Bu alanlar model veri setinden **tamamen çıkarılmıştır**.

---

## 7. Hedef Sızıntısı (Target Leakage) ve Hariç Tutulan Alanlar

Aşağıdaki alanlar modelleme feature matrix'ine **kesinlikle dahil edilmemiştir**:

| Kolon | Kategori | Hariç Tutulma Nedeni |
|---|:---:|---|
| `ended_at` | Target Leakage | İptal tarihini doğrudan içerir. Yalnızca hedef oluşturmak ve kohort seçmek için kullanılmıştır. |
| `status` | Target Leakage | 2025-12-31 tarihindeki nihai durumu yansıtır; churn olayını doğrudan sızdırır. |
| `churn` | Target Leakage | EDA aşamasındaki genel churn bayrağıdır; doğrudan hedefin kendisidir. |
| `tenure_days` | Target Leakage | EDA veri setindeki tenure hesabı iptal edilenlerde `ended_at - started_at` kullandığı için post-$T_0$ bilgisi sızdırır. |
| `seats` | Unverified Temporal | Satın alınan koltuk adedinin $T_0$ tarihindeki tarihsel değeri doğrulanamamıştır. |
| `mrr` | Unverified Temporal | Sözleşmeli MRR değerinin $T_0$ öncesinde mi yoksa sonrasında mı değiştiğine dair log bulunmamaktadır. |
| `additional_seats` | Unverified Temporal | `seats - included_seats` farkından türetildiği için koltuk adedi belirsizliğini taşır. |
| `subscription_id` | Identifier / Metadata | Kayıt takip amaçlı tutulmuştur, modele girdi olarak verilmemelidir. |
| `account_id` | Identifier / Metadata | Kayıt takip amaçlı tutulmuştur, modele girdi olarak verilmemelidir. |

---

## 8. Fatura ve Ödeme Verisi Kapsam Sınırlılığı ($T_0$ İtibarıyla)

EDA aşamasında tespit edilen fatura ve ödeme verisi kapsamındaki kısıt, $T_0$ tarihi itibarıyla incelenmiş ve doğrulanmıştır:

- **31 Aralık 2024'e kadar kesilen fatura sayısı**: 917 fatura
- **Bu faturalarla ilişkili tekil abonelik sayısı**: 90 abonelik
- **31 Aralık 2024'e kadar yapılan ödeme denemesi sayısı**: 946 ödeme denemesi
- **Ödeme denemesi bulunan tekil abonelik sayısı**: 88 abonelik
- **1.242 kişilik modelleme kohortundaki fatura geçmişi**: Yalnızca **78 abonelik** (%6.28)
- **1.242 kişilik modelleme kohortundaki ödeme geçmişi**: Yalnızca **76 abonelik** (%6.12)
- **365 günlük churn hedefinde pozitif olan 182 aboneliğin fatura geçmişi**: Yalnızca **11 abonelik** (%6.04)

> [!CAUTION]
> **Stratejik Karar – Fatura Özelliklerinin Baseline Dışında Bırakılması**:
> Fatura ve ödeme geçmişi popülasyonun %94'ünde mevcut değildir. Bu değişkenlerin baseline modelleme veri setine dahil edilmesi durumunda, model genel müşteri davranışı yerine yalnızca ilk 120 aboneye özel yapay örüntüleri öğrenecektir (Selection Bias). Bu nedenle fatura ve ödeme değişkenleri ana modelleme veri setine **alınmamıştır**.

---

## 9. Eksik Değer Politikası

- **`region` (321 eksik)**: Analitik tarafsızlığı korumak amacıyla eksik değerler mod/ortalama gibi yapay yöntemlerle doldurulmamış, `NULL` olarak bırakılmıştır. Birlikte `region_missing_flag` (1/0) bayrağı üretilmiştir.
- **`created_at` (Kohortta 0 eksik)**: Tüm kohort hesaplarında açılış tarihi eksiksizdir. Olası sentetik durumlara karşı `account_created_at_missing_flag` üretilmiştir.
- **Kategorik Değişkenler**: Kategorik alanlara (`region`, `segment`, `acquisition_channel`, `plan_name`) bu aşamada One-Hot Encoding veya Ordinal Encoding uygulanmamıştır. Ham kategorik değerler korunmuştur.

---

## 10. Modelleme Aşamasına Kalan Sınırlılıklar ve Sonraki Adımlar

1. **Sınıf Dengesizliği Mücadelesi**: Özellikle 90 günlük hedef (%4,51 pozitif) için maliyete duyarlı öğrenme (Cost-sensitive Learning) veya threshold tuning gerekecektir.
2. **Ayrı Fatura Alt-Modeli Fırsatı**: Faturası bulunan ~80 aboneliklik alt küme üzerinde, ödeme başarısızlıklarının etkisini ölçen ayrı bir mikro-deney tasarlanabilir.
3. **Metrik Önceliği**: Kurumsal değerlendirmede False Negative (churn edeceği öngörülemeyen müşteri) maliyeti False Positive maliyetinden çok daha yüksek olduğu için Recall ve PR-AUC hedeflenmelidir.
4. **Plan Fiyatlandırma Varsayımı**: Güncel fiyatlar baz alındığından, geçmişte fiyat artışı veya paket değişikliği yapılmışsa bu etkinin izlenemeyeceği göz önünde bulundurulmalıdır.