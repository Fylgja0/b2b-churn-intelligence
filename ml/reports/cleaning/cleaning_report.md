# Veri Ön İşleme ve Temizlik Raporu (Data Cleaning Report)

## 1. Genel Özet

Bu rapor, SaaS müşteri kaybı veri altyapısındaki 5 ham tablonun ön işleme sonuçlarını belgeler.

| Tablo | Ham Satır | Temiz Satır | Silinen Tekrar | PK Benzersiz | Boşluk Düzeltme | Kasa Standardizasyonu | Non-ISO Tarih |
|---|---:|---:|---:|:---:|---:|---:|---:|
| `plans` | 5 | 4 | 1 | ✅ Evet | 0 | 0 | 0 |
| `accounts` | 2,001 | 2,000 | 1 | ✅ Evet | 1 | 0 | 1 |
| `subscriptions` | 2,001 | 2,000 | 1 | ✅ Evet | 1 | 1 | 1 |
| `invoices` | 2,001 | 2,000 | 1 | ✅ Evet | 1 | 1 | 1 |
| `payments` | 2,001 | 2,000 | 1 | ✅ Evet | 1 | 1 | 1 |

## 2. Eksik Değer Politikası ve Bulgular

- **`accounts.region`**: 479 satırda coğrafi bölge eksiktir (%23.95). Bu eksik değerler gelişigüzel doldurulmamış, analitik doğruluğu korumak adına eksik (`NaN`) olarak muhafaza edilmiştir.
- **`subscriptions.ended_at`**: Ham veride 1,591 satırda, mükerrer satır temizliği sonrasında 1,590 temiz satırda `ended_at` boştur. Bu durum veri hatası değil, aktif (`active`) aboneliklerin henüz sona ermediğini gösteren geçerli iş mantığı durumudur. 410 iptal edilmiş (`cancelled`) aboneliğin tamamında `ended_at` mevcuttur.
- **Eksik Tarihler**: `accounts` (1 adet `created_at`), `subscriptions` (1 adet `started_at`), `invoices` (1 adet `invoice_date`), `payments` (1 adet `attempted_at`) tablolarında 1'er adet eksik tarih tespit edilmiştir. Bu tarihler teyit edilmemiş varsayımlarla doldurulmamıştır.

### Tablo Bazında Eksik Değer Dağılımı (Temizlik Sonrası)

#### `plans` Tablosu
| Sütun | Eksik Değer Sayısı | Eksik Değer Oranı (%) |
|---|---:|---:|
| `plan_id` | 0 | 0.00% |
| `plan_name` | 0 | 0.00% |
| `monthly_price` | 0 | 0.00% |
| `included_seats` | 0 | 0.00% |

#### `accounts` Tablosu
| Sütun | Eksik Değer Sayısı | Eksik Değer Oranı (%) |
|---|---:|---:|
| `account_id` | 0 | 0.00% |
| `created_at` | 1 | 0.05% |
| `region` | 479 | 23.95% |
| `segment` | 0 | 0.00% |
| `acquisition_channel` | 0 | 0.00% |

#### `subscriptions` Tablosu
| Sütun | Eksik Değer Sayısı | Eksik Değer Oranı (%) |
|---|---:|---:|
| `subscription_id` | 0 | 0.00% |
| `account_id` | 0 | 0.00% |
| `plan_id` | 0 | 0.00% |
| `started_at` | 1 | 0.05% |
| `ended_at` | 1,590 | 79.50% |
| `status` | 0 | 0.00% |
| `seats` | 0 | 0.00% |
| `mrr` | 0 | 0.00% |

#### `invoices` Tablosu
| Sütun | Eksik Değer Sayısı | Eksik Değer Oranı (%) |
|---|---:|---:|
| `invoice_id` | 0 | 0.00% |
| `subscription_id` | 0 | 0.00% |
| `invoice_date` | 1 | 0.05% |
| `due_date` | 0 | 0.00% |
| `amount_due` | 0 | 0.00% |
| `invoice_status` | 0 | 0.00% |

#### `payments` Tablosu
| Sütun | Eksik Değer Sayısı | Eksik Değer Oranı (%) |
|---|---:|---:|
| `payment_id` | 0 | 0.00% |
| `invoice_id` | 0 | 0.00% |
| `attempted_at` | 1 | 0.05% |
| `payment_status` | 0 | 0.00% |
| `amount` | 0 | 0.00% |

## 3. Uygulanan Temizlik Kuralları

1. **Tekrarlı Satır Temizliği**: Her tabloda yer alan tam tekrarlı kayıtlar (1'er adet) tespit edilmiş ve `keep='first'` mantığıyla temizlenmiştir.
2. **Primary Key Doğrulaması**: Tekrarlar kaldırıldıktan sonra tüm birincil anahtarların (`account_id`, `plan_id`, `subscription_id`, `invoice_id`, `payment_id`) %100 benzersiz olduğu doğrulanmıştır.
3. **Metin ve Boşluk Standardizasyonu**: Hücre başı ve sonundaki gereksiz boşluklar (`strip`) temizlenmiştir (örn. `' APAC '` -> `'APAC'`, `' ACTIVE '` -> `'active'`).
4. **Kategorik Kasa Standardizasyonu**: Durum alanları küçük harfe (`active`, `cancelled`, `paid`, `open`, `void`, `succeeded`, `failed`), bölge ve plan kodları büyük harfe (`APAC`, `EMEA`, `LATAM`, `NA`, `PLN_*`) standartlaştırılmıştır.
5. **Tarih Standardizasyonu**: Slash ile ayrılmış formatlar (`YYYY/MM/DD`) ISO-8601 (`YYYY-MM-DD`) formatına dönüştürülmüştür. Hiçbir geçersiz tarih tespit edilmemiştir.

## 4. Orijinal Veri Kaynağı ve Lisans Bildirimi

Bu veri seti **Creative Commons Attribution 4.0 International (CC BY 4.0)** lisansı ile lisanslanmıştır.
Önerilen kaynak gösterimi:
> Analytics Engineering, "SaaS Subscriptions & Customer Churn Dataset v1.0.0". https://www.analyticsengineering.com/datasets/saas-subscriptions