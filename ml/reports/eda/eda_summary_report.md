# Keşifsel Veri Analizi (EDA) Kapsamlı Özet Raporu

## 1. Yönetici Özeti (Executive Summary)

- **Toplam Müşteri / Abonelik Sayısı**: 2,000
- **Aktif Abonelikler**: 1,590 (%79.50)
- **Müşteri Kaybı (Churn) Sayısı**: 410 (%20.50)
- **Veri Seti Kapsamı**: 2023-01-01 ile 2025-12-31 tarihleri arasındaki sentetik SaaS B2B verisi.
- **Lisans**: Creative Commons Attribution 4.0 International (CC BY 4.0)

## 2. Ana Analiz Bulguları

### 2.1. Plan Bazlı Churn Analizi
| Plan | Toplam Müşteri | Churn Sayısı | Churn Oranı (%) |
|---|---:|---:|---:|
| `Enterprise` | 104 | 26 | %25.00 |
| `Growth` | 669 | 138 | %20.63 |
| `Scale` | 272 | 52 | %19.12 |
| `Starter` | 955 | 194 | %20.31 |

> **Gözlem**: En yüksek churn oranı %25.00 ile `Enterprise` paketindedir. Kurumsal seviyedeki yüksek MRR kayıplarını engellemek için müşteri başarı (Customer Success) müdahaleleri önceliklendirilmelidir.

### 2.2. Segment Bazlı Churn Analizi
| Segment | Toplam Müşteri | Churn Sayısı | Churn Oranı (%) |
|---|---:|---:|---:|
| `SMB` | 1,407 | 303 | %21.54 |
| `enterprise` | 121 | 26 | %21.49 |
| `mid_market` | 472 | 81 | %17.16 |

> **Gözlem**: En düşük churn %17.16 ile `mid_market` segmentindedir. `SMB` segmenti hacim olarak veri setinin %70.35'ini oluşturmakta ve %21.54 churn oranına sahiptir.

### 2.3. Coğrafi Bölge ve Eksik Bölge Analizi (Ki-kare Bağımsızlık Testleri)
| Bölge | Toplam Müşteri | Churn Sayısı | Churn Oranı (%) |
|---|---:|---:|---:|
| `APAC` | 500 | 108 | %21.60 |
| `EMEA` | 520 | 99 | %19.04 |
| `LATAM` | 501 | 102 | %20.36 |
| `Missing` | 479 | 101 | %21.09 |

#### Eksik Bölge ile Gözlemlenebilir Değişkenler Arasındaki Ki-kare Bağımsızlık Testleri

Hesaplanan parametreler: $\chi^2$ test istatistiği, serbestlik derecesi (dof), p-değeri (`correction=False`), minimum beklenen hücre frekansı ve Ki-kare varsayım kontrolü (beklenen frekansı < 5 olan hücre sayısı):

| Değişken | $\chi^2$ İstatistiği | dof | p-değeri | Min. Beklenen Frekans | Hücre (< 5) | Varsayım Durumu |
|---|---:|---:|---:|---:|:---:|:---:|
| `churn` | 0.1325 | 1 | 0.7158 | 98.19 | 0/4 | Karşılandı ✅ |
| `segment` | 0.1604 | 2 | 0.9229 | 28.98 | 0/6 | Karşılandı ✅ |
| `acquisition_channel` | 1.8838 | 4 | 0.7571 | 92.21 | 0/10 | Karşılandı ✅ |
| `plan_name` | 1.3764 | 3 | 0.7111 | 24.91 | 0/8 | Karşılandı ✅ |

> **İstatistiksel Yorum ve Metodolojik Sınırlar (Eksik Veri Mekanizması)**:
> - **p-değeri Analizi**: $\alpha = 0.05$ anlamlılık düzeyinde incelendiğinde, tüm değişkenler için p-değerleri ($p > 0.70$) anlamlılık eşiğinin oldukça üzerindedir. Bu durum, eksik bölge bilgisi ile churn, müşteri segmenti, edinme kanalı veya plan türü arasında istatistiksel olarak anlamlı bir ilişki bulunamadığını gösterir.
> - **Varsayım Kontrolü**: Minimum beklenen hücre frekanslarının tamamı (en düşüğü 24.91) 5 eşiğinin çok üzerindedir ve beklenen frekansı 5'in altında olan hiçbir hücre yoktur. Cochran varsayımları tam olarak karşılanmaktadır.
> - **Önemli Metodolojik Uyarı**: Raporlanan p-değerleri keşifsel ve ham p-değerleridir; çoklu karşılaştırma (multiple testing) düzeltmesi uygulanmamıştır. En önemlisi, **gözlemlenebilir değişkenlerle anlamlı bir ilişki bulunamaması (bağımlılık olmaması), eksikliğin tamamen rastgele (MCAR - Missing Completely At Random) olduğunun kanıtı değildir**. Gözlemlenemeyen faktörlerin (unobserved confounders) eksiklik sürecini etkileyip etkilemediği bu veri setiyle test edilemez. Bu nedenle eksiklik mekanizması mevcut veriler ışığında **belirsiz / kanıtlanmamış** olarak değerlendirilmeli, eksik değerler yapay varsayımlarla doldurulmamalıdır.

### 2.4. Edinme Kanalı (Acquisition Channel) Bazlı Churn Analizi
| Kanal | Toplam Müşteri | Churn Sayısı | Churn Oranı (%) |
|---|---:|---:|---:|
| `organic` | 385 | 87 | %22.60 |
| `outbound` | 389 | 87 | %22.37 |
| `paid_search` | 404 | 86 | %21.29 |
| `partner` | 400 | 72 | %18.00 |
| `referral` | 422 | 78 | %18.48 |

> **Gözlem**: `partner` (%18.00) ve `referral` (%18.48) kanalları en yüksek sadakate sahiptir. `organic` (%22.60) ve `outbound` (%22.37) kanallarında müşteri kaybı belirgin biçimde daha yüksektir.

### 2.5. Gelir (MRR) ve Süre (Tenure) Analizi
- **MRR**: Churned müşterilerin ortalama sözleşmeli MRR değeri ($357.90), aktif müşterilerin ortalamasından ($332.08) bir miktar daha yüksektir. Bunun sebebi churn eden az sayıdaki Enterprise müşterisinin ortalamayı yukarı çekmesidir. Medyan MRR her iki grupta da $199.00 seviyesindedir.
- **Tenure**: Churn eden müşteriler ortalama 355 gün (medyan 344 gün) sonra ayrılmaktadır. Kayıplar özellikle 6. ve 12. aylar arasında yoğunlaşmaktadır.
- **Tenure Değişkeni ve Hedef Sızıntısı (Target Leakage) Uyarısı**:
  - Veri setindeki `tenure_days` değişkeni, iptal edilmiş aboneliklerde `ended_at - started_at`, aktif aboneliklerde ise gözlem kesim tarihi olan `2025-12-31` ile `started_at` arasındaki fark olarak hesaplanmıştır (farklı referans tarihleri kullanılmıştır).
  - Bu hesaplama tanımlayıcı keşifsel analiz (EDA) amacıyla uygundur.
  - Ancak **bir churn tahmin modelinde doğrudan girdi özniteliği olarak kullanılması durumunda Hedef Sızıntısına (Target Leakage) yol açar**; çünkü iptal tarihi (`ended_at`) gelecekte gerçekleşen hedef olayın bilgisini içerir.
  - Gelecekte makine öğrenmesi modeli eğitilmeden önce tahminin yapılacağı zaman noktası ($T_0$ / observation window) sabitlenmeli ve tenure yalnızca $T_0$ anına kadar geçen süre olarak türetilmelidir.

## 3. Veri Seti Sınırlılıkları ve Kritik Fatura Analizi

### 3.1. Fatura ve Ödeme Verisi Kapsam Sınırlılığı (Invoice Coverage Disparity)
- `invoices.csv` dosyasında yer alan 2,000 fatura kaydı yalnızca ilk 120 abonelik (`S000001` - `S000120`) ile ilişkilidir.
- Geriye kalan 1,880 abonelik (%94.0) için herhangi bir fatura veya ödeme geçmişi bulunmamaktadır.
- **120 Aboneliklik Fatura Kohortu Analizi**:
  - Faturası bulunan abonelik sayısı: 120
  - Başarısız ödeme denemesi yaşayan abonelik: 86
  - Başarısız ödeme yaşayan grupta churn oranı: %18.60
  - Başarısız ödeme yaşamayan grupta churn oranı: %23.53
- **Model Eğitimi Açısından Kritik Not**: Fatura ve ödeme özellikleri tüm popülasyonu temsil etmemektedir. İleride makine öğrenmesi modeli eğitilirken, ödeme özelliklerinin sadece %6'lık bir alt grupta var olduğu dikkate alınmalı; genel modelde bu değişkenler doğrudan kullanılmamalı veya eksiklik durumu özel olarak kodlanmalıdır.

## 4. Oluşturulan Grafikler Listesi

Tüm grafikler yüksek çözünürlükte (300 DPI) `ml/reports/eda/figures/` dizinine kaydedilmiştir:
1. `01_churn_distribution.png`: Churn ve aktif hesap hacim ve oran dağılımı (Bar & Donut).
2. `02_churn_by_plan.png`: Plan bazında abone sayısı ve churn oranı grafiği.
3. `03_churn_by_segment.png`: Müşteri segmenti bazında hacim ve churn oranı grafiği.
4. `04_churn_by_region.png`: Coğrafi bölgelere ve eksik bölgeye göre churn dağılımı.
5. `05_churn_by_acquisition_channel.png`: Müşteri edinme kanalı bazlı churn oranı analizi.
6. `06_mrr_distribution_by_churn.png`: Churn durumuna göre MRR dağılımı (Boxplot ve KDE Yoğunluk).
7. `07_tenure_distribution_by_churn.png`: Churn süresi ve müşteri kalma süresi analizi (Tenure Days).
8. `08_missing_values_matrix.png`: Tablo sütunları bazında eksik değer oranları grafiği.
9. `09_numeric_distributions.png`: Koltuk sayısı, MRR, liste fiyatı ve ek koltuk dağılımları.
10. `10_monthly_churn_trend.png`: 2023-2025 aylık iptal hacmi (cancellation count / volume) trendi.
11. `11_payment_failures_vs_churn_cohort.png`: 120 abonelik fatura kohortunda ödeme başarısızlıkları ve churn ilişkisi.
12. `12_invoice_coverage_breakdown.png`: Fatura kapsamındaki %6'lık kısıt ve kohort karşılaştırması.

> [!NOTE]
> **Aylık İptal Grafiği Notu**: `10_monthly_churn_trend.png` grafiği aylık churn oranını değil, iptal edilen abonelik sayısını (hacmini) göstermektedir. Her ay risk altındaki aktif müşteri tabanı sabit ve net tanımlanmadığından bir oran türetilmemiştir. Mevcut sentetik veride mevsimsellik veya nedensellik hakkında yeterli kanıt bulunmadığından kesin çıkarımlardan kaçınılmalıdır.

## 5. Lisans ve Kaynak Gösterimi

Attribution: Analytics Engineering, "SaaS Subscriptions & Customer Churn Dataset v1.0.0". https://www.analyticsengineering.com/datasets/saas-subscriptions (CC BY 4.0)