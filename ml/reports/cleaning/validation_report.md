# Veri Bütünlüğü ve Doğrulama Raporu (Data Validation Report)

## 1. Doğrulama Özeti

- **Toplam Kontrol Sayısı**: 39
- **Başarılı Kontroller**: 39 ✅
- **Başarısız Kontroller**: 0 ❌
- **Genel Bütünlük Durumu**: BAŞARILI

## 2. Detaylı Kontrol Sonuçları

| Kategori | Kontrol Adı | Durum | Açıklama |
|---|---|:---:|---|
| row_counts | `row_counts_pre_post` | ✅ Geçti | Pre and post row counts verified (1 duplicate removed per table). |
| primary_key | `pk_uniqueness_plans` | ✅ Geçti | Primary key 'plan_id' in 'plans' is unique and non-null: True. |
| primary_key | `pk_uniqueness_accounts` | ✅ Geçti | Primary key 'account_id' in 'accounts' is unique and non-null: True. |
| primary_key | `pk_uniqueness_subscriptions` | ✅ Geçti | Primary key 'subscription_id' in 'subscriptions' is unique and non-null: True. |
| primary_key | `pk_uniqueness_invoices` | ✅ Geçti | Primary key 'invoice_id' in 'invoices' is unique and non-null: True. |
| primary_key | `pk_uniqueness_payments` | ✅ Geçti | Primary key 'payment_id' in 'payments' is unique and non-null: True. |
| foreign_key | `fk_subscriptions_account_id_to_accounts_account_id` | ✅ Geçti | FK relationship subscriptions.account_id -> accounts.account_id passed (0 nulls, 0 orphans). |
| foreign_key | `fk_subscriptions_plan_id_to_plans_plan_id` | ✅ Geçti | FK relationship subscriptions.plan_id -> plans.plan_id passed (0 nulls, 0 orphans). |
| foreign_key | `fk_invoices_subscription_id_to_subscriptions_subscription_id` | ✅ Geçti | FK relationship invoices.subscription_id -> subscriptions.subscription_id passed (0 nulls, 0 orphans). |
| foreign_key | `fk_payments_invoice_id_to_invoices_invoice_id` | ✅ Geçti | FK relationship payments.invoice_id -> invoices.invoice_id passed (0 nulls, 0 orphans). |
| missing_values | `missing_values_audit` | ✅ Geçti | Missing value audit completed. Primary keys have 0 nulls. |
| categoricals | `categorical_domain_accounts_region` | ✅ Geçti | Categorical 'region' in 'accounts' passed (valid domain, nullability rule satisfied: nullable=True, nulls=479). |
| categoricals | `categorical_domain_accounts_segment` | ✅ Geçti | Categorical 'segment' in 'accounts' passed (valid domain, nullability rule satisfied: nullable=False, nulls=0). |
| categoricals | `categorical_domain_accounts_acquisition_channel` | ✅ Geçti | Categorical 'acquisition_channel' in 'accounts' passed (valid domain, nullability rule satisfied: nullable=False, nulls=0). |
| categoricals | `categorical_domain_subscriptions_status` | ✅ Geçti | Categorical 'status' in 'subscriptions' passed (valid domain, nullability rule satisfied: nullable=False, nulls=0). |
| categoricals | `categorical_domain_subscriptions_plan_id` | ✅ Geçti | Categorical 'plan_id' in 'subscriptions' passed (valid domain, nullability rule satisfied: nullable=False, nulls=0). |
| categoricals | `categorical_domain_invoices_invoice_status` | ✅ Geçti | Categorical 'invoice_status' in 'invoices' passed (valid domain, nullability rule satisfied: nullable=False, nulls=0). |
| categoricals | `categorical_domain_payments_payment_status` | ✅ Geçti | Categorical 'payment_status' in 'payments' passed (valid domain, nullability rule satisfied: nullable=False, nulls=0). |
| categoricals | `categorical_domain_plans_plan_id` | ✅ Geçti | Categorical 'plan_id' in 'plans' passed (valid domain, nullability rule satisfied: nullable=False, nulls=0). |
| categoricals | `categorical_domain_plans_plan_name` | ✅ Geçti | Categorical 'plan_name' in 'plans' passed (valid domain, nullability rule satisfied: nullable=False, nulls=0). |
| date_conversion | `date_conversion_accounts_created_at` | ✅ Geçti | Date conversion check on 'accounts.created_at' PASSED: 1999 converted, 1 originally missing preserved, 0 errors. |
| date_conversion | `date_conversion_subscriptions_started_at` | ✅ Geçti | Date conversion check on 'subscriptions.started_at' PASSED: 1999 converted, 1 originally missing preserved, 0 errors. |
| date_conversion | `date_conversion_subscriptions_ended_at` | ✅ Geçti | Date conversion check on 'subscriptions.ended_at' PASSED: 410 converted, 1590 originally missing preserved, 0 errors. |
| date_conversion | `date_conversion_invoices_invoice_date` | ✅ Geçti | Date conversion check on 'invoices.invoice_date' PASSED: 1999 converted, 1 originally missing preserved, 0 errors. |
| date_conversion | `date_conversion_invoices_due_date` | ✅ Geçti | Date conversion check on 'invoices.due_date' PASSED: 2000 converted, 0 originally missing preserved, 0 errors. |
| date_conversion | `date_conversion_payments_attempted_at` | ✅ Geçti | Date conversion check on 'payments.attempted_at' PASSED: 1999 converted, 1 originally missing preserved, 0 errors. |
| date_consistency | `date_order_subscriptions_started_vs_ended` | ✅ Geçti | subscriptions.ended_at >= subscriptions.started_at check passed. |
| date_consistency | `date_order_account_created_vs_subscription_started` | ✅ Geçti | subscriptions.started_at >= accounts.created_at check passed. |
| date_consistency | `date_order_invoice_date_vs_due_date` | ✅ Geçti | invoices.due_date >= invoices.invoice_date check passed. |
| date_consistency | `date_order_invoice_date_vs_payment_attempted` | ✅ Geçti | payments.attempted_at >= invoices.invoice_date check passed. |
| numerical_validity | `numeric_range_plans_monthly_price` | ✅ Geçti | Numeric rule 'monthly_price >= 0' on 'plans' passed: True. |
| numerical_validity | `numeric_range_plans_included_seats` | ✅ Geçti | Numeric rule 'included_seats > 0' on 'plans' passed: True. |
| numerical_validity | `numeric_range_subscriptions_seats` | ✅ Geçti | Numeric rule 'seats > 0' on 'subscriptions' passed: True. |
| numerical_validity | `numeric_range_subscriptions_mrr` | ✅ Geçti | Numeric rule 'mrr >= 0' on 'subscriptions' passed: True. |
| numerical_validity | `numeric_range_invoices_amount_due` | ✅ Geçti | Numeric rule 'amount_due >= 0' on 'invoices' passed: True. |
| numerical_validity | `numeric_range_payments_amount` | ✅ Geçti | Numeric rule 'amount >= 0' on 'payments' passed: True. |
| coverage | `coverage_account_subscription_1to1` | ✅ Geçti | Account-Subscription 1:1 relationship confirmed (2,000 accounts = 2,000 subscriptions, 0 mismatches). |
| coverage | `coverage_invoice_cohort_match` | ✅ Geçti | Invoice cohort matches expected 120 subscriptions (S000001 - S000120, 6.0% coverage). |
| coverage | `coverage_payments_and_invoices` | ✅ Geçti | Payments coverage verified: 118 subscriptions with payments across 1815 invoices, 0 orphan invoice references. |

## 3. İlişkisel Bütünlük (Foreign Keys)

- `subscriptions.account_id` → `accounts.account_id`: 0 NULL, 0 yetim kayıt (%100 geçerli)
- `subscriptions.plan_id` → `plans.plan_id`: 0 NULL, 0 yetim kayıt (%100 geçerli)
- `invoices.subscription_id` → `subscriptions.subscription_id`: 0 NULL, 0 yetim kayıt (%100 geçerli)
- `payments.invoice_id` → `invoices.invoice_id`: 0 NULL, 0 yetim kayıt (%100 geçerli)

## 4. Tarih Dönüşüm ve Gerçek Takvim Doğrulaması

Tarihler hem ISO biçimi hem de takvim geçerliliği (`datetime.strptime`) açısından denetlenmiş,
ham ve temizlenmiş tarihlerin aynı takvim gününü temsil ettiği doğrulanmıştır:
- `accounts.created_at`: 1 ham eksik korundu, 0 geçersiz takvim tarihi, 0 uyuşmazlık, 1,999 başarıyla dönüştürüldü.
- `subscriptions.started_at`: 1 ham eksik korundu, 0 geçersiz takvim tarihi, 0 uyuşmazlık, 1,999 başarıyla dönüştürüldü.
- `subscriptions.ended_at`: 1,590 aktif abonelik (doğal eksik) korundu, 0 geçersiz takvim tarihi, 0 uyuşmazlık, 410 başarıyla dönüştürüldü.
- `invoices.invoice_date`: 1 ham eksik korundu, 0 geçersiz takvim tarihi, 0 uyuşmazlık, 1,999 başarıyla dönüştürüldü.
- `invoices.due_date`: 0 ham eksik, 0 geçersiz takvim tarihi, 0 uyuşmazlık, 2,000 başarıyla dönüştürüldü.
- `payments.attempted_at`: 1 ham eksik korundu, 0 geçersiz takvim tarihi, 0 uyuşmazlık, 1,999 başarıyla dönüştürüldü.

## 5. Kategorik Alanlar ve Nullability Politikası

- `accounts.region`: Nullable (479 eksik bölge korundu, geçerli).
- `accounts.segment`: Zorunlu (0 NULL, geçerli).
- `accounts.acquisition_channel`: Zorunlu (0 NULL, geçerli).
- `subscriptions.status`: Zorunlu (0 NULL, geçerli).
- `subscriptions.plan_id`: Zorunlu (0 NULL, geçerli).
- `invoices.invoice_status`: Zorunlu (0 NULL, geçerli).
- `payments.payment_status`: Zorunlu (0 NULL, geçerli).
- `plans.plan_id`: Zorunlu PK (0 NULL, geçerli).
- `plans.plan_name`: Zorunlu (0 NULL, geçerli).

## 6. Tablolar Arası Kapsam ve Bire Bir (1:1) İlişki Doğrulaması

- **Hesap - Abonelik İlişkisi**: 2,000 hesap ve 2,000 abonelik arasında tam bire bir ilişki mevcuttur.
  - Eksik hesap referansı: 0
  - Beklenmeyen hesap referansı: 0
  - Tekrarlı hesap referansı: 0
  - Toplam uyumsuzluk (mismatch_count): 0
- **Fatura Kohort Kapsamı**: 2,000 fatura kaydı tam olarak ilk 120 abonelik (`S000001` - `S000120`) ile ilişkilidir (0 uyumsuzluk).
- **Ödeme Kapsamı**: 2,000 faturanın 1,815'i için ödeme denemesi mevcuttur; bu ödemeler 118 farklı abonelikle ilişkilidir (0 yetim fatura referansı).

## 7. Lisans Bildirimi

Attribution: Analytics Engineering, "SaaS Subscriptions & Customer Churn Dataset v1.0.0". https://www.analyticsengineering.com/datasets/saas-subscriptions (CC BY 4.0)