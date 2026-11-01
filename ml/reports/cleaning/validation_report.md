# Veri Bütünlüğü ve Doğrulama Raporu (Data Validation Report)

## 1. Doğrulama Özeti

- **Toplam Kontrol Sayısı**: 30
- **Başarılı Kontroller**: 30 ✅
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
| foreign_key | `fk_subscriptions_account_id_to_accounts_account_id` | ✅ Geçti | FK relationship subscriptions.account_id -> accounts.account_id has 0 orphans. |
| foreign_key | `fk_subscriptions_plan_id_to_plans_plan_id` | ✅ Geçti | FK relationship subscriptions.plan_id -> plans.plan_id has 0 orphans. |
| foreign_key | `fk_invoices_subscription_id_to_subscriptions_subscription_id` | ✅ Geçti | FK relationship invoices.subscription_id -> subscriptions.subscription_id has 0 orphans. |
| foreign_key | `fk_payments_invoice_id_to_invoices_invoice_id` | ✅ Geçti | FK relationship payments.invoice_id -> invoices.invoice_id has 0 orphans. |
| missing_values | `missing_values_audit` | ✅ Geçti | Missing value audit completed. Primary keys have 0 nulls. |
| categoricals | `categorical_domain_accounts_region` | ✅ Geçti | Categorical 'region' in 'accounts' contains only valid allowed domain values: True. |
| categoricals | `categorical_domain_accounts_segment` | ✅ Geçti | Categorical 'segment' in 'accounts' contains only valid allowed domain values: True. |
| categoricals | `categorical_domain_accounts_acquisition_channel` | ✅ Geçti | Categorical 'acquisition_channel' in 'accounts' contains only valid allowed domain values: True. |
| categoricals | `categorical_domain_subscriptions_status` | ✅ Geçti | Categorical 'status' in 'subscriptions' contains only valid allowed domain values: True. |
| categoricals | `categorical_domain_subscriptions_plan_id` | ✅ Geçti | Categorical 'plan_id' in 'subscriptions' contains only valid allowed domain values: True. |
| categoricals | `categorical_domain_invoices_invoice_status` | ✅ Geçti | Categorical 'invoice_status' in 'invoices' contains only valid allowed domain values: True. |
| categoricals | `categorical_domain_payments_payment_status` | ✅ Geçti | Categorical 'payment_status' in 'payments' contains only valid allowed domain values: True. |
| categoricals | `categorical_domain_plans_plan_id` | ✅ Geçti | Categorical 'plan_id' in 'plans' contains only valid allowed domain values: True. |
| date_consistency | `date_order_subscriptions_started_vs_ended` | ✅ Geçti | subscriptions.ended_at >= subscriptions.started_at check passed: True. |
| date_consistency | `date_order_account_created_vs_subscription_started` | ✅ Geçti | subscriptions.started_at >= accounts.created_at check passed: True. |
| date_consistency | `date_order_invoice_date_vs_due_date` | ✅ Geçti | invoices.due_date >= invoices.invoice_date check passed: True. |
| date_consistency | `date_order_invoice_date_vs_payment_attempted` | ✅ Geçti | payments.attempted_at >= invoices.invoice_date check passed: True. |
| numerical_validity | `numeric_range_plans_monthly_price` | ✅ Geçti | Numeric rule 'monthly_price >= 0' on 'plans' passed: True. |
| numerical_validity | `numeric_range_plans_included_seats` | ✅ Geçti | Numeric rule 'included_seats > 0' on 'plans' passed: True. |
| numerical_validity | `numeric_range_subscriptions_seats` | ✅ Geçti | Numeric rule 'seats > 0' on 'subscriptions' passed: True. |
| numerical_validity | `numeric_range_subscriptions_mrr` | ✅ Geçti | Numeric rule 'mrr >= 0' on 'subscriptions' passed: True. |
| numerical_validity | `numeric_range_invoices_amount_due` | ✅ Geçti | Numeric rule 'amount_due >= 0' on 'invoices' passed: True. |
| numerical_validity | `numeric_range_payments_amount` | ✅ Geçti | Numeric rule 'amount >= 0' on 'payments' passed: True. |
| coverage | `cross_table_coverage_and_cardinality` | ✅ Geçti | Cross-table coverage analyzed: 1:1 account-subscription mapping confirmed; 120 subscriptions covered by invoices. |

## 3. İlişkisel Bütünlük (Foreign Keys)

- `subscriptions.account_id` → `accounts.account_id`: %100 eşleşme (0 yetim kayıt)
- `subscriptions.plan_id` → `plans.plan_id`: %100 eşleşme (0 yetim kayıt)
- `invoices.subscription_id` → `subscriptions.subscription_id`: %100 eşleşme (0 yetim kayıt)
- `payments.invoice_id` → `invoices.invoice_id`: %100 eşleşme (0 yetim kayıt)

## 4. Tablolar Arası Kapsam ve Tane Seviyesi (Grain)

- **Hesap - Abonelik İlişkisi**: 2,000 hesap ve 2,000 abonelik arasında bire bir (1:1) ilişki mevcuttur.
- **Fatura Kapsamı**: Faturalar tablosundaki 2,000 fatura kaydı yalnızca ilk 120 abonelik (`S000001` - `S000120`) ile ilişkilidir (%6.0 kapsam).
- **Ödeme Kapsamı**: 2,000 fatura kaydının 1,815'i için en az 1 ödeme denemesi mevcuttur (%90.75 fatura ödeme kapsamı).

## 5. Lisans Bildirimi

Attribution: Analytics Engineering, "SaaS Subscriptions & Customer Churn Dataset v1.0.0". https://www.analyticsengineering.com/datasets/saas-subscriptions (CC BY 4.0)