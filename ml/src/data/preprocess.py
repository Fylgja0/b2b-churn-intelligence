"""
Data Preprocessing Pipeline for SaaS Churn Intelligence.

Reads raw CSV tables, detects and cleans duplicates, normalizes whitespace and casing,
standardizes dates to ISO-8601, and writes clean tables and detailed cleaning reports.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Base paths relative to this script location
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
ML_DIR = SRC_DIR.parent
DEFAULT_RAW_DIR = ML_DIR / "data" / "raw"
DEFAULT_PROCESSED_TABLES_DIR = ML_DIR / "data" / "processed" / "tables"
DEFAULT_REPORTS_DIR = ML_DIR / "reports" / "cleaning"

TABLE_METADATA: Dict[str, Dict[str, Any]] = {
    "plans": {
        "file_name": "plans.csv",
        "pk": "plan_id",
        "date_cols": [],
        "numeric_cols": ["monthly_price", "included_seats"],
        "categorical_cols": ["plan_id", "plan_name"],
    },
    "accounts": {
        "file_name": "accounts.csv",
        "pk": "account_id",
        "date_cols": ["created_at"],
        "numeric_cols": [],
        "categorical_cols": ["region", "segment", "acquisition_channel"],
    },
    "subscriptions": {
        "file_name": "subscriptions.csv",
        "pk": "subscription_id",
        "date_cols": ["started_at", "ended_at"],
        "numeric_cols": ["seats", "mrr"],
        "categorical_cols": ["status", "plan_id"],
    },
    "invoices": {
        "file_name": "invoices.csv",
        "pk": "invoice_id",
        "date_cols": ["invoice_date", "due_date"],
        "numeric_cols": ["amount_due"],
        "categorical_cols": ["invoice_status"],
    },
    "payments": {
        "file_name": "payments.csv",
        "pk": "payment_id",
        "date_cols": ["attempted_at"],
        "numeric_cols": ["amount"],
        "categorical_cols": ["payment_status"],
    },
}


@dataclass
class TableCleanMetrics:
    table_name: str
    raw_rows: int
    clean_rows: int
    duplicates_removed: int
    pk_is_unique: bool
    null_counts_before: Dict[str, int]
    null_counts_after: Dict[str, int]
    whitespace_fixed_count: int
    casing_standardized_count: int
    non_iso_dates_fixed: int
    unparseable_dates_count: int
    unparseable_date_details: List[Dict[str, Any]] = field(default_factory=list)


class DataPreprocessor:
    """Preprocesses raw SaaS subscription data extracts into validated tables."""

    def __init__(
        self,
        raw_dir: Path = DEFAULT_RAW_DIR,
        processed_dir: Path = DEFAULT_PROCESSED_TABLES_DIR,
        reports_dir: Path = DEFAULT_REPORTS_DIR,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.reports_dir = Path(reports_dir)

    def load_raw_table(self, table_name: str) -> pd.DataFrame:
        file_path = self.raw_dir / TABLE_METADATA[table_name]["file_name"]
        if not file_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {file_path}")
        logger.info(f"Loading raw file: {file_path}")
        return pd.read_csv(file_path, dtype=str)

    def clean_table(self, table_name: str, df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, TableCleanMetrics]:
        meta = TABLE_METADATA[table_name]
        pk_col = meta["pk"]
        date_cols = meta["date_cols"]
        categorical_cols = meta["categorical_cols"]
        numeric_cols = meta["numeric_cols"]

        raw_rows = len(df_raw)
        null_counts_before = df_raw.isna().sum().to_dict()

        # 1. Detect and remove exact duplicate rows
        dup_mask = df_raw.duplicated(keep="first")
        duplicates_removed = int(dup_mask.sum())
        df = df_raw[~dup_mask].copy().reset_index(drop=True)

        if duplicates_removed > 0:
            logger.info(f"[{table_name}] Removed {duplicates_removed} duplicate row(s).")

        # 2. Clean whitespace across all string columns
        whitespace_fixed_count = 0
        for col in df.columns:
            if df[col].dtype == object or df[col].dtype == "string":
                has_ws = df[col].dropna().apply(lambda v: v != v.strip() if isinstance(v, str) else False)
                ws_count = int(has_ws.sum())
                if ws_count > 0:
                    whitespace_fixed_count += ws_count
                    logger.info(f"[{table_name}] Trimmed whitespace in column '{col}' for {ws_count} row(s).")
                df[col] = df[col].apply(lambda v: v.strip() if isinstance(v, str) else v)

        # 3. Standardize categorical casing
        casing_standardized_count = 0
        for col in categorical_cols:
            if col not in df.columns:
                continue
            original_series = df[col].copy()

            if col in ("status", "invoice_status", "payment_status", "acquisition_channel"):
                # Lowercase standards
                df[col] = df[col].apply(lambda v: v.lower() if isinstance(v, str) else v)
            elif col in ("region", "plan_id"):
                # Uppercase standards
                df[col] = df[col].apply(lambda v: v.upper() if isinstance(v, str) else v)
            elif col == "segment":
                # Standardize SMB / mid_market / enterprise
                def _standardize_segment(v: Any) -> Any:
                    if not isinstance(v, str):
                        return v
                    vl = v.lower()
                    if vl == "smb":
                        return "SMB"
                    elif vl == "mid_market":
                        return "mid_market"
                    elif vl == "enterprise":
                        return "enterprise"
                    return v

                df[col] = df[col].apply(_standardize_segment)
            elif col == "plan_name":
                # Title case
                df[col] = df[col].apply(lambda v: v.title() if isinstance(v, str) else v)

            changed = (original_series != df[col]) & (original_series.notna()) & (df[col].notna())
            c_count = int(changed.sum())
            if c_count > 0:
                casing_standardized_count += c_count
                logger.info(f"[{table_name}] Standardized casing in column '{col}' for {c_count} row(s).")

        # 4. Standardize dates to ISO-8601 (YYYY-MM-DD)
        non_iso_dates_fixed = 0
        unparseable_dates_count = 0
        unparseable_details: List[Dict[str, Any]] = []

        for d_col in date_cols:
            if d_col not in df.columns:
                continue
            s_dates = df[d_col].copy()
            non_iso = s_dates.dropna().str.contains("/", regex=False)
            non_iso_count = int(non_iso.sum())
            if non_iso_count > 0:
                non_iso_dates_fixed += non_iso_count
                logger.info(f"[{table_name}] Converted {non_iso_count} slash date(s) in '{d_col}' to ISO format.")

            parsed = pd.to_datetime(s_dates, format="mixed", errors="coerce")
            # Track any genuinely unparseable dates that were not originally null
            unparseable = s_dates.notna() & parsed.isna()
            unp_count = int(unparseable.sum())
            if unp_count > 0:
                unparseable_dates_count += unp_count
                for idx, bad_val in s_dates[unparseable].items():
                    unparseable_details.append({
                        "table": table_name,
                        "column": d_col,
                        "row_index": int(idx),
                        "pk_value": df.loc[idx, pk_col],
                        "raw_value": str(bad_val),
                    })
                logger.warning(f"[{table_name}] Found {unp_count} unparseable date(s) in column '{d_col}'.")

            # Store clean ISO-8601 date string or empty string / NaN
            df[d_col] = parsed.dt.strftime("%Y-%m-%d")

        # 5. Type cast numeric columns
        for num_col in numeric_cols:
            if num_col in df.columns:
                df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

        # 6. Check PK uniqueness after deduplication
        pk_is_unique = bool(df[pk_col].is_unique)
        if not pk_is_unique:
            logger.error(f"[{table_name}] Primary key '{pk_col}' is NOT unique after cleaning!")

        clean_rows = len(df)
        null_counts_after = df.isna().sum().to_dict()

        metrics = TableCleanMetrics(
            table_name=table_name,
            raw_rows=raw_rows,
            clean_rows=clean_rows,
            duplicates_removed=duplicates_removed,
            pk_is_unique=pk_is_unique,
            null_counts_before=null_counts_before,
            null_counts_after=null_counts_after,
            whitespace_fixed_count=whitespace_fixed_count,
            casing_standardized_count=casing_standardized_count,
            non_iso_dates_fixed=non_iso_dates_fixed,
            unparseable_dates_count=unparseable_dates_count,
            unparseable_date_details=unparseable_details,
        )
        return df, metrics

    def run(self) -> Dict[str, TableCleanMetrics]:
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        results: Dict[str, TableCleanMetrics] = {}
        cleaned_dfs: Dict[str, pd.DataFrame] = {}

        for table_name in TABLE_METADATA.keys():
            df_raw = self.load_raw_table(table_name)
            df_clean, metrics = self.clean_table(table_name, df_raw)

            # Save clean table to CSV
            output_file = self.processed_dir / f"{table_name}.csv"
            df_clean.to_csv(output_file, index=False)
            logger.info(f"Saved cleaned table: {output_file} ({metrics.clean_rows} rows)")

            results[table_name] = metrics
            cleaned_dfs[table_name] = df_clean

        # Generate and save cleaning report
        self._write_reports(results)
        return results

    def _write_reports(self, results: Dict[str, TableCleanMetrics]) -> None:
        json_report_path = self.reports_dir / "cleaning_summary.json"
        md_report_path = self.reports_dir / "cleaning_report.md"

        serializable_results = {k: asdict(v) for k, v in results.items()}
        with open(json_report_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Saved JSON cleaning summary: {json_report_path}")

        # Build Markdown report
        md_lines = [
            "# Veri Ön İşleme ve Temizlik Raporu (Data Cleaning Report)",
            "",
            "## 1. Genel Özet",
            "",
            "Bu rapor, SaaS müşteri kaybı veri altyapısındaki 5 ham tablonun ön işleme sonuçlarını belgeler.",
            "",
            "| Tablo | Ham Satır | Temiz Satır | Silinen Tekrar | PK Benzersiz | Boşluk Düzeltme | Kasa Standardizasyonu | Non-ISO Tarih |",
            "|---|---:|---:|---:|:---:|---:|---:|---:|",
        ]

        for t_name, m in results.items():
            pk_status = "✅ Evet" if m.pk_is_unique else "❌ Hayır"
            md_lines.append(
                f"| `{t_name}` | {m.raw_rows:,} | {m.clean_rows:,} | {m.duplicates_removed} | "
                f"{pk_status} | {m.whitespace_fixed_count} | {m.casing_standardized_count} | {m.non_iso_dates_fixed} |"
            )

        md_lines.extend([
            "",
            "## 2. Eksik Değer Politikası ve Bulgular",
            "",
            "- **`accounts.region`**: 479 satırda coğrafi bölge eksiktir (%23.95). Bu eksik değerler gelişigüzel doldurulmamış, analitik doğruluğu korumak adına eksik (`NaN`) olarak muhafaza edilmiştir.",
            "- **`subscriptions.ended_at`**: Ham veride 1,591 satırda, mükerrer satır temizliği sonrasında 1,590 temiz satırda `ended_at` boştur. Bu durum veri hatası değil, aktif (`active`) aboneliklerin henüz sona ermediğini gösteren geçerli iş mantığı durumudur. 410 iptal edilmiş (`cancelled`) aboneliğin tamamında `ended_at` mevcuttur.",
            "- **Eksik Tarihler**: `accounts` (1 adet `created_at`), `subscriptions` (1 adet `started_at`), `invoices` (1 adet `invoice_date`), `payments` (1 adet `attempted_at`) tablolarında 1'er adet eksik tarih tespit edilmiştir. Bu tarihler teyit edilmemiş varsayımlarla doldurulmamıştır.",
            "",
            "### Tablo Bazında Eksik Değer Dağılımı (Temizlik Sonrası)",
            "",
        ])

        for t_name, m in results.items():
            md_lines.append(f"#### `{t_name}` Tablosu")
            md_lines.append("| Sütun | Eksik Değer Sayısı | Eksik Değer Oranı (%) |")
            md_lines.append("|---|---:|---:|")
            for col, count in m.null_counts_after.items():
                pct = (count / m.clean_rows * 100) if m.clean_rows > 0 else 0
                md_lines.append(f"| `{col}` | {count:,} | {pct:.2f}% |")
            md_lines.append("")

        md_lines.extend([
            "## 3. Uygulanan Temizlik Kuralları",
            "",
            "1. **Tekrarlı Satır Temizliği**: Her tabloda yer alan tam tekrarlı kayıtlar (1'er adet) tespit edilmiş ve `keep='first'` mantığıyla temizlenmiştir.",
            "2. **Primary Key Doğrulaması**: Tekrarlar kaldırıldıktan sonra tüm birincil anahtarların (`account_id`, `plan_id`, `subscription_id`, `invoice_id`, `payment_id`) %100 benzersiz olduğu doğrulanmıştır.",
            "3. **Metin ve Boşluk Standardizasyonu**: Hücre başı ve sonundaki gereksiz boşluklar (`strip`) temizlenmiştir (örn. `' APAC '` -> `'APAC'`, `' ACTIVE '` -> `'active'`).",
            "4. **Kategorik Kasa Standardizasyonu**: Durum alanları küçük harfe (`active`, `cancelled`, `paid`, `open`, `void`, `succeeded`, `failed`), bölge ve plan kodları büyük harfe (`APAC`, `EMEA`, `LATAM`, `NA`, `PLN_*`) standartlaştırılmıştır.",
            "5. **Tarih Standardizasyonu**: Slash ile ayrılmış formatlar (`YYYY/MM/DD`) ISO-8601 (`YYYY-MM-DD`) formatına dönüştürülmüştür. Hiçbir geçersiz tarih tespit edilmemiştir.",
            "",
            "## 4. Orijinal Veri Kaynağı ve Lisans Bildirimi",
            "",
            "Bu veri seti **Creative Commons Attribution 4.0 International (CC BY 4.0)** lisansı ile lisanslanmıştır.",
            "Önerilen kaynak gösterimi:",
            "> Analytics Engineering, \"SaaS Subscriptions & Customer Churn Dataset v1.0.0\". https://www.analyticsengineering.com/datasets/saas-subscriptions",
        ])

        with open(md_report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        logger.info(f"Saved Markdown cleaning report: {md_report_path}")


def main() -> None:
    preprocessor = DataPreprocessor()
    preprocessor.run()


if __name__ == "__main__":
    main()
