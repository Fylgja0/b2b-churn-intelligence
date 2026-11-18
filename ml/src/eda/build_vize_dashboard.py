"""
B2B SaaS Churn Intelligence - Vize Sınavı EDA Dashboard Üreticisi
-----------------------------------------------------------------
Bu modül, 'Uygulamalı Veri Analizi' dersi vize sınavı için profesyonel,
baskıya hazır, tek sayfalık A4 yatay formatında Keşifsel Veri Analizi (EDA)
dashboardunu dinamik veri doğrulamasıyla üretir.

Tüm görünür metinler Türkçedir ve tüm metrikler doğrudan CSV veri setinden
dinamik olarak hesaplanmaktadır.
Üretilen Çıktılar:
  - ml/reports/vize/churn_intelligence_dashboard.pdf (Baskıya hazır tek sayfa A4)
  - ml/reports/vize/churn_intelligence_dashboard.png (300 DPI yüksek çözünürlük)
  - ml/reports/vize/dashboard_metrics.json (Hesaplanan ve doğrulanan metrikler)
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Matplotlib cache dizini için yazılabilir alan belirleme
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib
matplotlib.use("Agg")  # Etkileşimsiz arka uç
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

# -----------------------------------------------------------------------------
# YAPILANDIRMA VE SABİTLER
# -----------------------------------------------------------------------------
STUDENT_NAME = "Ahmetcan Bağlı"
STUDENT_ID = "2516016010"
COURSE_TITLE = "Uygulamalı Veri Analizi Dersi - Vize Çalışması"
PROJECT_TITLE = "SaaS Abonelik Müşteri Kaybı: Keşifsel Veri Analizi"
DATA_SOURCE_NAME = "Analytics Engineering — SaaS Subscriptions & Customer Churn Dataset v1.0.0"
DATA_SOURCE_URL = "https://www.analyticsengineering.com/datasets/saas-subscriptions"
DATA_LICENSE = "Creative Commons Attribution 4.0 International (CC BY 4.0)"
DECLARED_DATE_RANGE = "01.01.2023 – 31.12.2025"

# Görsel sunum katmanı Türkçe kategori eşlemeleri (CSV'deki ham veriler korunur)
PLAN_TR_MAP = {
    "Enterprise": "Enterprise (Kurumsal Plan)",
    "Growth": "Büyüme",
    "Starter": "Başlangıç",
    "Scale": "Ölçek",
}

PLAN_AXIS_TR_MAP = {
    "Starter": "Başlangıç\n(Starter)",
    "Growth": "Büyüme\n(Growth)",
    "Scale": "Ölçek\n(Scale)",
    "Enterprise": "Enterprise\n(Kurumsal Plan)",
}

PLAN_BAR_TR_MAP = {
    "Enterprise": "Enterprise (n={n})\n(Kurumsal Plan)",
    "Growth": "Büyüme (n={n})",
    "Starter": "Başlangıç (n={n})",
    "Scale": "Ölçek (n={n})",
}

SEGMENT_TR_MAP = {
    "SMB": "KOBİ",
    "enterprise": "Kurumsal Müşteri",
    "mid_market": "Orta Ölçekli Pazar",
}

CHANNEL_TR_MAP = {
    "organic": "Organik",
    "outbound": "Giden Pazarlama",
    "paid_search": "Ücretli Arama",
    "referral": "Tavsiye",
    "partner": "İş Ortağı",
}

# Tutarlı Kavram Renk Paleti (Aynı kavram tüm grafiklerde aynı renkte)
COLOR_CHURN = "#DC2626"         # Kırmızı: Churn / Kayıp / Risk
COLOR_ACTIVE = "#16A34A"        # Yeşil: Aktif / Elde Tutma (Sağkalım)
COLOR_PLAN = "#1D4ED8"          # Mavi: Abonelik Planı / Toplam MRR Payı
COLOR_PLAN_DARK = "#1E3A8A"     # Koyu Mavi: Plan metin vurgusu
COLOR_SEGMENT = "#6D28D9"       # Mor: Müşteri Segmenti
COLOR_SEGMENT_DARK = "#5B21B6"  # Koyu Mor: Segment metin vurgusu
COLOR_CHANNEL = "#0F766E"       # Turkuaz: Edinme Kanalı
COLOR_CHANNEL_DARK = "#115E59"  # Koyu Turkuaz: Kanal metin vurgusu

# Tolerans sabitleri
FLOAT_ABS_TOL = 1e-4  # Yüzdelik oran karşılaştırma toleransı (%0.0001)

# Dosya yolları (pathlib ile güvenilir çözümleme)
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
ML_DIR = SRC_DIR.parent
REPO_ROOT = ML_DIR.parent

DATASET_PATH = ML_DIR / "data" / "processed" / "merged" / "account_level_eda_dataset.csv"
EDA_METRICS_PATH = ML_DIR / "reports" / "eda" / "eda_metrics.json"
OUTPUT_DIR = ML_DIR / "reports" / "vize"
OUTPUT_PDF_PATH = OUTPUT_DIR / "churn_intelligence_dashboard.pdf"
OUTPUT_PNG_PATH = OUTPUT_DIR / "churn_intelligence_dashboard.png"
OUTPUT_JSON_PATH = OUTPUT_DIR / "dashboard_metrics.json"

# Loglama ayarları
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("vize_dashboard")

# Tipografi ve Türkçe karakter desteği
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "Arial", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


# -----------------------------------------------------------------------------
# KAPLAN-MEIER SAĞKALIM (ELDE TUTMA) HESAPLAMA MOTORU
# -----------------------------------------------------------------------------
def compute_kaplan_meier(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], Dict[str, float]]:
    """
    tenure_days ve churn sütunlarını kullanarak Kaplan-Meier çarpım-limit (product-limit)
    elde tutma olasılığını hesaplar. Aktif abonelikler (churn == 0) sağdan sansürlüdür.
    Tam hassasiyetli değerler üzerinden koşullu dönemsel kayıp olasılıklarını q(a,b) hesaplar.
    """
    valid = df[["tenure_days", "churn"]].dropna()
    unique_times = np.sort(valid["tenure_days"].unique())

    km_times: List[float] = [0.0]
    km_survival: List[float] = [100.0]
    s_curr = 1.0

    for t in unique_times:
        if t < 0:
            continue
        n_at_risk = int((valid["tenure_days"] >= t).sum())
        d_events = int(((valid["tenure_days"] == t) & (valid["churn"] == 1)).sum())
        if n_at_risk > 0 and d_events > 0:
            s_curr *= 1.0 - (d_events / n_at_risk)
        km_times.append(float(t))
        km_survival.append(float(s_curr * 100.0))

    t_arr = np.array(km_times, dtype=float)
    s_arr = np.array(km_survival, dtype=float)

    def get_s_exact(day: float) -> float:
        idx = int(np.searchsorted(t_arr, day, side="right") - 1)
        idx = max(0, min(idx, len(s_arr) - 1))
        return float(s_arr[idx])

    milestones: Dict[str, float] = {}
    for day in [90, 180, 365, 540, 720]:
        milestones[f"day_{day}_pct"] = round(get_s_exact(day), 2)

    s0 = 100.0
    s180 = get_s_exact(180)
    s365 = get_s_exact(365)
    s540 = get_s_exact(540)

    q_0_180 = 1.0 - (s180 / s0)
    q_180_365 = 1.0 - (s365 / s180)
    q_365_540 = 1.0 - (s540 / s365)

    conditional_loss: Dict[str, float] = {
        "q_0_to_180_pct": round(float(q_0_180 * 100.0), 2),
        "q_180_to_365_pct": round(float(q_180_365 * 100.0), 2),
        "q_365_to_540_pct": round(float(q_365_540 * 100.0), 2),
        "q_0_to_180_exact": float(q_0_180),
        "q_180_to_365_exact": float(q_180_365),
        "q_365_to_540_exact": float(q_365_540),
    }

    return t_arr, s_arr, milestones, conditional_loss


# -----------------------------------------------------------------------------
# VERİ YÜKLEME VE ÇOK KATMANLI DOĞRULAMA MOTORU
# -----------------------------------------------------------------------------
def load_and_validate_data(
    csv_path: Path, eda_json_path: Path
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Veri setini ve mevcut EDA metriklerini yükler, tüm metrikleri doğrudan CSV'den
    dinamik olarak hesaplar ve çok katmanlı doğrulama yapar:
      1. Sütun bütünlüğü, geçerli değer kümeleri ve durum-churn çapraz tutarlılığı.
      2. Dinamik hesaplanan portföy, MRR, Kaplan-Meier ve Ki-Kare metriklerinin doğrulanması.
      3. Dinamik hesaplanan değerler ile ml/reports/eda/eda_metrics.json karşılaştırması.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Veri seti bulunamadı: {csv_path}")
    if not eda_json_path.exists():
        raise FileNotFoundError(f"EDA metrikleri JSON dosyası bulunamadı: {eda_json_path}")

    logger.info("Veri seti yükleniyor: %s", csv_path)
    df = pd.read_csv(csv_path)

    logger.info("EDA metrikleri referans dosyası yükleniyor: %s", eda_json_path)
    with open(eda_json_path, "r", encoding="utf-8") as f:
        eda_ref = json.load(f)

    # 1. ADIM: Sütun Bütünlüğü ve Durum-Churn Tutarlılık Kontrolleri
    for col_name in ["status", "churn", "mrr", "plan_name", "segment", "acquisition_channel"]:
        null_cnt = int(df[col_name].isna().sum())
        if null_cnt > 0:
            raise ValueError(f"Veri doğrulama hatası: '{col_name}' sütununda {null_cnt} adet eksik değer bulundu!")

    unique_statuses = set(df["status"].dropna().unique())
    if unique_statuses != {"active", "cancelled"}:
        raise ValueError(f"Beklenmeyen status değerleri: {unique_statuses}")

    unique_churn = set(df["churn"].dropna().unique())
    if unique_churn != {0, 1}:
        raise ValueError(f"Beklenmeyen churn değerleri: {unique_churn}")

    if len(df[(df["status"] == "active") & (df["churn"] != 0)]) > 0:
        raise ValueError("Çapraz doğrulama hatası: status=='active' olup churn!=0 olan kayıt var!")
    if len(df[(df["status"] == "cancelled") & (df["churn"] != 1)]) > 0:
        raise ValueError("Çapraz doğrulama hatası: status=='cancelled' olup churn!=1 olan kayıt var!")

    # 2. ADIM: Dinamik Temel Portföy ve MRR Hesaplamaları
    total_records = int(len(df))
    active_count = int((df["status"] == "active").sum())
    churn_count = int((df["churn"] == 1).sum())
    active_pct = (active_count / total_records) * 100.0
    overall_churn_rate = (churn_count / total_records) * 100.0

    total_mrr = float(df["mrr"].sum())
    active_mrr = float(df.loc[df["churn"] == 0, "mrr"].sum())
    churned_mrr = float(df.loc[df["churn"] == 1, "mrr"].sum())
    mrr_churn_pct = (churned_mrr / total_mrr) * 100.0

    # Kaplan-Meier Sağkalım Hesaplaması
    km_t, km_s, km_milestones, km_conditional = compute_kaplan_meier(df)
    retention_12m_pct = km_milestones["day_365_pct"]

    # EDA JSON ile Çapraz Doğrulama
    if total_records != eda_ref["total_subscriptions"]:
        raise ValueError(f"EDA JSON uyuşmazlığı: Toplam abonelik ({total_records} != {eda_ref['total_subscriptions']})")
    if active_count != eda_ref["active_subscriptions"]:
        raise ValueError(f"EDA JSON uyuşmazlığı: Aktif abonelik ({active_count} != {eda_ref['active_subscriptions']})")
    if churn_count != eda_ref["churned_subscriptions"]:
        raise ValueError(f"EDA JSON uyuşmazlığı: İptal abonelik ({churn_count} != {eda_ref['churned_subscriptions']})")
    if not math.isclose(overall_churn_rate, eda_ref["overall_churn_rate_pct"], abs_tol=FLOAT_ABS_TOL):
        raise ValueError("EDA JSON genel churn oranı uyuşmazlığı!")

    # 3. ADIM: Plan Bazlı Churn ve MRR Payı Hesaplamaları
    plan_order = ["Starter", "Growth", "Scale", "Enterprise"]
    plan_analysis: Dict[str, Any] = {}
    mrr_by_plan: Dict[str, Any] = {}

    for p_name in plan_order:
        sub = df[df["plan_name"] == p_name]
        p_tot = int(len(sub))
        p_ch = int(sub["churn"].sum())
        p_rate = (p_ch / p_tot) * 100.0

        exp_eda = eda_ref["plan_metrics"][p_name]
        if p_tot != exp_eda["total"] or p_ch != exp_eda["churned"]:
            raise ValueError(f"Plan '{p_name}' sayı uyuşmazlığı!")
        if not math.isclose(p_rate, exp_eda["churn_rate"], abs_tol=FLOAT_ABS_TOL):
            raise ValueError(f"Plan '{p_name}' oran uyuşmazlığı!")

        p_mrr_tot = float(sub["mrr"].sum())
        p_mrr_ch = float(sub.loc[sub["churn"] == 1, "mrr"].sum())
        p_mrr_share = (p_mrr_tot / total_mrr) * 100.0
        p_ch_mrr_share = (p_mrr_ch / churned_mrr) * 100.0
        p_within_mrr_churn = (p_mrr_ch / p_mrr_tot) * 100.0
        p_sub_share = (p_tot / total_records) * 100.0

        tr_key = PLAN_TR_MAP[p_name]
        plan_analysis[tr_key] = {
            "raw_key": p_name,
            "total": p_tot,
            "churned": p_ch,
            "subscription_share_pct": round(p_sub_share, 2),
            "churn_rate_pct": round(p_rate, 2),
            "total_mrr": round(p_mrr_tot, 2),
            "churned_mrr": round(p_mrr_ch, 2),
            "total_mrr_share_pct": round(p_mrr_share, 2),
            "churned_mrr_share_pct": round(p_ch_mrr_share, 2),
            "within_plan_mrr_churn_pct": round(p_within_mrr_churn, 2),
        }
        mrr_by_plan[tr_key] = {
            "raw_key": p_name,
            "total_mrr": round(p_mrr_tot, 2),
            "churned_mrr": round(p_mrr_ch, 2),
            "total_mrr_share_pct": round(p_mrr_share, 2),
            "churned_mrr_share_pct": round(p_ch_mrr_share, 2),
            "total_mrr_share_exact": p_mrr_share,
            "churned_mrr_share_exact": p_ch_mrr_share,
            "within_plan_mrr_churn_pct": round(p_within_mrr_churn, 2),
        }

    # MRR Paylarının %100'e toplandığını doğrula
    sum_total_mrr_share = sum(v["total_mrr_share_exact"] for v in mrr_by_plan.values())
    sum_churn_mrr_share = sum(v["churned_mrr_share_exact"] for v in mrr_by_plan.values())
    if not math.isclose(sum_total_mrr_share, 100.0, abs_tol=1e-4):
        raise ValueError(f"Toplam MRR payları %100'e toplanmıyor: {sum_total_mrr_share}")
    if not math.isclose(sum_churn_mrr_share, 100.0, abs_tol=1e-4):
        raise ValueError(f"Kayıp MRR payları %100'e toplanmıyor: {sum_churn_mrr_share}")

    # 4. ADIM: Segment ve Kanal Analizi
    seg_calc = (
        df.groupby("segment")["churn"]
        .agg(total="count", churned="sum", churn_rate=lambda s: s.mean() * 100.0)
        .to_dict(orient="index")
    )
    for s_name, exp_eda in eda_ref["segment_metrics"].items():
        calc = seg_calc[s_name]
        if calc["total"] != exp_eda["total"] or calc["churned"] != exp_eda["churned"]:
            raise ValueError(f"Segment '{s_name}' sayı uyuşmazlığı!")
        if not math.isclose(calc["churn_rate"], exp_eda["churn_rate"], abs_tol=FLOAT_ABS_TOL):
            raise ValueError(f"Segment '{s_name}' oran uyuşmazlığı!")

    ch_calc = (
        df.groupby("acquisition_channel")["churn"]
        .agg(total="count", churned="sum", churn_rate=lambda s: s.mean() * 100.0)
        .to_dict(orient="index")
    )
    for c_name, exp_eda in eda_ref["channel_metrics"].items():
        calc = ch_calc[c_name]
        if calc["total"] != exp_eda["total"] or calc["churned"] != exp_eda["churned"]:
            raise ValueError(f"Kanal '{c_name}' sayı uyuşmazlığı!")
        if not math.isclose(calc["churn_rate"], exp_eda["churn_rate"], abs_tol=FLOAT_ABS_TOL):
            raise ValueError(f"Kanal '{c_name}' oran uyuşmazlığı!")

    # 5. ADIM: Ki-Kare Bağımsızlık Testleri ve Enterprise vs Diğer Planlar Fisher Kesin Testi
    chi_square_tests: Dict[str, Any] = {}
    for col_key in ["plan_name", "segment", "acquisition_channel"]:
        ct = pd.crosstab(df[col_key], df["churn"])
        chi2_stat, p_val, dof, exp_grid = chi2_contingency(ct)
        min_exp = float(exp_grid.min())
        chi_square_tests[col_key] = {
            "chi2_statistic": round(float(chi2_stat), 4),
            "p_value": round(float(p_val), 4),
            "degrees_of_freedom": int(dof),
            "min_expected_frequency": round(min_exp, 2),
            "cochran_satisfied": bool(min_exp >= 5.0),
            "significant_at_0_05": bool(p_val < 0.05),
        }

    ent_mask = df["plan_name"] == "Enterprise"
    ent_ch = int((ent_mask & (df["churn"] == 1)).sum())
    ent_act = int((ent_mask & (df["churn"] == 0)).sum())
    oth_ch = int((~ent_mask & (df["churn"] == 1)).sum())
    oth_act = int((~ent_mask & (df["churn"] == 0)).sum())
    fisher_or, fisher_p = fisher_exact([[ent_ch, ent_act], [oth_ch, oth_act]])
    chi_square_tests["enterprise_vs_others_fisher"] = {
        "odds_ratio": round(float(fisher_or), 4),
        "p_value": round(float(fisher_p), 4),
        "significant_at_0_05": bool(fisher_p < 0.05),
    }

    # 6. ADIM: Veri Kalitesi ve Sınırlılık Metrikleri
    missing_region_count = int(df["region"].isna().sum())
    missing_region_pct = (missing_region_count / total_records) * 100.0
    billing_data_count = int(df["has_billing_data"].sum())
    billing_data_pct = (billing_data_count / total_records) * 100.0
    payment_data_count = int(df["has_payment_data"].sum())
    payment_data_pct = (payment_data_count / total_records) * 100.0

    events_up_to_90d = int(((df["tenure_days"] <= 90) & (df["churn"] == 1)).sum())
    d0_to_180_pp = round(100.0 - km_milestones["day_180_pct"], 2)
    d180_to_365_pp = round(km_milestones["day_180_pct"] - km_milestones["day_365_pct"], 2)
    d365_to_540_pp = round(km_milestones["day_365_pct"] - km_milestones["day_540_pct"], 2)

    logger.info("Tüm dinamik hesaplamalar, MRR payları, KM sağkalım ve Ki-Kare kontrolleri başarıyla geçildi.")

    metrics: Dict[str, Any] = {
        "validation_status": "SUCCESS",
        "student_info": {
            "name": STUDENT_NAME,
            "student_id": STUDENT_ID,
            "course": COURSE_TITLE,
            "project_title": PROJECT_TITLE,
        },
        "data_provenance": {
            "source": DATA_SOURCE_NAME,
            "url": DATA_SOURCE_URL,
            "license": DATA_LICENSE,
            "dataset_type": "Sentetik (simüle) veri seti",
            "observation_unit": f"{total_records:,} abonelik".replace(",", "."),
            "declared_period": DECLARED_DATE_RANGE,
            "total_records": total_records,
        },
        "overall_kpis": {
            "total_subscriptions": total_records,
            "active_subscriptions": active_count,
            "active_pct": round(active_pct, 2),
            "churned_subscriptions": churn_count,
            "churn_pct": round(overall_churn_rate, 2),
            "total_mrr": round(total_mrr, 2),
            "active_mrr": round(active_mrr, 2),
            "churned_mrr": round(churned_mrr, 2),
            "mrr_churn_pct": round(mrr_churn_pct, 2),
            "retention_12m_pct": retention_12m_pct,
        },
        "kaplan_meier": {
            "milestones": km_milestones,
            "conditional_loss_probabilities": km_conditional,
            "events_up_to_90d": events_up_to_90d,
            "six_month_drops_pp": {
                "d0_to_180_pp": d0_to_180_pp,
                "d180_to_365_pp": d180_to_365_pp,
                "d365_to_540_pp": d365_to_540_pp,
            },
            "curve_days": [float(x) for x in km_t if x <= 720],
            "curve_survival_pct": [round(float(km_s[i]), 4) for i, x in enumerate(km_t) if x <= 720],
        },
        "mrr_by_plan": {
            k: {
                "raw_key": v["raw_key"],
                "total_mrr": v["total_mrr"],
                "churned_mrr": v["churned_mrr"],
                "total_mrr_share_pct": v["total_mrr_share_pct"],
                "churned_mrr_share_pct": v["churned_mrr_share_pct"],
                "within_plan_mrr_churn_pct": v["within_plan_mrr_churn_pct"],
            }
            for k, v in mrr_by_plan.items()
        },
        "plan_analysis": plan_analysis,
        "segment_analysis": {
            SEGMENT_TR_MAP.get(name, name): {
                "raw_key": name,
                "total": int(data["total"]),
                "churned": int(data["churned"]),
                "churn_rate_pct": round(float(data["churn_rate"]), 2),
            }
            for name, data in seg_calc.items()
        },
        "channel_analysis": {
            CHANNEL_TR_MAP.get(name, name): {
                "raw_key": name,
                "total": int(data["total"]),
                "churned": int(data["churned"]),
                "churn_rate_pct": round(float(data["churn_rate"]), 2),
            }
            for name, data in ch_calc.items()
        },
        "chi_square_tests": chi_square_tests,
        "data_quality_limitations": {
            "missing_region_count": missing_region_count,
            "missing_region_pct": round(missing_region_pct, 2),
            "billing_data_coverage_count": billing_data_count,
            "billing_data_coverage_pct": round(billing_data_pct, 2),
            "payment_data_coverage_count": payment_data_count,
            "payment_data_coverage_pct": round(payment_data_pct, 2),
        },
    }

    return df, metrics


# -----------------------------------------------------------------------------
# YARDIMCI KART VE GRAFİK ÇİZİM FONKSİYONLARI (MİNİMUM 8 PT TİPOGRAFİ)
# -----------------------------------------------------------------------------
def draw_card_frame(
    ax: plt.Axes,
    title: str,
    subtitle: str,
    has_ref_badge: bool = False,
    baseline: float = 20.50,
    title_fontsize: float = 9.0,
) -> None:
    """
    Grafik panelleri için kurumsal kart çerçevesi, üst bulgu/soru başlığı,
    isteğe bağlı referans rozeti ve alt tek cümlelik 'ne gösteriyor' açıklamasını çizer.
    """
    ax.axis("off")
    bg = patches.FancyBboxPatch(
        (0.0, 0.0),
        1.0,
        1.0,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
        linewidth=1.0,
        transform=ax.transAxes,
        zorder=0,
    )
    ax.add_patch(bg)

    # Üst Başlık (Net bulgu cümlesi veya soru biçimi)
    ax.text(
        0.025, 0.945,
        title,
        fontsize=title_fontsize,
        fontweight="bold",
        color="#0F172A",
        va="top",
        ha="left",
        transform=ax.transAxes,
    )

    if has_ref_badge:
        badge_text = f"Kesikli çizgi: Genel %{baseline:.2f}".replace(".", ",")
        ax.text(
            0.975, 0.825,
            badge_text,
            fontsize=8.0,
            fontweight="bold",
            color=COLOR_CHURN,
            va="center",
            ha="right",
            transform=ax.transAxes,
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor="#FEF2F2",
                edgecolor="#FECACA",
                linewidth=0.8,
            ),
        )

    # Alt tek cümlelik "ne gösteriyor" alt başlığı (min 8.0 pt)
    ax.text(
        0.025, 0.032,
        subtitle,
        fontsize=8.0,
        fontstyle="italic",
        color="#334155",
        va="bottom",
        ha="left",
        transform=ax.transAxes,
    )


def render_chart_1_km_survival(ax: plt.Axes, metrics: Dict[str, Any]) -> None:
    """
    Grafik 1A (Sol Üst): Abonelik Süresine Göre Elde Tutma Eğrisi (Kaplan-Meier).
    x: Gün (0-720), y: Sağkalım / Elde Tutma Oranı (%), dikey eksen %0'dan başlar.
    180, 365 ve 540. günler işaretlenir ve değerleri gösterilir.
    """
    ax.set_facecolor("#FFFFFF")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#94A3B8")
        ax.spines[spine].set_linewidth(0.9)

    days = np.array(metrics["kaplan_meier"]["curve_days"], dtype=float)
    surv = np.array(metrics["kaplan_meier"]["curve_survival_pct"], dtype=float)
    milestones = metrics["kaplan_meier"]["milestones"]

    # Basamaklı Kaplan-Meier eğrisi (Aktif / Elde Tutma = Yeşil)
    ax.step(days, surv, where="post", color=COLOR_ACTIVE, linewidth=2.3, zorder=4, label="Kaplan-Meier Elde Tutma (%)")
    ax.fill_between(days, 0, surv, step="post", color=COLOR_ACTIVE, alpha=0.10, zorder=2)

    # 180, 365 ve 540. gün işaretçileri ve değer etiketleri (farklı yükseklik ve sütunlarda, çakışmasız)
    mark_configs = [
        (180, milestones["day_180_pct"], (115, 66), "180. Gün\n%{val}"),
        (365, milestones["day_365_pct"], (285, 48), "365. Gün\n%{val}"),
        (540, milestones["day_540_pct"], (465, 30), "540. Gün\n%{val}"),
    ]

    for day_x, val_y, text_xy, tpl in mark_configs:
        val_str = f"{val_y:.2f}".replace(".", ",")
        label_str = tpl.format(val=val_str)

        # Dikey kesikli kılavuz çizgisi
        ax.plot([day_x, day_x], [0, val_y], color="#64748B", linestyle="--", linewidth=0.9, zorder=3)
        # İşaret noktası
        ax.scatter([day_x], [val_y], color="#0F172A", s=30, zorder=6, edgecolors="#FFFFFF", linewidth=1.2)

        # Ok ve kutulu etiket (çizgi ve diğer kutularla kesişmeyecek konumda)
        ax.annotate(
            label_str,
            xy=(day_x, val_y),
            xytext=text_xy,
            ha="center",
            va="top",
            fontsize=7.6,
            fontweight="bold",
            color="#0F172A",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#F0FDF4", edgecolor="#86EFAC", linewidth=0.9),
            arrowprops=dict(arrowstyle="->", color="#15803D", lw=1.0, shrinkA=0, shrinkB=3),
            zorder=7,
        )

    ax.set_xlim(0, 735)
    ax.set_ylim(0, 106)
    ax.set_xticks([0, 90, 180, 270, 365, 450, 540, 630, 720])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="both", labelsize=8.0, colors="#1E293B")
    ax.set_xlabel("Abonelik Süresi (Gün, 0–720)", fontsize=8.2, fontweight="bold", color="#1E293B", labelpad=3)
    ax.set_ylabel("Elde Tutma Oranı (%)", fontsize=8.2, fontweight="bold", color="#1E293B", labelpad=3)
    ax.grid(axis="both", linestyle=":", alpha=0.65, color="#CBD5E1", zorder=1)
    ax.set_axisbelow(True)


def render_chart_2_mrr_by_plan(ax: plt.Axes, metrics: Dict[str, Any]) -> None:
    """
    Grafik 1B (Sağ Üst): Plan Bazında Gelir (MRR) Payı ve Kaybedilen MRR Payı (%).
    Enterprise, Scale, Growth, Starter sırasıyla iki gruplu çubuk grafik:
      - Toplam MRR Payı (%): Nötr Gri (#64748B)
      - Kaybedilen MRR Payı (%): Kayıp Kırmızısı (#DC2626)
    """
    ax.set_facecolor("#FFFFFF")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#94A3B8")
        ax.spines[spine].set_linewidth(0.9)

    raw_order = ["Enterprise", "Scale", "Growth", "Starter"]
    mrr_plan_map = {v["raw_key"]: v for v in metrics["mrr_by_plan"].values()}

    x_pos = np.arange(len(raw_order))
    width = 0.36
    offset = 0.19

    tot_shares = [mrr_plan_map[k]["total_mrr_share_pct"] for k in raw_order]
    lost_shares = [mrr_plan_map[k]["churned_mrr_share_pct"] for k in raw_order]
    x_labels = [PLAN_AXIS_TR_MAP[k] for k in raw_order]

    bars_tot = ax.bar(
        x_pos - offset,
        tot_shares,
        width=width,
        color="#64748B",
        edgecolor="none",
        label="Toplam MRR Payı (%)",
        zorder=3,
    )
    bars_lost = ax.bar(
        x_pos + offset,
        lost_shares,
        width=width,
        color=COLOR_CHURN,
        edgecolor="none",
        label="Kaybedilen MRR Payı (%)",
        zorder=3,
    )

    for bar, val in zip(bars_tot, tot_shares):
        val_str = f"%{val:.1f}".replace(".", ",")
        ax.text(
            bar.get_x() + bar.get_width() / 2.0 - 0.02,
            val + 1.0,
            val_str,
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
            color="#334155",
            zorder=5,
        )

    for bar, val in zip(bars_lost, lost_shares):
        val_str = f"%{val:.1f}".replace(".", ",")
        ax.text(
            bar.get_x() + bar.get_width() / 2.0 + 0.02,
            val + 1.0,
            val_str,
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
            color="#991B1B",
            zorder=5,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=8.0, fontweight="bold", color="#1E293B")
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.set_ylim(0, 53)
    ax.tick_params(axis="y", labelsize=8.0, colors="#1E293B")
    ax.set_ylabel("MRR Payı (%)", fontsize=8.2, fontweight="bold", color="#1E293B", labelpad=3)
    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="#F8FAFC",
        edgecolor="#CBD5E1",
        fontsize=8.0,
    )
    ax.grid(axis="y", linestyle=":", alpha=0.65, color="#CBD5E1", zorder=1)
    ax.set_axisbelow(True)


def render_dynamic_bar_labels(
    ax: plt.Axes, data_list: List[Dict[str, Any]], dark_color: str, baseline: float = 20.50
) -> None:
    """
    Yatay çubuk grafikler için dinamik etiket konumlandırması (min 8.0 pt).
    Değer etiketini max(çubuk, çizgi) değerinin sağına yerleştirerek kesikli referans
    çizgisiyle (20.50) çakışmasını tamamen önler.
    """
    for idx, d in enumerate(data_list):
        rate = d["rate"]
        val_str = f"%{rate:.2f}".replace(".", ",")
        x_pos = max(rate, baseline) + 0.65
        ax.text(
            x_pos,
            idx,
            val_str,
            va="center",
            ha="left",
            fontsize=8.0,
            fontweight="bold",
            color=dark_color,
            zorder=6,
        )


def render_horizontal_churn_chart(
    ax: plt.Axes,
    items: List[Dict[str, Any]],
    bar_color: str,
    dark_color: str,
    baseline: float,
    sort_by_rate: bool = True,
) -> None:
    """
    Alt satırdaki 3 yatay çubuk grafik (Plan, Segment, Kanal) için ortak çizim motoru.
    Tüm eksen ve veri etiketleri >= 8.0 pt boyutundadır.
    """
    ax.set_facecolor("#FFFFFF")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#94A3B8")
        ax.spines[spine].set_linewidth(0.9)

    items_ordered = sorted(items, key=lambda x: x["rate"], reverse=False) if sort_by_rate else list(items)
    y_pos = np.arange(len(items_ordered))
    rates = [d["rate"] for d in items_ordered]
    labels = [d["label"] for d in items_ordered]

    ax.barh(y_pos, rates, height=0.54, color=bar_color, edgecolor="none", zorder=3)

    ax.axvline(
        baseline,
        color=COLOR_CHURN,
        linestyle="--",
        linewidth=1.2,
        zorder=4,
    )

    render_dynamic_bar_labels(ax, items_ordered, dark_color=dark_color, baseline=baseline)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.0, fontweight="bold", color="#1E293B")
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xlim(0, 34.5)
    ax.set_ylim(-0.55, len(items_ordered) - 0.45)
    ax.tick_params(axis="x", labelsize=8.0, colors="#1E293B")
    ax.set_xlabel("Müşteri Kaybı Oranı (%)", fontsize=8.0, fontweight="bold", color="#1E293B", labelpad=2)
    ax.grid(axis="x", linestyle=":", alpha=0.65, color="#CBD5E1", zorder=1)
    ax.set_axisbelow(True)


def render_footer_section(ax: plt.Axes, metrics: Dict[str, Any]) -> None:
    """
    Alt Bölüm:
      1. Sol Kutu: DİKKAT: İSTATİSTİKSEL ANLAMLILIK VE METODOLOJİK UYARILAR (min 8.0 pt)
      2. Sağ Kutu: VERİ BİLGİSİ, KALİTE SINIRLILIKLARI VE KAYNAK GÖSTERİMİ (min 8.0 pt)
    """
    ax.axis("off")

    chi_plan_p = metrics["chi_square_tests"]["plan_name"]["p_value"]
    chi_seg_p = metrics["chi_square_tests"]["segment"]["p_value"]
    chi_ch_p = metrics["chi_square_tests"]["acquisition_channel"]["p_value"]

    dq = metrics["data_quality_limitations"]
    tot_rec = metrics["overall_kpis"]["total_subscriptions"]

    # --- SOL KUTU: DİKKAT KUTUSU (X: 0.000 -> 0.485, Genişlik: 0.485) ---
    panel_left = patches.FancyBboxPatch(
        (0.0, 0.0),
        0.485,
        1.0,
        boxstyle="round,pad=0.0,rounding_size=0.025",
        facecolor="#FFFBEB",
        edgecolor="#F59E0B",
        linewidth=1.1,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(panel_left)

    accent_left = patches.FancyBboxPatch(
        (0.0, 0.0),
        0.006,
        1.0,
        boxstyle="round,pad=0.0,rounding_size=0.015",
        facecolor=COLOR_CHURN,
        edgecolor="none",
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(accent_left)

    ax.text(
        0.014, 0.91,
        "DİKKAT: İSTATİSTİKSEL ANLAMLILIK VE METODOLOJİK UYARILAR",
        fontsize=8.4,
        fontweight="bold",
        color="#991B1B",
        va="top",
        ha="left",
        transform=ax.transAxes,
    )

    p_plan_s = f"{chi_plan_p:.2f}".replace(".", ",")
    p_seg_s = f"{chi_seg_p:.2f}".replace(".", ",")
    p_ch_s = f"{chi_ch_p:.2f}".replace(".", ",")
    p_fisher_s = f"{metrics['chi_square_tests']['enterprise_vs_others_fisher']['p_value']:.2f}".replace(".", ",")

    attention_text = (
        f"• İstatistiksel Anlamlılık: Plan (p = {p_plan_s}), segment (p = {p_seg_s}) ve kanal (p = {p_ch_s}) için\n"
        f"  p > 0,05'tir. Bu örneklemde gruplar arasında istatistiksel olarak anlamlı farklılık\n"
        f"  bulunduğuna dair kanıt yoktur; bu durum grupların eşit olduğunu kanıtlamaz.\n"
        f"• Sansürleme: Aktifler sağdan sansürlü olduğu için Kaplan-Meier eğrisi kullanılmıştır.\n"
        f"• Enterprise Planı: Diğer planlarla fark anlamlı değildir (Fisher kesin test p = {p_fisher_s})."
    )
    ax.text(
        0.014, 0.77,
        attention_text,
        fontsize=8.0,
        color="#1E293B",
        va="top",
        ha="left",
        linespacing=1.12,
        transform=ax.transAxes,
    )

    # --- SAĞ KUTU: VERİ BİLGİSİ VE KAYNAKÇA KUTUSU (X: 0.502 -> 1.000, Genişlik: 0.498) ---
    panel_right = patches.FancyBboxPatch(
        (0.502, 0.0),
        0.498,
        1.0,
        boxstyle="round,pad=0.0,rounding_size=0.025",
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
        linewidth=1.1,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(panel_right)

    accent_right = patches.FancyBboxPatch(
        (0.502, 0.0),
        0.006,
        1.0,
        boxstyle="round,pad=0.0,rounding_size=0.015",
        facecolor=COLOR_PLAN,
        edgecolor="none",
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(accent_right)

    ax.text(
        0.516, 0.91,
        "VERİ BİLGİSİ, KALİTE SINIRLILIKLARI VE KAYNAK GÖSTERİMİ",
        fontsize=8.4,
        fontweight="bold",
        color=COLOR_PLAN_DARK,
        va="top",
        ha="left",
        transform=ax.transAxes,
    )

    tot_str = f"{tot_rec:,}".replace(",", ".")
    reg_pct_s = f"%{dq['missing_region_pct']:.2f}".replace(".", ",")
    bil_pct_s = f"%{dq['billing_data_coverage_pct']:.2f}".replace(".", ",")
    pay_pct_s = f"%{dq['payment_data_coverage_pct']:.2f}".replace(".", ",")

    provenance_text = (
        f"• Veri Yapısı: Sentetik (simüle) | {tot_str} abonelik | Kapsam: 01.01.2023–31.12.2025.\n"
        f"• Eksik Bölge ({reg_pct_s}): {dq['missing_region_count']} abonelikte bölge eksiktir; veri setindeki haliyle korunmuştur.\n"
        f"• Fatura & Ödeme: {tot_str} aboneliğin {dq['billing_data_coverage_count']}'sinde ({bil_pct_s}) fatura, {dq['payment_data_coverage_count']}'inde ({pay_pct_s}) ödeme\n"
        f"  geçmişi bulunmaktadır; tüm popülasyona genellenemez.\n"
        f"• Kaynak: Analytics Engineering — SaaS Subscriptions & Customer Churn Dataset v1.0.0\n"
        f"  URL: https://www.analyticsengineering.com/datasets/saas-subscriptions (CC BY 4.0)"
    )
    ax.text(
        0.516, 0.77,
        provenance_text,
        fontsize=8.0,
        color="#1E293B",
        va="top",
        ha="left",
        linespacing=1.10,
        transform=ax.transAxes,
    )


# -----------------------------------------------------------------------------
# DASHBOARD ÇİZİM MOTORU (A4 Yatay: 297x210 mm, 300 DPI)
# -----------------------------------------------------------------------------
def build_full_dashboard(df: pd.DataFrame, metrics: Dict[str, Any]) -> plt.Figure:
    """
    A4 Yatay formatında, sıfır çakışmalı, tüm metinleri >= 8 pt ve KPI değerleri >= 20 pt
    olan profesyonel akademik EDA dashboardu oluşturur.
    """
    fig_width = 11.693  # 297 mm
    fig_height = 8.268  # 210 mm

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="#F8FAFC")

    # -------------------------------------------------------------------------
    # 1. BÖLÜM: ÜST BAŞLIK ALANI (HEADER)
    # -------------------------------------------------------------------------
    ax_header = fig.add_axes([0.030, 0.896, 0.940, 0.086])
    ax_header.axis("off")

    ax_header.text(
        0.0, 0.98,
        "B2B MÜŞTERİ KAYBI VE ELDE TUTMA ANALİZİ",
        fontsize=14.5,
        fontweight="bold",
        color="#0F172A",
        va="top",
        ha="left",
    )
    ax_header.text(
        0.0, 0.48,
        PROJECT_TITLE,
        fontsize=10.5,
        fontweight="bold",
        color=COLOR_PLAN_DARK,
        va="top",
        ha="left",
    )
    ax_header.text(
        0.0, 0.14,
        "Sentetik (Simüle) Veri Seti: 2.000 Abonelik | Kapsam: 01.01.2023 – 31.12.2025",
        fontsize=8.2,
        fontweight="normal",
        color="#475569",
        va="top",
        ha="left",
    )

    info_box_text = (
        f"Öğrenci: {STUDENT_NAME}  |  Öğrenci No: {STUDENT_ID}\n"
        f"Ders: {COURSE_TITLE}\n"
        "Kaynak: Analytics Engineering — SaaS Subscriptions\n"
        "& Customer Churn Dataset v1.0.0 (CC BY 4.0)"
    )
    ax_header.text(
        1.0, 0.98,
        info_box_text,
        fontsize=8.0,
        color="#1E293B",
        va="top",
        ha="right",
        linespacing=1.20,
        bbox=dict(boxstyle="round,pad=0.38", facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=0.9),
    )

    # -------------------------------------------------------------------------
    # 2. BÖLÜM: 6 ADET KPI GÖSTERGE KARTI (Değerler: 20.0 pt, Tanımlar: 8.0 pt)
    # -------------------------------------------------------------------------
    ax_kpi = fig.add_axes([0.030, 0.780, 0.940, 0.100])
    ax_kpi.axis("off")

    kpis_data = metrics["overall_kpis"]
    kpis = [
        {
            "title": "TOPLAM ABONELİK",
            "value": f"{kpis_data['total_subscriptions']:,}".replace(",", "."),
            "sub": "İncelenen tüm sözleşmeler",
            "accent": COLOR_PLAN,
            "val_color": "#0F172A",
        },
        {
            "title": "AKTİF ABONELİK",
            "value": f"{kpis_data['active_subscriptions']:,}".replace(",", "."),
            "sub": f"Toplamın %{kpis_data['active_pct']:.2f}'si aktif".replace(".", ","),
            "accent": COLOR_ACTIVE,
            "val_color": "#15803D",
        },
        {
            "title": "İPTAL EDİLEN ABONELİK",
            "value": f"{kpis_data['churned_subscriptions']:,}".replace(",", "."),
            "sub": f"Toplamın %{kpis_data['churn_pct']:.2f}'si iptal".replace(".", ","),
            "accent": COLOR_CHURN,
            "val_color": "#B91C1C",
        },
        {
            "title": "MÜŞTERİ KAYBI ORANI",
            "value": f"%{kpis_data['churn_pct']:.2f}".replace(".", ","),
            "sub": "Genel abonelik terk oranı",
            "accent": COLOR_CHURN,
            "val_color": "#B91C1C",
        },
        {
            "title": "İPTAL MRR PAYI (%)",
            "value": f"%{kpis_data['mrr_churn_pct']:.2f}".replace(".", ","),
            "sub": "İptal MRR / Toplam MRR",
            "accent": COLOR_CHURN,
            "val_color": "#B91C1C",
        },
        {
            "title": "12. AY ELDE TUTMA (%)",
            "value": f"%{kpis_data['retention_12m_pct']:.2f}".replace(".", ","),
            "sub": "365. gün sağkalım oranı",
            "accent": COLOR_ACTIVE,
            "val_color": "#15803D",
        },
    ]

    n_kpis = len(kpis)
    card_width = 0.154
    card_gap = (1.0 - (n_kpis * card_width)) / (n_kpis - 1)

    for idx, kpi in enumerate(kpis):
        x0 = idx * (card_width + card_gap)
        card_rect = patches.FancyBboxPatch(
            (x0, 0.0),
            card_width,
            1.0,
            boxstyle="round,pad=0.0,rounding_size=0.025",
            facecolor="#FFFFFF",
            edgecolor="#CBD5E1",
            linewidth=1.0,
            transform=ax_kpi.transAxes,
            clip_on=False,
        )
        ax_kpi.add_patch(card_rect)

        accent_bar = patches.FancyBboxPatch(
            (x0, 0.0),
            0.005,
            1.0,
            boxstyle="round,pad=0.0,rounding_size=0.015",
            facecolor=kpi["accent"],
            edgecolor="none",
            transform=ax_kpi.transAxes,
            clip_on=False,
        )
        ax_kpi.add_patch(accent_bar)

        ax_kpi.text(
            x0 + 0.009, 0.88,
            kpi["title"],
            fontsize=8.0,
            fontweight="bold",
            color="#475569",
            va="top",
            ha="left",
            transform=ax_kpi.transAxes,
        )
        ax_kpi.text(
            x0 + 0.009, 0.48,
            kpi["value"],
            fontsize=20.0,
            fontweight="bold",
            color=kpi["val_color"],
            va="center",
            ha="left",
            transform=ax_kpi.transAxes,
        )
        ax_kpi.text(
            x0 + 0.009, 0.10,
            kpi["sub"],
            fontsize=8.0,
            fontweight="normal",
            color="#334155",
            va="bottom",
            ha="left",
            linespacing=1.05,
            transform=ax_kpi.transAxes,
        )

    # -------------------------------------------------------------------------
    # 3. BÖLÜM: ÜST GRAFİK SATIRI (1A: KAPLAN-MEIER, 1B: PLAN BAZINDA MRR PAYI)
    # -------------------------------------------------------------------------
    tot_mrr_str = f"{kpis_data['total_mrr']:,.0f}".replace(",", ".")
    mrr_loss_str = f"%{kpis_data['mrr_churn_pct']:.2f}".replace(".", ",")
    km_365_str = f"%{kpis_data['retention_12m_pct']:.2f}".replace(".", ",")

    # Kart 1A: Kaplan-Meier Elde Tutma Eğrisi (Sol Üst)
    ax_card1 = fig.add_axes([0.030, 0.484, 0.460, 0.280])
    draw_card_frame(
        ax_card1,
        title=f"Abonelik Süresine Göre Elde Tutma Eğrisi (12. Ay: {km_365_str})",
        subtitle="Aktifler sansürlü sayılarak hesaplanan sağkalım olasılığı (dikey eksen %0'dan başlar).",
        title_fontsize=8.8,
    )
    ax_g1 = fig.add_axes([0.078, 0.556, 0.398, 0.168])
    render_chart_1_km_survival(ax_g1, metrics)

    # Kart 1B: Plan Bazında Gelir (MRR) Payı ve Kaybedilen MRR (Sağ Üst)
    ax_card2 = fig.add_axes([0.510, 0.484, 0.460, 0.280])
    draw_card_frame(
        ax_card2,
        title=f"Plan Bazında MRR ve Kayıp MRR Payı (Kayıp Payı: {mrr_loss_str})",
        subtitle="Toplam MRR Payı: Plan/Portföy; Kaybedilen MRR Payı: Plan Kayıp/Toplam Kayıp.",
        title_fontsize=8.8,
    )
    ax_g2 = fig.add_axes([0.558, 0.556, 0.398, 0.168])
    render_chart_2_mrr_by_plan(ax_g2, metrics)

    # -------------------------------------------------------------------------
    # 4. BÖLÜM: ALT GRAFİK SATIRI (3 PANEL YAN YANA: PLAN, SEGMENT, KANAL)
    # -------------------------------------------------------------------------
    baseline_val = kpis_data["churn_pct"]
    chi_p_plan = f"{metrics['chi_square_tests']['plan_name']['p_value']:.2f}".replace(".", ",")
    chi_p_seg = f"{metrics['chi_square_tests']['segment']['p_value']:.2f}".replace(".", ",")
    chi_p_ch = f"{metrics['chi_square_tests']['acquisition_channel']['p_value']:.2f}".replace(".", ",")

    # Kart 2A: Plan Bazında Müşteri Kaybı Oranı (Sol Alt) - Sıra: Enterprise, Scale, Growth, Starter (yukarıdan aşağıya)
    ax_card3 = fig.add_axes([0.030, 0.204, 0.302, 0.264])
    draw_card_frame(
        ax_card3,
        title="Hangi Planda Kayıp Oranı Daha Yüksek?",
        subtitle=f"Plan bazında müşteri kaybı oranı (Ki-kare p={chi_p_plan}).",
        has_ref_badge=True,
        baseline=baseline_val,
        title_fontsize=8.5,
    )
    plan_by_raw = {vals["raw_key"]: (tr_name, vals) for tr_name, vals in metrics["plan_analysis"].items()}
    # ax.barh y=0 en altta, y=3 en üstte çizdiği için yukarıdan aşağıya Enterprise, Scale, Growth, Starter görünmesi için:
    plan_order_bottom_to_top = ["Starter", "Growth", "Scale", "Enterprise"]
    plan_items = [
        {
            "name": plan_by_raw[rk][0],
            "rate": plan_by_raw[rk][1]["churn_rate_pct"],
            "label": PLAN_BAR_TR_MAP[rk].format(n=f"{plan_by_raw[rk][1]['total']:,}".replace(",", ".")),
        }
        for rk in plan_order_bottom_to_top
    ]
    ax_g3 = fig.add_axes([0.148, 0.278, 0.172, 0.125])
    render_horizontal_churn_chart(ax_g3, plan_items, COLOR_PLAN, COLOR_PLAN_DARK, baseline_val, sort_by_rate=False)

    # Kart 2B: Segment Bazında Müşteri Kaybı Oranı (Orta Alt)
    ax_card4 = fig.add_axes([0.349, 0.204, 0.302, 0.264])
    draw_card_frame(
        ax_card4,
        title="Segmentlere Göre Kayıp Farklılaşıyor mu?",
        subtitle=f"Segment bazında müşteri kaybı oranı (Ki-kare p={chi_p_seg}).",
        has_ref_badge=True,
        baseline=baseline_val,
        title_fontsize=8.5,
    )
    seg_items = [
        {
            "name": tr_name,
            "rate": vals["churn_rate_pct"],
            "label": f"{tr_name}\n(n={vals['total']:,})".replace(",", "."),
        }
        for tr_name, vals in metrics["segment_analysis"].items()
    ]
    ax_g4 = fig.add_axes([0.467, 0.278, 0.172, 0.125])
    render_horizontal_churn_chart(ax_g4, seg_items, COLOR_SEGMENT, COLOR_SEGMENT_DARK, baseline_val)

    # Kart 2C: Edinme Kanalına Göre Müşteri Kaybı Oranı (Sağ Alt)
    ax_card5 = fig.add_axes([0.668, 0.204, 0.302, 0.264])
    draw_card_frame(
        ax_card5,
        title="Edinme Kanalına Göre Kayıp Oranı Farklılaşıyor mu?",
        subtitle=f"Kanal bazında müşteri kaybı oranı (Ki-kare p={chi_p_ch}).",
        has_ref_badge=True,
        baseline=baseline_val,
        title_fontsize=8.0,
    )
    ch_items = [
        {
            "name": tr_name,
            "rate": vals["churn_rate_pct"],
            "label": f"{tr_name} (n={vals['total']:,})".replace(",", "."),
        }
        for tr_name, vals in metrics["channel_analysis"].items()
    ]
    ax_g5 = fig.add_axes([0.812, 0.278, 0.146, 0.125])
    render_horizontal_churn_chart(ax_g5, ch_items, COLOR_CHANNEL, COLOR_CHANNEL_DARK, baseline_val)

    # -------------------------------------------------------------------------
    # 5. BÖLÜM: ALT ALAN (DİKKAT KUTUSU VE VERİ BİLGİSİ / KAYNAKÇA KUTUSU)
    # -------------------------------------------------------------------------
    ax_footer = fig.add_axes([0.030, 0.016, 0.940, 0.174])
    render_footer_section(ax_footer, metrics)

    fig._dashboard_axes = {
        "card1": ax_card1,
        "card2": ax_card2,
        "card3": ax_card3,
        "card4": ax_card4,
        "card5": ax_card5,
        "g1": ax_g1,
        "g2": ax_g2,
        "g3": ax_g3,
        "g4": ax_g4,
        "g5": ax_g5,
        "footer": ax_footer,
        "header": ax_header,
        "kpi": ax_kpi,
    }

    return fig


# -----------------------------------------------------------------------------
# ANA ÇALIŞTIRMA VE OTOMATİK DOĞRULAMA FONKSİYONU
# -----------------------------------------------------------------------------
def build_vize_dashboard() -> Tuple[Path, Path, Path]:
    """
    Tüm dashboard üretim sürecini koordine eder:
    Veriyi yükler, tüm metrikleri dinamik hesaplar, grafikleri çizer,
    piksel düzeyinde görsel sınır / çakışma denetimi yapar ve çıktıları kaydeder.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Veri yükleme ve çok katmanlı doğrulama
    df, metrics = load_and_validate_data(DATASET_PATH, EDA_METRICS_PATH)

    # 2. JSON metrik raporunu kaydet
    logger.info("Doğrulanmış metrikler JSON formatında kaydediliyor: %s", OUTPUT_JSON_PATH)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # 3. Dashboard figürünü oluştur
    logger.info("A4 yatay profesyonel dashboard figürü çiziliyor...")
    fig = build_full_dashboard(df, metrics)

    # 4. Çizim ve görsel sınır aşımı (bounding box) & referans çizgisi çakışmasızlığı doğrulaması
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    canvas_w, canvas_h = canvas.get_width_height()

    tol_px = 3.5  # piksel toleransı

    def check_artist_within(artist, container_bbox, desc: str, tol: float = tol_px) -> None:
        bb = artist.get_window_extent(renderer)
        if bb.width == 0 or bb.height == 0:
            return
        if bb.x0 < container_bbox.x0 - tol:
            raise ValueError(
                f"Görsel sınır aşımı: {desc} alanın solundan taştı! (x0={bb.x0:.1f}, min={container_bbox.x0:.1f})"
            )
        if bb.x1 > container_bbox.x1 + tol:
            raise ValueError(
                f"Görsel sınır aşımı: {desc} alanın sağından taştı! (x1={bb.x1:.1f}, max={container_bbox.x1:.1f})"
            )
        if bb.y0 < container_bbox.y0 - tol:
            raise ValueError(
                f"Görsel sınır aşımı: {desc} alanın altından taştı! (y0={bb.y0:.1f}, min={container_bbox.y0:.1f})"
            )
        if bb.y1 > container_bbox.y1 + tol:
            raise ValueError(
                f"Görsel sınır aşımı: {desc} alanın üstünden taştı! (y1={bb.y1:.1f}, max={container_bbox.y1:.1f})"
            )

    axes_dict = getattr(fig, "_dashboard_axes", {})

    # A) HEADER DOĞRULAMASI
    ax_header = axes_dict.get("header")
    if ax_header is not None:
        h_bb = ax_header.get_window_extent(renderer)
        for idx, t in enumerate(ax_header.texts):
            check_artist_within(t, h_bb, f"Header Metni {idx} ('{t.get_text()[:25]}')")
            t_bb = t.get_window_extent(renderer)
            if t_bb.x0 < 0 or t_bb.x1 > canvas_w or t_bb.y0 < 0 or t_bb.y1 > canvas_h:
                raise ValueError(f"Header metni sayfa sınırları dışına taştı: '{t.get_text()[:25]}'")

    # B) 6 ADET KPI KARTININ BAĞIMSIZ VE ÇAPRAZ ÇAKIŞMA DOĞRULAMASI
    ax_kpi = axes_dict.get("kpi")
    if ax_kpi is not None:
        n_kpis = 6
        for idx in range(n_kpis):
            card_patch = ax_kpi.patches[idx * 2]
            c_bb = card_patch.get_window_extent(renderer)

            t_title = ax_kpi.texts[idx * 3]
            t_val = ax_kpi.texts[idx * 3 + 1]
            t_sub = ax_kpi.texts[idx * 3 + 2]

            check_artist_within(t_title, c_bb, f"KPI Kart {idx + 1} - Başlık '{t_title.get_text()}'")
            check_artist_within(t_val, c_bb, f"KPI Kart {idx + 1} - Değer '{t_val.get_text()}'")
            check_artist_within(t_sub, c_bb, f"KPI Kart {idx + 1} - Alt Açıklama '{t_sub.get_text()}'")

            for other_idx in range(n_kpis):
                if other_idx != idx:
                    other_bb = ax_kpi.patches[other_idx * 2].get_window_extent(renderer)
                    for t_elem in (t_title, t_val, t_sub):
                        t_bb = t_elem.get_window_extent(renderer)
                        if (
                            t_bb.x0 < other_bb.x1
                            and t_bb.x1 > other_bb.x0
                            and t_bb.y0 < other_bb.y1
                            and t_bb.y1 > other_bb.y0
                        ):
                            raise ValueError(
                                f"KPI kartları arası çakışma: Kart {idx + 1} metni '{t_elem.get_text()}' "
                                f"Kart {other_idx + 1} sınırları içerisine taştı!"
                            )

    # C) FOOTER PANELLERİNİN BAĞIMSIZ VE ÇAPRAZ ÇAKIŞMA DOĞRULAMASI
    ax_footer = axes_dict.get("footer")
    if ax_footer is not None:
        p_left_bb = ax_footer.patches[0].get_window_extent(renderer)
        p_right_bb = ax_footer.patches[2].get_window_extent(renderer)

        t_l_title = ax_footer.texts[0]
        t_l_body = ax_footer.texts[1]
        check_artist_within(t_l_title, p_left_bb, "Footer Sol Panel (Dikkat) - Başlık")
        check_artist_within(t_l_body, p_left_bb, "Footer Sol Panel (Dikkat) - Gövde Metni")

        for t_l in (t_l_title, t_l_body):
            if t_l.get_window_extent(renderer).x1 > p_right_bb.x0:
                raise ValueError("Footer sol panel metni sağ panele taştı!")

        t_r_title = ax_footer.texts[2]
        t_r_body = ax_footer.texts[3]
        check_artist_within(t_r_title, p_right_bb, "Footer Sağ Panel (Veri Bilgisi) - Başlık")
        check_artist_within(t_r_body, p_right_bb, "Footer Sağ Panel (Veri Bilgisi) - Gövde Metni")

        for t_r in (t_r_title, t_r_body):
            if t_r.get_window_extent(renderer).x0 < p_left_bb.x1:
                raise ValueError("Footer sağ panel metni sol panele taştı!")

    # D) 5 GRAFİK KARTI VE ELEMANLARININ DOĞRULAMASI (+ METİN-METİN ÇAKIŞMA KONTROLÜ)
    card_pairs = [
        (axes_dict.get("card1"), axes_dict.get("g1"), "Grafik 1A (Kaplan-Meier)"),
        (axes_dict.get("card2"), axes_dict.get("g2"), "Grafik 1B (Plan MRR)"),
        (axes_dict.get("card3"), axes_dict.get("g3"), "Grafik 2A (Plan Kayıp)"),
        (axes_dict.get("card4"), axes_dict.get("g4"), "Grafik 2B (Segment Kayıp)"),
        (axes_dict.get("card5"), axes_dict.get("g5"), "Grafik 2C (Kanal Kayıp)"),
    ]

    for ax_c, ax_p, name in card_pairs:
        if ax_c is None or ax_p is None:
            continue
        c_bb = ax_c.get_window_extent(renderer)
        for t in ax_c.texts:
            check_artist_within(t, c_bb, f"{name} - Kart Metni '{t.get_text()[:25]}'")
        if ax_p.xaxis.label.get_text():
            check_artist_within(ax_p.xaxis.label, c_bb, f"{name} - X Ekseni Başlığı '{ax_p.xaxis.label.get_text()}'")
        if ax_p.yaxis.label.get_text():
            check_artist_within(ax_p.yaxis.label, c_bb, f"{name} - Y Ekseni Başlığı '{ax_p.yaxis.label.get_text()}'")
        for tl in ax_p.get_xticklabels():
            if tl.get_text():
                check_artist_within(tl, c_bb, f"{name} - X Tick '{tl.get_text()}'")
        for yl in ax_p.get_yticklabels():
            if yl.get_text():
                check_artist_within(yl, c_bb, f"{name} - Y Tick '{yl.get_text()}'")
        for t in ax_p.texts:
            check_artist_within(t, c_bb, f"{name} - Sayısal Etiket '{t.get_text()}'")

        # Aynı grafik içerisindeki metinlerin ve eksen etiketlerinin birbiriyle çakışmadığını doğrula
        plot_texts = [
            t for t in list(ax_p.texts) + list(ax_p.get_yticklabels()) + list(ax_p.get_xticklabels())
            if t.get_text().strip()
        ]
        for i_t in range(len(plot_texts)):
            for j_t in range(i_t + 1, len(plot_texts)):
                b1 = plot_texts[i_t].get_window_extent(renderer)
                b2 = plot_texts[j_t].get_window_extent(renderer)
                if b1.x0 < b2.x1 - 1.0 and b1.x1 > b2.x0 + 1.0 and b1.y0 < b2.y1 - 1.0 and b1.y1 > b2.y0 + 1.0:
                    raise ValueError(
                        f"Metin çakışması tespit edildi ({name}): '{plot_texts[i_t].get_text()}' "
                        f"ile '{plot_texts[j_t].get_text()}' üst üste biniyor!"
                    )

    # E) KESİKLİ REFERANS ÇİZGİSİNİN (%20,50) PİKSEL DÜZEYİNDE ÇAKIŞMASIZLIK DOĞRULAMASI
    baseline_val = metrics["overall_kpis"]["churn_pct"]
    for ax_key, name in [("g3", "Planlar"), ("g4", "Segmentler"), ("g5", "Kanallar")]:
        ax_p = axes_dict.get(ax_key)
        if ax_p is None:
            continue
        x_line_px = ax_p.transData.transform((baseline_val, 0))[0]
        for t in ax_p.texts:
            t_bb = t.get_window_extent(renderer)
            if (t_bb.x0 + 1.5) < x_line_px < (t_bb.x1 - 1.5):
                raise ValueError(
                    f"Referans çizgisi çakışması tespit edildi! {name} grafiğinde '{t.get_text()}' "
                    f"etiketi referans çizgisiyle kesişiyor! (x_line={x_line_px:.1f}, text_x=[{t_bb.x0:.1f}, {t_bb.x1:.1f}])"
                )

    logger.info(
        "Otomatik görsel yerleşim (Header, 6 KPI Kartı, 5 Grafik Paneli, 2 Footer Kutusu), "
        "metinler arası çakışmasızlık ve referans çizgisi doğrulaması başarıyla tamamlandı."
    )

    # 5. PDF ve PNG olarak kaydet
    logger.info("PDF kaydediliyor (300 DPI, yüksek kaliteli baskı): %s", OUTPUT_PDF_PATH)
    fig.savefig(OUTPUT_PDF_PATH, format="pdf", dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")

    logger.info("PNG kaydediliyor (300 DPI): %s", OUTPUT_PNG_PATH)
    fig.savefig(OUTPUT_PNG_PATH, format="png", dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")

    plt.close(fig)

    logger.info("Tüm çıktılar başarıyla oluşturuldu:")
    logger.info("  1. PDF: %s", OUTPUT_PDF_PATH)
    logger.info("  2. PNG: %s", OUTPUT_PNG_PATH)
    logger.info("  3. JSON: %s", OUTPUT_JSON_PATH)

    return OUTPUT_PDF_PATH, OUTPUT_PNG_PATH, OUTPUT_JSON_PATH


if __name__ == "__main__":
    try:
        build_vize_dashboard()
        print("\n✅ Vize dashboard başarıyla üretildi ve doğrulandı.")
    except Exception as exc:
        logger.error("Dashboard üretimi sırasında hata oluştu: %s", exc, exc_info=True)
        sys.exit(1)
