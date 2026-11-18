"""
B2B SaaS Churn Intelligence - Vize Sınavı EDA Dashboard Üreticisi
-----------------------------------------------------------------
Bu modül, 'Uygulamalı Veri Analizi' dersi vize sınavı için profesyonel,
baskıya hazır, tek sayfalık A4 yatay formatında Keşifsel Veri Analizi (EDA)
dashboardunu dinamik veri doğrulamasıyla üretir.

Tüm görünür metinler Türkçedir.
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
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# YAPILANDIRMA VE SABİTLER
# -----------------------------------------------------------------------------
STUDENT_NAME = "Ahmetcan Bağlı"
STUDENT_ID = "2516016010"
COURSE_TITLE = "Uygulamalı Veri Analizi Dersi - Vize Çalışması"
DATA_SOURCE_NAME = "Analytics Engineering, 'SaaS Subscriptions & Customer Churn Dataset v1.0.0'"
DATA_SOURCE_URL = "https://www.analyticsengineering.com/datasets/saas-subscriptions"
DATA_LICENSE = "Creative Commons Attribution 4.0 International (CC BY 4.0)"
DECLARED_DATE_RANGE = "01.01.2023 – 31.12.2025"

# Görsel sunum katmanı Türkçe kategori eşlemeleri (CSV'deki ham veriler korunur)
PLAN_TR_MAP = {
    "Enterprise": "Kurumsal",
    "Growth": "Büyüme",
    "Starter": "Başlangıç",
    "Scale": "Ölçek",
}

SEGMENT_TR_MAP = {
    "SMB": "KOBİ",
    "enterprise": "Kurumsal",
    "mid_market": "Orta Ölçekli Pazar",
}

CHANNEL_TR_MAP = {
    "organic": "Organik",
    "outbound": "Giden Pazarlama",
    "paid_search": "Ücretli Arama",
    "referral": "Tavsiye",
    "partner": "İş Ortağı",
}

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
# VERİ YÜKLEME VE ÇOK KATMANLI DOĞRULAMA MOTORU
# -----------------------------------------------------------------------------
def load_and_validate_data(
    csv_path: Path, eda_json_path: Path
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Veri setini ve mevcut EDA metriklerini yükler, çok katmanlı doğrulama yapar:
      1. Sütun bütünlüğü, geçerli değer kümeleri ve durum-churn çapraz tutarlılığı.
      2. Dinamik hesaplanan portföy metrikleri ile beklenen değer kontrolleri.
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

    # -------------------------------------------------------------------------
    # 1. ADIM: Sütun Bütünlüğü ve Durum-Churn Tutarlılık Kontrolleri
    # -------------------------------------------------------------------------
    status_null_count = int(df["status"].isna().sum())
    if status_null_count > 0:
        raise ValueError(
            f"Veri doğrulama hatası: 'status' sütununda {status_null_count} adet eksik (NaN/null) değer bulundu!"
        )

    churn_null_count = int(df["churn"].isna().sum())
    if churn_null_count > 0:
        raise ValueError(
            f"Veri doğrulama hatası: 'churn' sütununda {churn_null_count} adet eksik (NaN/null) değer bulundu!"
        )

    unique_statuses = set(df["status"].dropna().unique())
    expected_statuses = {"active", "cancelled"}
    if unique_statuses != expected_statuses:
        raise ValueError(
            f"Veri doğrulama hatası: 'status' sütununda beklenmeyen değerler var! "
            f"Beklenen: {expected_statuses}, Bulunan: {unique_statuses}"
        )

    unique_churn = set(df["churn"].dropna().unique())
    expected_churn = {0, 1}
    if unique_churn != expected_churn:
        raise ValueError(
            f"Veri doğrulama hatası: 'churn' sütununda beklenmeyen değerler var! "
            f"Beklenen: {expected_churn}, Bulunan: {unique_churn}"
        )

    active_mismatch = df[(df["status"] == "active") & (df["churn"] != 0)]
    if len(active_mismatch) > 0:
        raise ValueError(
            f"Çapraz doğrulama hatası: 'status' active olan {len(active_mismatch)} kaydın 'churn' değeri 0 değil!"
        )

    cancelled_mismatch = df[(df["status"] == "cancelled") & (df["churn"] != 1)]
    if len(cancelled_mismatch) > 0:
        raise ValueError(
            f"Çapraz doğrulama hatası: 'status' cancelled olan {len(cancelled_mismatch)} kaydın 'churn' değeri 1 değil!"
        )

    logger.info("Durum ve churn mantıksal tutarlılık kontrolleri başarıyla tamamlandı.")

    # -------------------------------------------------------------------------
    # 2. ADIM: Dinamik Hesaplamalar ve Temel Doğrulama
    # -------------------------------------------------------------------------
    total_records = len(df)
    active_count = int((df["status"] == "active").sum())
    churn_count = int((df["churn"] == 1).sum())
    overall_churn_rate = (churn_count / total_records) * 100 if total_records > 0 else 0.0

    if total_records != 2000:
        raise ValueError(f"Doğrulama hatası: Toplam kayıt 2000 olmalı, bulunan: {total_records}")
    if active_count != 1590:
        raise ValueError(f"Doğrulama hatası: Aktif abonelik 1590 olmalı, bulunan: {active_count}")
    if churn_count != 410:
        raise ValueError(f"Doğrulama hatası: İptal edilmiş abonelik 410 olmalı, bulunan: {churn_count}")
    if not math.isclose(overall_churn_rate, 20.50, abs_tol=FLOAT_ABS_TOL):
        raise ValueError(f"Doğrulama hatası: Genel churn oranı %20.50 olmalı, bulunan: %{overall_churn_rate:.2f}")

    # -------------------------------------------------------------------------
    # 3. ADIM: Mevcut EDA Raporu (eda_metrics.json) ile Karşılaştırma
    # -------------------------------------------------------------------------
    if total_records != eda_ref["total_subscriptions"]:
        raise ValueError(
            f"EDA JSON uyuşmazlığı: Toplam abonelik ({total_records} != {eda_ref['total_subscriptions']})"
        )
    if active_count != eda_ref["active_subscriptions"]:
        raise ValueError(
            f"EDA JSON uyuşmazlığı: Aktif abonelik ({active_count} != {eda_ref['active_subscriptions']})"
        )
    if churn_count != eda_ref["churned_subscriptions"]:
        raise ValueError(
            f"EDA JSON uyuşmazlığı: İptal abonelik ({churn_count} != {eda_ref['churned_subscriptions']})"
        )
    if not math.isclose(overall_churn_rate, eda_ref["overall_churn_rate_pct"], abs_tol=FLOAT_ABS_TOL):
        raise ValueError(
            f"EDA JSON uyuşmazlığı: Genel churn oranı (%{overall_churn_rate:.2f} != %{eda_ref['overall_churn_rate_pct']:.2f})"
        )

    # Plan bazlı hesaplama ve EDA JSON karşılaştırması
    plan_calc = (
        df.groupby("plan_name")["churn"]
        .agg(total="count", churned="sum", churn_rate=lambda s: s.mean() * 100)
        .to_dict(orient="index")
    )
    for p_name, exp_eda in eda_ref["plan_metrics"].items():
        if p_name not in plan_calc:
            raise ValueError(f"Plan doğrulama hatası: '{p_name}' planı dinamik veride bulunamadı.")
        calc = plan_calc[p_name]
        if calc["total"] != exp_eda["total"] or calc["churned"] != exp_eda["churned"]:
            raise ValueError(
                f"Plan '{p_name}' sayı uyuşmazlığı (EDA JSON): Dinamik (n={calc['total']}, churn={calc['churned']}), "
                f"EDA JSON (n={exp_eda['total']}, churn={exp_eda['churned']})"
            )
        if not math.isclose(calc["churn_rate"], exp_eda["churn_rate"], abs_tol=FLOAT_ABS_TOL):
            raise ValueError(
                f"Plan '{p_name}' oran uyuşmazlığı (EDA JSON): Dinamik %{calc['churn_rate']:.2f}, "
                f"EDA JSON %{exp_eda['churn_rate']:.2f}"
            )

    # Segment bazlı hesaplama ve EDA JSON karşılaştırması
    seg_calc = (
        df.groupby("segment")["churn"]
        .agg(total="count", churned="sum", churn_rate=lambda s: s.mean() * 100)
        .to_dict(orient="index")
    )
    for s_name, exp_eda in eda_ref["segment_metrics"].items():
        if s_name not in seg_calc:
            raise ValueError(f"Segment doğrulama hatası: '{s_name}' segmenti dinamik veride bulunamadı.")
        calc = seg_calc[s_name]
        if calc["total"] != exp_eda["total"] or calc["churned"] != exp_eda["churned"]:
            raise ValueError(
                f"Segment '{s_name}' sayı uyuşmazlığı (EDA JSON): Dinamik (n={calc['total']}, churn={calc['churned']}), "
                f"EDA JSON (n={exp_eda['total']}, churn={exp_eda['churned']})"
            )
        if not math.isclose(calc["churn_rate"], exp_eda["churn_rate"], abs_tol=FLOAT_ABS_TOL):
            raise ValueError(
                f"Segment '{s_name}' oran uyuşmazlığı (EDA JSON): Dinamik %{calc['churn_rate']:.2f}, "
                f"EDA JSON %{exp_eda['churn_rate']:.2f}"
            )

    # Kanal bazlı hesaplama ve EDA JSON karşılaştırması
    ch_calc = (
        df.groupby("acquisition_channel")["churn"]
        .agg(total="count", churned="sum", churn_rate=lambda s: s.mean() * 100)
        .to_dict(orient="index")
    )
    for c_name, exp_eda in eda_ref["channel_metrics"].items():
        if c_name not in ch_calc:
            raise ValueError(f"Kanal doğrulama hatası: '{c_name}' kanalı dinamik veride bulunamadı.")
        calc = ch_calc[c_name]
        if calc["total"] != exp_eda["total"] or calc["churned"] != exp_eda["churned"]:
            raise ValueError(
                f"Kanal '{c_name}' sayı uyuşmazlığı (EDA JSON): Dinamik (n={calc['total']}, churn={calc['churned']}), "
                f"EDA JSON (n={exp_eda['total']}, churn={exp_eda['churned']})"
            )
        if not math.isclose(calc["churn_rate"], exp_eda["churn_rate"], abs_tol=FLOAT_ABS_TOL):
            raise ValueError(
                f"Kanal '{c_name}' oran uyuşmazlığı (EDA JSON): Dinamik %{calc['churn_rate']:.2f}, "
                f"EDA JSON %{exp_eda['churn_rate']:.2f}"
            )

    # 4. Veri kalitesi sayıları
    missing_region_count = int(df["region"].isna().sum())
    missing_region_pct = (missing_region_count / total_records) * 100
    billing_data_count = int(df["has_billing_data"].sum())
    billing_data_pct = (billing_data_count / total_records) * 100
    payment_data_count = int(df["has_payment_data"].sum())
    payment_data_pct = (payment_data_count / total_records) * 100

    if missing_region_count != 479:
        raise ValueError(f"Eksik bölge sayısı doğrulama hatası: Beklenen 479, bulunan: {missing_region_count}")
    if billing_data_count != 120:
        raise ValueError(f"Fatura verisi kapsamı doğrulama hatası: Beklenen 120, bulunan: {billing_data_count}")
    if payment_data_count != 118:
        raise ValueError(f"Ödeme verisi kapsamı doğrulama hatası: Beklenen 118, bulunan: {payment_data_count}")

    logger.info("Tüm veri bütünlüğü ve EDA JSON çapraz kontrolleri başarıyla geçildi.")

    metrics: Dict[str, Any] = {
        "validation_status": "SUCCESS",
        "student_info": {
            "name": STUDENT_NAME,
            "student_id": STUDENT_ID,
            "course": COURSE_TITLE,
        },
        "data_provenance": {
            "source": DATA_SOURCE_NAME,
            "url": DATA_SOURCE_URL,
            "license": DATA_LICENSE,
            "declared_period": DECLARED_DATE_RANGE,
            "total_records": total_records,
        },
        "overall_kpis": {
            "total_subscriptions": total_records,
            "active_subscriptions": active_count,
            "active_pct": round((active_count / total_records) * 100, 2),
            "churned_subscriptions": churn_count,
            "churn_pct": round(overall_churn_rate, 2),
        },
        "plan_analysis": {
            PLAN_TR_MAP.get(name, name): {
                "raw_key": name,
                "total": int(data["total"]),
                "churned": int(data["churned"]),
                "churn_rate_pct": round(float(data["churn_rate"]), 2),
            }
            for name, data in plan_calc.items()
        },
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
# DASHBOARD ÇİZİM MOTORU (A4 Yatay: 297x210 mm, 300 DPI)
# -----------------------------------------------------------------------------
def build_full_dashboard(df: pd.DataFrame, metrics: Dict[str, Any]) -> plt.Figure:
    """
    A4 Yatay formatında, sıfır çakışmalı, profesyonel kurumsal BI dashboardu oluşturur.
    Tüm görünür metinler tamamen Türkçe, alt panel metinleri 7,6–7,8 puntodur.
    """
    fig_width = 11.693  # 297 mm
    fig_height = 8.268  # 210 mm

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="#F8FAFC")

    # -------------------------------------------------------------------------
    # 1. BÖLÜM: ÜST BAŞLIK ALANI (HEADER)
    # [left, bottom, width, height]
    # -------------------------------------------------------------------------
    ax_header = fig.add_axes([0.040, 0.868, 0.920, 0.098])
    ax_header.axis("off")

    # Sol Başlık Grubu (Tamamen Türkçe)
    ax_header.text(
        0.0, 0.96,
        "B2B MÜŞTERİ KAYBI ANALİZİ",
        fontsize=18,
        fontweight="bold",
        color="#0F172A",
        va="top",
        ha="left",
    )
    ax_header.text(
        0.0, 0.46,
        "SaaS Abonelik ve Terk Davranışları İncelemesi",
        fontsize=11.5,
        fontweight="bold",
        color="#1E3A8A",
        va="top",
        ha="left",
    )
    ax_header.text(
        0.0, 0.15,
        "Keşifsel Veri Analizi | 2023–2025",
        fontsize=8.8,
        fontweight="normal",
        color="#475569",
        va="top",
        ha="left",
    )

    # Sağ Başlık Bilgi Künyesi
    info_box_text = (
        f"Öğrenci: {STUDENT_NAME}  |  Öğrenci No: {STUDENT_ID}\n"
        f"Ders: {COURSE_TITLE}\n"
        f"Veri Kaynağı: {DATA_SOURCE_NAME} (CC BY 4.0)\n"
        f"Beyan Edilen Kapsam: {DECLARED_DATE_RANGE} (Bağımsız Doğrulanmamış)"
    )
    ax_header.text(
        1.0, 0.96,
        info_box_text,
        fontsize=7.2,
        color="#334155",
        va="top",
        ha="right",
        linespacing=1.32,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=0.8),
    )

    # -------------------------------------------------------------------------
    # 2. BÖLÜM: 4 ADET KPI GÖSTERGE KARTI
    # -------------------------------------------------------------------------
    ax_kpi = fig.add_axes([0.040, 0.752, 0.920, 0.096])
    ax_kpi.axis("off")

    kpis = [
        {
            "title": "TOPLAM ABONELİK",
            "value": f"{metrics['overall_kpis']['total_subscriptions']:,}".replace(",", "."),
            "sub": "Analiz Edilen Tüm Sözleşmeler",
            "accent": "#2563EB",  # Canlı Kurumsal Mavi
        },
        {
            "title": "AKTİF ABONELİK",
            "value": f"{metrics['overall_kpis']['active_subscriptions']:,}".replace(",", "."),
            "sub": f"Toplam aboneliklerin %{metrics['overall_kpis']['active_pct']:.2f}'si".replace(".", ","),
            "accent": "#16A34A",  # Pozitif Yeşil
        },
        {
            "title": "İPTAL EDİLMİŞ ABONELİK",
            "value": f"{metrics['overall_kpis']['churned_subscriptions']:,}".replace(",", "."),
            "sub": f"Toplam aboneliklerin %{metrics['overall_kpis']['churn_pct']:.2f}'si".replace(".", ","),
            "accent": "#DC2626",  # Uyarıcı Kırmızı
        },
        {
            "title": "GENEL MÜŞTERİ KAYBI ORANI",
            "value": f"%{metrics['overall_kpis']['churn_pct']:.2f}".replace(".", ","),
            "sub": "Genel oran referansı",
            "accent": "#D97706",  # Kehribar / Turuncu
        },
    ]

    card_width = 0.233
    card_gap = (1.0 - (4 * card_width)) / 3.0

    for idx, kpi in enumerate(kpis):
        x0 = idx * (card_width + card_gap)
        card_rect = patches.FancyBboxPatch(
            (x0, 0.0),
            card_width,
            1.0,
            boxstyle="round,pad=0.015,rounding_size=0.035",
            facecolor="#FFFFFF",
            edgecolor="#E2E8F0",
            linewidth=1.0,
            transform=ax_kpi.transAxes,
            clip_on=False,
        )
        ax_kpi.add_patch(card_rect)

        accent_bar = patches.FancyBboxPatch(
            (x0, 0.0),
            0.007,
            1.0,
            boxstyle="round,pad=0.0,rounding_size=0.02",
            facecolor=kpi["accent"],
            edgecolor="none",
            transform=ax_kpi.transAxes,
            clip_on=False,
        )
        ax_kpi.add_patch(accent_bar)

        ax_kpi.text(
            x0 + 0.016, 0.82,
            kpi["title"],
            fontsize=7.2,
            fontweight="bold",
            color="#64748B",
            va="top",
            ha="left",
            transform=ax_kpi.transAxes,
        )
        ax_kpi.text(
            x0 + 0.016, 0.48,
            kpi["value"],
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
            va="center",
            ha="left",
            transform=ax_kpi.transAxes,
        )
        ax_kpi.text(
            x0 + 0.016, 0.15,
            kpi["sub"],
            fontsize=6.8,
            fontweight="normal",
            color="#64748B",
            va="bottom",
            ha="left",
            transform=ax_kpi.transAxes,
        )

    # -------------------------------------------------------------------------
    # 3. BÖLÜM: 2x2 GRAFİK YERLEŞİMİ (KART ARKA PLANLARI İLE)
    # -------------------------------------------------------------------------
    baseline_val = metrics["overall_kpis"]["churn_pct"]

    # Kart 1 Arka Planı (Sol Üst)
    ax_card1 = fig.add_axes([0.040, 0.494, 0.445, 0.244])
    draw_card_background(ax_card1, "Genel Abonelik Durumu Dağılımı")

    # Kart 2 Arka Planı (Sağ Üst)
    ax_card2 = fig.add_axes([0.515, 0.494, 0.445, 0.244])
    draw_card_background(
        ax_card2, "Abonelik Planlarına Göre Müşteri Kaybı Oranı",
        has_ref_badge=True, baseline=baseline_val
    )

    # Kart 3 Arka Planı (Sol Alt) - Kart yüksekliği 0.276'ya çıkarılarak alt marjin artırıldı
    ax_card3 = fig.add_axes([0.040, 0.202, 0.445, 0.276])
    draw_card_background(
        ax_card3, "Müşteri Segmentlerine Göre Müşteri Kaybı Oranı",
        has_ref_badge=True, baseline=baseline_val
    )

    # Kart 4 Arka Planı (Sağ Alt) - Kart yüksekliği 0.276'ya çıkarılarak alt marjin artırıldı
    ax_card4 = fig.add_axes([0.515, 0.202, 0.445, 0.276])
    draw_card_background(
        ax_card4, "Müşteri Edinme Kanalına Göre Müşteri Kaybı Oranı",
        has_ref_badge=True, baseline=baseline_val
    )

    # --- GRAFİK 1: Genel Abonelik Durumu Dağılımı (Donut) ---
    ax_g1 = fig.add_axes([0.050, 0.502, 0.425, 0.185])
    render_chart_1_donut(ax_g1, metrics)

    # --- GRAFİK 2: Abonelik Planlarına Göre Müşteri Kaybı Oranı (Yatay Çubuk) ---
    # Alt eksen y=0.548, yükseklik 0.118; rozet ile tepe arasında 15 px, x ekseni altında 8 px ferah marjin
    ax_g2 = fig.add_axes([0.640, 0.548, 0.300, 0.118])
    render_chart_2_plans(ax_g2, metrics)

    # --- GRAFİK 3: Müşteri Segmentlerine Göre Müşteri Kaybı Oranı (Yatay Çubuk) ---
    # Alt eksen y=0.256'ya yükseltildi; x ekseni başlığı için kart altında 32 pt ferah marjin
    ax_g3 = fig.add_axes([0.220, 0.256, 0.245, 0.155])
    render_chart_3_segments(ax_g3, metrics)

    # --- GRAFİK 4: Müşteri Edinme Kanalına Göre Müşteri Kaybı Oranı (Yatay Çubuk) ---
    # Alt eksen y=0.256'ya yükseltildi; x ekseni başlığı için kart altında 32 pt ferah marjin
    ax_g4 = fig.add_axes([0.695, 0.256, 0.245, 0.155])
    render_chart_4_channels(ax_g4, metrics)

    # -------------------------------------------------------------------------
    # 4. BÖLÜM: ALT ALAN (VERİ KALİTESİ, ANALİTİK SINIRLILIKLAR VE KAYNAKÇA)
    # Baskı okunabilirliği ve taşma engelleme için Y: 0.018 - 0.186 (Yükseklik: 0.168)
    # -------------------------------------------------------------------------
    ax_footer = fig.add_axes([0.040, 0.018, 0.920, 0.168])
    render_footer_section(ax_footer, metrics)

    fig._dashboard_axes = {
        "card1": ax_card1,
        "card2": ax_card2,
        "card3": ax_card3,
        "card4": ax_card4,
        "g1": ax_g1,
        "g2": ax_g2,
        "g3": ax_g3,
        "g4": ax_g4,
        "footer": ax_footer,
        "header": ax_header,
        "kpi": ax_kpi,
    }

    return fig


def draw_card_background(
    ax: plt.Axes, title: str, has_ref_badge: bool = False, baseline: float = 20.50
) -> None:
    """Grafikler için kurumsal kart arka planı, başlık ve referans rozeti çizer."""
    ax.axis("off")
    bg = patches.FancyBboxPatch(
        (0.0, 0.0),
        1.0,
        1.0,
        boxstyle="round,pad=0.015,rounding_size=0.03",
        facecolor="#FFFFFF",
        edgecolor="#E2E8F0",
        linewidth=1.0,
        transform=ax.transAxes,
        zorder=0,
    )
    ax.add_patch(bg)
    ax.text(
        0.035, 0.92,
        title,
        fontsize=9.0,
        fontweight="bold",
        color="#0F172A",
        va="top",
        ha="left",
        transform=ax.transAxes,
    )

    if has_ref_badge:
        # Kartın üst bölümünde, veri alanının dışında, beyaz arka planlı referans rozeti
        badge_text = f"Kesikli çizgi: Genel oran %{baseline:.2f}".replace(".", ",")
        ax.text(
            0.965, 0.805,
            badge_text,
            fontsize=6.8,
            fontweight="bold",
            color="#DC2626",
            va="center",
            ha="right",
            transform=ax.transAxes,
            bbox=dict(
                boxstyle="round,pad=0.28",
                facecolor="#FFFFFF",
                edgecolor="#CBD5E1",
                linewidth=0.8,
            ),
        )



# -----------------------------------------------------------------------------
# GRAFİK ÇİZİM YARDIMCILARI (TÜRKÇE ETİKETLER)
# -----------------------------------------------------------------------------
def render_chart_1_donut(ax: plt.Axes, metrics: Dict[str, Any]) -> None:
    """
    Grafik 1: Genel Abonelik Durumu Dağılımı (Donut Grafik)
    Aktif: 1.590 (%79,50) - Yeşil (#16A34A)
    İptal Edilmiş: 410 (%20,50) - Kırmızı (#DC2626)
    """
    ax.axis("off")

    active_cnt = metrics["overall_kpis"]["active_subscriptions"]
    churn_cnt = metrics["overall_kpis"]["churned_subscriptions"]
    total_cnt = metrics["overall_kpis"]["total_subscriptions"]
    active_pct = metrics["overall_kpis"]["active_pct"]
    churn_pct = metrics["overall_kpis"]["churn_pct"]

    values = [active_cnt, churn_cnt]
    colors = ["#16A34A", "#DC2626"]

    # Donut pie çizimi
    ax.pie(
        values,
        center=(-0.30, 0.0),
        radius=0.78,
        startangle=140,
        colors=colors,
        wedgeprops=dict(width=0.33, edgecolor="#FFFFFF", linewidth=2.5),
        frame=False,
    )

    # Donut ortası:
    # 2.000 merkezin üst kısmında (y=0.16)
    # Toplam Abonelik sayının altında (y=-0.16) belirgin ve dengeli dikey boşlukla
    ax.text(
        -0.30, 0.16,
        f"{total_cnt:,}".replace(",", "."),
        ha="center",
        va="center",
        fontsize=13.0,
        fontweight="bold",
        color="#0F172A",
    )
    ax.text(
        -0.30, -0.16,
        "Toplam\nAbonelik",
        ha="center",
        va="center",
        fontsize=6.8,
        fontweight="bold",
        color="#64748B",
        linespacing=1.10,
    )

    # Sağ taraf bilgi kartları
    active_str = f"{active_cnt:,}".replace(",", ".")
    active_pct_str = f"%{active_pct:.2f}".replace(".", ",")
    churn_str = f"{churn_cnt:,}".replace(",", ".")
    churn_pct_str = f"%{churn_pct:.2f}".replace(".", ",")

    ax.text(
        0.52, 0.28,
        f"  Aktif Abonelik\n"
        f"  {active_str} Adet  |  {active_pct_str}",
        fontsize=8.5,
        fontweight="bold",
        color="#15803D",
        va="center",
        ha="left",
        linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#F0FDF4", edgecolor="#BBF7D0", linewidth=0.8),
    )

    ax.text(
        0.52, -0.32,
        f"  İptal Edilmiş Abonelik\n"
        f"  {churn_str} Adet  |  {churn_pct_str}",
        fontsize=8.5,
        fontweight="bold",
        color="#B91C1C",
        va="center",
        ha="left",
        linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#FEF2F2", edgecolor="#FECACA", linewidth=0.8),
    )


def render_dynamic_bar_labels(
    ax: plt.Axes, data_list: List[Dict[str, Any]], dark_color: str, baseline: float = 20.50
) -> None:
    """
    Yatay çubuk grafikler için dinamik etiket konumlandırması.
    Eğer dışarı yerleştirilen etiket kırmızı kesikli referans çizgisi (20.50) ile
    kesişecekse, etiket çubuğun içine (sağa dayalı, yüksek kontrastlı beyaz) yerleştirilir.
    Böylece hiçbir etiket referans çizgisi tarafından kesilmez.
    """
    for idx, d in enumerate(data_list):
        rate = d["rate"]
        val_str = f"%{rate:.2f}".replace(".", ",")

        # Referans çizgisi (20.50) altında kalan veya çizgiye çok yakın olan barlarda
        # etiketi çubuk içine (sağa dayalı, beyaz) alarak çizgi kesişmesini tamamen önle:
        # (Ölçek %19.12, Orta Ölçekli Pazar %17.16, Tavsiye %18.48, İş Ortağı %18.00, Başlangıç %20.31)
        will_collide = rate <= baseline + 0.1

        if will_collide:
            # Çubuk içine sağa dayalı beyaz kalın metin
            ax.text(
                rate - 0.7, idx,
                val_str,
                va="center",
                ha="right",
                fontsize=8.2,
                fontweight="bold",
                color="#FFFFFF",
                zorder=6,
            )
        else:
            # Çubuk dışına sola dayalı koyu renk metin
            x_pos = rate + 0.6
            # Eğer çubuk baseline'ı geçmişse ve baseline çizgisine çok yakınsa hafif kaydır
            if rate >= baseline and x_pos < baseline + 0.7:
                x_pos = baseline + 0.7
            ax.text(
                x_pos, idx,
                val_str,
                va="center",
                ha="left",
                fontsize=8.2,
                fontweight="bold",
                color=dark_color,
                zorder=6,
            )


def render_chart_2_plans(ax: plt.Axes, metrics: Dict[str, Any]) -> None:
    """
    Grafik 2: Abonelik Planlarına Göre Müşteri Kaybı Oranı (Yatay Çubuk Grafik)
    Türkçe Kategoriler: Kurumsal (%25.00) -> Büyüme (%20.63) -> Başlangıç (%20.31) -> Ölçek (%19.12)
    """
    ax.set_facecolor("#FFFFFF")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#CBD5E1")
        ax.spines[spine].set_linewidth(0.8)

    plan_data = []
    for tr_name, vals in metrics["plan_analysis"].items():
        plan_data.append({
            "name": tr_name,
            "rate": vals["churn_rate_pct"],
            "total": vals["total"],
            "churned": vals["churned"],
            "label": f"{tr_name} (n={vals['total']:,})".replace(",", "."),
        })

    plan_data.sort(key=lambda x: x["rate"], reverse=False)

    y_pos = np.arange(len(plan_data))
    rates = [d["rate"] for d in plan_data]
    labels = [d["label"] for d in plan_data]

    bars = ax.barh(y_pos, rates, height=0.52, color="#3B82F6", edgecolor="none", zorder=3)

    # Genel Ortalama Referans Çizgisi (%20,50) - Çizgi temiz, üzerinde metin bulunmaz
    baseline = metrics["overall_kpis"]["churn_pct"]
    ax.axvline(
        baseline,
        color="#DC2626",
        linestyle="--",
        linewidth=1.2,
        zorder=4,
    )

    render_dynamic_bar_labels(ax, plan_data, dark_color="#1E3A8A", baseline=baseline)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.0, fontweight="bold", color="#1E293B")
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    ax.set_xlim(0, 30.5)
    ax.set_ylim(-0.55, 3.55)
    ax.set_xlabel("Müşteri Kaybı Oranı (%)", fontsize=7.4, fontweight="bold", color="#475569", labelpad=2)
    ax.grid(axis="x", linestyle=":", alpha=0.6, color="#E2E8F0", zorder=1)
    ax.set_axisbelow(True)


def render_chart_3_segments(ax: plt.Axes, metrics: Dict[str, Any]) -> None:
    """
    Grafik 3: Müşteri Segmentlerine Göre Müşteri Kaybı Oranı (Yatay Çubuk Grafik)
    Türkçe Kategoriler: KOBİ (%21.54) -> Kurumsal (%21.49) -> Orta Ölçekli Pazar (%17.16)
    """
    ax.set_facecolor("#FFFFFF")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#CBD5E1")
        ax.spines[spine].set_linewidth(0.8)

    seg_data = []
    for tr_name, vals in metrics["segment_analysis"].items():
        seg_data.append({
            "name": tr_name,
            "rate": vals["churn_rate_pct"],
            "total": vals["total"],
            "churned": vals["churned"],
            "label": f"{tr_name} (n={vals['total']:,})".replace(",", "."),
        })

    seg_data.sort(key=lambda x: x["rate"], reverse=False)

    y_pos = np.arange(len(seg_data))
    rates = [d["rate"] for d in seg_data]
    labels = [d["label"] for d in seg_data]

    bars = ax.barh(y_pos, rates, height=0.48, color="#8B5CF6", edgecolor="none", zorder=3)

    baseline = metrics["overall_kpis"]["churn_pct"]
    ax.axvline(
        baseline,
        color="#DC2626",
        linestyle="--",
        linewidth=1.2,
        zorder=4,
    )

    render_dynamic_bar_labels(ax, seg_data, dark_color="#5B21B6", baseline=baseline)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7.8, fontweight="bold", color="#1E293B")
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xlim(0, 30.5)
    ax.set_ylim(-0.55, 2.55)
    ax.set_xlabel("Müşteri Kaybı Oranı (%)", fontsize=7.4, fontweight="bold", color="#475569", labelpad=2)
    ax.grid(axis="x", linestyle=":", alpha=0.6, color="#E2E8F0", zorder=1)
    ax.set_axisbelow(True)


def render_chart_4_channels(ax: plt.Axes, metrics: Dict[str, Any]) -> None:
    """
    Grafik 4: Müşteri Edinme Kanalına Göre Müşteri Kaybı Oranı (Yatay Çubuk Grafik)
    Türkçe Kategoriler: Organik (%22.60) -> Giden Pazarlama (%22.37) -> Ücretli Arama (%21.29)
                        -> Tavsiye (%18.48) -> İş Ortağı (%18.00)
    """
    ax.set_facecolor("#FFFFFF")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#CBD5E1")
        ax.spines[spine].set_linewidth(0.8)

    ch_data = []
    for tr_name, vals in metrics["channel_analysis"].items():
        ch_data.append({
            "name": tr_name,
            "rate": vals["churn_rate_pct"],
            "total": vals["total"],
            "churned": vals["churned"],
            "label": f"{tr_name} (n={vals['total']:,})".replace(",", "."),
        })

    ch_data.sort(key=lambda x: x["rate"], reverse=False)

    y_pos = np.arange(len(ch_data))
    rates = [d["rate"] for d in ch_data]
    labels = [d["label"] for d in ch_data]

    bars = ax.barh(y_pos, rates, height=0.52, color="#0EA5E9", edgecolor="none", zorder=3)

    baseline = metrics["overall_kpis"]["churn_pct"]
    ax.axvline(
        baseline,
        color="#DC2626",
        linestyle="--",
        linewidth=1.2,
        zorder=4,
    )

    render_dynamic_bar_labels(ax, ch_data, dark_color="#0369A1", baseline=baseline)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7.8, fontweight="bold", color="#1E293B")
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xlim(0, 30.5)
    ax.set_ylim(-0.55, 4.55)
    ax.set_xlabel("Müşteri Kaybı Oranı (%)", fontsize=7.4, fontweight="bold", color="#475569", labelpad=2)
    ax.grid(axis="x", linestyle=":", alpha=0.6, color="#E2E8F0", zorder=1)
    ax.set_axisbelow(True)


def render_footer_section(ax: plt.Axes, metrics: Dict[str, Any]) -> None:
    """
    Alt Bölüm: Veri Kalitesi, Analitik Sınırlılıklar ve Kaynakça
    Baskı okunabilirliği ve taşma koruması için punto boyutu 7.1–8.5 pt olarak optimize edilmiştir.
    Metinler kutu sınırlarına kusursuz uyum sağlayacak şekilde kontrollü satırlandırılmıştır.
    """
    ax.axis("off")

    # Sol Panel: VERİ KALİTESİ VE ANALİTİK SINIRLILIKLAR (Genişlik: 0.615)
    panel_left = patches.FancyBboxPatch(
        (0.0, 0.0),
        0.615,
        1.0,
        boxstyle="round,pad=0.015,rounding_size=0.03",
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
        linewidth=1.0,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(panel_left)

    accent_left = patches.FancyBboxPatch(
        (0.0, 0.0),
        0.007,
        1.0,
        boxstyle="round,pad=0.0,rounding_size=0.02",
        facecolor="#E11D48",  # Rose/Red
        edgecolor="none",
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(accent_left)

    ax.text(
        0.018, 0.90,
        "VERİ KALİTESİ VE ANALİTİK SINIRLILIKLAR",
        fontsize=8.5,
        fontweight="bold",
        color="#9F1239",
        va="top",
        ha="left",
        transform=ax.transAxes,
    )

    limitations_text = (
        "1. Coğrafi Bölge Eksikliği: 2.000 gözlemin 479 tanesinde (%23,95) eksiktir. Rastgele doldurulmamış,\n"
        "   tarafsızlığı korumak amacıyla eksik veri olarak tutulmuştur.\n"
        "2. Fatura & Ödeme Kapsamı: Fatura geçmişi 120 (%6,00), ödeme geçmişi 118 (%5,90) kayıtta mevcuttur.\n"
        "3. Metodolojik Uyarı: Fatura ve ödeme bulguları yalnızca sınırlı bir alt kohortu (%6) temsil eder;\n"
        "   tüm popülasyona genellenemez. Verinin doğrudan desteklemediği nedensellik çıkarımları kurulmamalıdır."
    )
    ax.text(
        0.018, 0.72,
        limitations_text,
        fontsize=7.2,
        color="#334155",
        va="top",
        ha="left",
        linespacing=1.28,
        transform=ax.transAxes,
    )

    # Sağ Panel: KAYNAK GÖSTERİMİ VE AKADEMİK BİLDİRİM (Genişlik: 0.365)
    panel_right = patches.FancyBboxPatch(
        (0.635, 0.0),
        0.365,
        1.0,
        boxstyle="round,pad=0.015,rounding_size=0.03",
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
        linewidth=1.0,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(panel_right)

    accent_right = patches.FancyBboxPatch(
        (0.635, 0.0),
        0.007,
        1.0,
        boxstyle="round,pad=0.0,rounding_size=0.02",
        facecolor="#3B82F6",
        edgecolor="none",
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(accent_right)

    ax.text(
        0.652, 0.90,
        "KAYNAK GÖSTERİMİ VE AKADEMİK BİLDİRİM",
        fontsize=8.5,
        fontweight="bold",
        color="#1E3A8A",
        va="top",
        ha="left",
        transform=ax.transAxes,
    )

    provenance_text = (
        f"Kaynak: Analytics Engineering, \"SaaS Subscriptions &\n"
        f"Customer Churn Dataset v1.0.0\"\n"
        f"URL: https://www.analyticsengineering.com/\n"
        f"datasets/saas-subscriptions\n"
        f"Lisans: Creative Commons Attribution 4.0 International (CC BY 4.0)\n"
        f"Veri Kapsamı: 2023–2025 kaynak beyanıdır (bağımsız doğrulanmamış)\n"
        f"Öğrenci: {STUDENT_NAME}  |  Öğrenci No: {STUDENT_ID}"
    )
    ax.text(
        0.652, 0.74,
        provenance_text,
        fontsize=7.1,
        color="#334155",
        va="top",
        ha="left",
        linespacing=1.22,
        transform=ax.transAxes,
    )


# -----------------------------------------------------------------------------
# ANA ÇALIŞTIRMA VE ÇIKTI ÜRETİM FONKSİYONU
# -----------------------------------------------------------------------------
def build_vize_dashboard() -> Tuple[Path, Path, Path]:
    """
    Tüm dashboard üretim sürecini koordine eder:
    Veriyi yükler, EDA JSON ile çapraz doğrular, grafikleri çizer,
    bounding box taşma denetimi yapar ve çıktıları kaydeder.
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

    tol_px = 3.0  # piksel toleransı

    def check_artist_within(artist, container_bbox, desc: str) -> None:
        bb = artist.get_window_extent(renderer)
        if bb.width == 0 or bb.height == 0:
            return
        if bb.x0 < container_bbox.x0 - tol_px:
            raise ValueError(
                f"Görsel sınır aşımı: {desc} kartın solundan taştı! (x0={bb.x0:.1f}, min={container_bbox.x0:.1f})"
            )
        if bb.x1 > container_bbox.x1 + tol_px:
            raise ValueError(
                f"Görsel sınır aşımı: {desc} kartın sağından taştı! (x1={bb.x1:.1f}, max={container_bbox.x1:.1f})"
            )
        if bb.y0 < container_bbox.y0 - tol_px:
            raise ValueError(
                f"Görsel sınır aşımı: {desc} kartın altından taştı! (y0={bb.y0:.1f}, min={container_bbox.y0:.1f})"
            )
        if bb.y1 > container_bbox.y1 + tol_px:
            raise ValueError(
                f"Görsel sınır aşımı: {desc} kartın üstünden taştı! (y1={bb.y1:.1f}, max={container_bbox.y1:.1f})"
            )

    axes_dict = getattr(fig, "_dashboard_axes", {})
    card_pairs = [
        (axes_dict.get("card1"), axes_dict.get("g1"), "Grafik 1 (Donut)"),
        (axes_dict.get("card2"), axes_dict.get("g2"), "Grafik 2 (Planlar)"),
        (axes_dict.get("card3"), axes_dict.get("g3"), "Grafik 3 (Segmentler)"),
        (axes_dict.get("card4"), axes_dict.get("g4"), "Grafik 4 (Kanallar)"),
    ]

    for ax_c, ax_p, name in card_pairs:
        if ax_c is None or ax_p is None:
            continue
        c_bb = ax_c.get_window_extent(renderer)
        for t in ax_c.texts:
            check_artist_within(t, c_bb, f"{name} - Kart Metni '{t.get_text()[:20]}'")
        if ax_p.xaxis.label.get_text():
            check_artist_within(ax_p.xaxis.label, c_bb, f"{name} - X Ekseni Başlığı '{ax_p.xaxis.label.get_text()}'")
        for tl in ax_p.get_xticklabels():
            if tl.get_text():
                check_artist_within(tl, c_bb, f"{name} - X Tick '{tl.get_text()}'")
        for yl in ax_p.get_yticklabels():
            if yl.get_text():
                check_artist_within(yl, c_bb, f"{name} - Y Tick '{yl.get_text()}'")
        for t in ax_p.texts:
            check_artist_within(t, c_bb, f"{name} - Sayısal Etiket '{t.get_text()}'")

    # Kesikli referans çizgisinin (%20,50) hiçbir veri etiketiyle kesişmediğini piksel düzeyinde doğrula
    baseline_val = metrics["overall_kpis"]["churn_pct"]
    for ax_key, name in [("g2", "Planlar"), ("g3", "Segmentler"), ("g4", "Kanallar")]:
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

    # Footer metinlerinin alt/sağ sınır taşma denetimi
    ax_footer = axes_dict.get("footer")
    if ax_footer is not None:
        f_bb = ax_footer.get_window_extent(renderer)
        for t in ax_footer.texts:
            check_artist_within(t, f_bb, f"Footer Metni '{t.get_text()[:25]}'")

    logger.info("Otomatik görsel yerleşim, metin sınırı ve referans çizgisi çakışmasızlık doğrulaması başarıyla geçti.")

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
