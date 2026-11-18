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
# YAPILANDIRMA VE SABİTLER (Kolayca düzenlenebilir alanlar)
# -----------------------------------------------------------------------------
STUDENT_NAME = "Ahmetcan Bağlı"
# Öğrenci Numarası: Rastgele veya uydurma değer verilmemiştir.
# Sınav teslimi öncesinde gerekirse aşağıdaki değişkene gerçek numaranızı yazınız.
STUDENT_ID = "Belirtilmedi"
COURSE_TITLE = "Uygulamalı Veri Analizi Dersi - Vize Çalışması"
DATA_SOURCE_NAME = "Analytics Engineering, 'SaaS Subscriptions & Customer Churn Dataset v1.0.0'"
DATA_SOURCE_URL = "https://www.analyticsengineering.com/datasets/saas-subscriptions"
DATA_LICENSE = "Creative Commons Attribution 4.0 International (CC BY 4.0)"
DECLARED_DATE_RANGE = "01.01.2023 – 31.12.2025"

# Dosya yolları (pathlib ile güvenilir çözümleme)
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
ML_DIR = SRC_DIR.parent
REPO_ROOT = ML_DIR.parent

DATASET_PATH = ML_DIR / "data" / "processed" / "merged" / "account_level_eda_dataset.csv"
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
# VERİ YÜKLEME VE HESAPLAMA MOTORU
# -----------------------------------------------------------------------------
def load_and_validate_data(csv_path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Veri setini yükler, EDA metriklerini dinamik olarak hesaplar
    ve beklenen değerlerle titiz doğrulama gerçekleştirir.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Veri seti bulunamadı: {csv_path}")

    logger.info("Veri seti yükleniyor: %s", csv_path)
    df = pd.read_csv(csv_path)

    total_records = len(df)
    active_count = int((df["status"] == "active").sum())
    churn_count = int((df["churn"] == 1).sum())
    overall_churn_rate = (churn_count / total_records) * 100 if total_records > 0 else 0.0

    # Doğrulama: Toplam gözlem ve durum dağılımı
    if total_records != 2000:
        raise ValueError(f"Veri doğrulama hatası: Toplam kayıt 2000 olmalı, bulunan: {total_records}")
    if active_count != 1590:
        raise ValueError(f"Veri doğrulama hatası: Aktif abonelik 1590 olmalı, bulunan: {active_count}")
    if churn_count != 410:
        raise ValueError(f"Veri doğrulama hatası: İptal edilmiş abonelik 410 olmalı, bulunan: {churn_count}")
    if not math.isclose(overall_churn_rate, 20.50, abs_tol=1e-2):
        raise ValueError(f"Veri doğrulama hatası: Genel churn oranı %20.50 olmalı, bulunan: %{overall_churn_rate:.2f}")

    # 1. Plan bazlı hesaplama ve doğrulama
    plan_expected = {
        "Enterprise": {"total": 104, "churned": 26, "rate": 25.00},
        "Growth": {"total": 669, "churned": 138, "rate": 20.63},
        "Starter": {"total": 955, "churned": 194, "rate": 20.31},
        "Scale": {"total": 272, "churned": 52, "rate": 19.12},
    }
    plan_calc = (
        df.groupby("plan_name")["churn"]
        .agg(total="count", churned="sum", churn_rate=lambda s: s.mean() * 100)
        .to_dict(orient="index")
    )
    for p_name, exp in plan_expected.items():
        if p_name not in plan_calc:
            raise ValueError(f"Plan metrik doğrulama hatası: '{p_name}' planı veride bulunamadı.")
        calc = plan_calc[p_name]
        if calc["total"] != exp["total"] or calc["churned"] != exp["churned"]:
            raise ValueError(
                f"Plan '{p_name}' sayı uyuşmazlığı: Beklenen (n={exp['total']}, churn={exp['churned']}), "
                f"Hesaplanan (n={calc['total']}, churn={calc['churned']})"
            )
        if not math.isclose(calc["churn_rate"], exp["rate"], abs_tol=0.02):
            raise ValueError(
                f"Plan '{p_name}' oran uyuşmazlığı: Beklenen %{exp['rate']:.2f}, Hesaplanan %{calc['churn_rate']:.2f}"
            )

    # 2. Segment bazlı hesaplama ve doğrulama
    segment_expected = {
        "SMB": {"total": 1407, "churned": 303, "rate": 21.54, "tr_name": "SMB"},
        "enterprise": {"total": 121, "churned": 26, "rate": 21.49, "tr_name": "Enterprise"},
        "mid_market": {"total": 472, "churned": 81, "rate": 17.16, "tr_name": "Orta Ölçekli Pazar (Mid-market)"},
    }
    seg_calc = (
        df.groupby("segment")["churn"]
        .agg(total="count", churned="sum", churn_rate=lambda s: s.mean() * 100)
        .to_dict(orient="index")
    )
    for s_name, exp in segment_expected.items():
        if s_name not in seg_calc:
            raise ValueError(f"Segment metrik doğrulama hatası: '{s_name}' segmenti veride bulunamadı.")
        calc = seg_calc[s_name]
        if calc["total"] != exp["total"] or calc["churned"] != exp["churned"]:
            raise ValueError(
                f"Segment '{s_name}' sayı uyuşmazlığı: Beklenen (n={exp['total']}, churn={exp['churned']}), "
                f"Hesaplanan (n={calc['total']}, churn={calc['churned']})"
            )
        if not math.isclose(calc["churn_rate"], exp["rate"], abs_tol=0.02):
            raise ValueError(
                f"Segment '{s_name}' oran uyuşmazlığı: Beklenen %{exp['rate']:.2f}, Hesaplanan %{calc['churn_rate']:.2f}"
            )

    # 3. Kanal bazlı hesaplama ve doğrulama
    channel_expected = {
        "organic": {"total": 385, "churned": 87, "rate": 22.60, "tr_name": "Organik"},
        "outbound": {"total": 389, "churned": 87, "rate": 22.37, "tr_name": "Giden Pazarlama (Outbound)"},
        "paid_search": {"total": 404, "churned": 86, "rate": 21.29, "tr_name": "Ücretli Arama"},
        "referral": {"total": 422, "churned": 78, "rate": 18.48, "tr_name": "Tavsiye"},
        "partner": {"total": 400, "churned": 72, "rate": 18.00, "tr_name": "İş Ortağı"},
    }
    ch_calc = (
        df.groupby("acquisition_channel")["churn"]
        .agg(total="count", churned="sum", churn_rate=lambda s: s.mean() * 100)
        .to_dict(orient="index")
    )
    for c_name, exp in channel_expected.items():
        if c_name not in ch_calc:
            raise ValueError(f"Kanal metrik doğrulama hatası: '{c_name}' kanalı veride bulunamadı.")
        calc = ch_calc[c_name]
        if calc["total"] != exp["total"] or calc["churned"] != exp["churned"]:
            raise ValueError(
                f"Kanal '{c_name}' sayı uyuşmazlığı: Beklenen (n={exp['total']}, churn={exp['churned']}), "
                f"Hesaplanan (n={calc['total']}, churn={calc['churned']})"
            )
        if not math.isclose(calc["churn_rate"], exp["rate"], abs_tol=0.02):
            raise ValueError(
                f"Kanal '{c_name}' oran uyuşmazlığı: Beklenen %{exp['rate']:.2f}, Hesaplanan %{calc['churn_rate']:.2f}"
            )

    # 4. Veri kalitesi kontrolleri
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

    logger.info("Tüm veri bütünlüğü ve metrik doğrulama kontrolleri başarıyla geçildi.")

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
            name: {
                "total": int(data["total"]),
                "churned": int(data["churned"]),
                "churn_rate_pct": round(float(data["churn_rate"]), 2),
            }
            for name, data in plan_calc.items()
        },
        "segment_analysis": {
            segment_expected[name]["tr_name"]: {
                "raw_key": name,
                "total": int(data["total"]),
                "churned": int(data["churned"]),
                "churn_rate_pct": round(float(data["churn_rate"]), 2),
            }
            for name, data in seg_calc.items()
        },
        "channel_analysis": {
            channel_expected[name]["tr_name"]: {
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
    """
    fig_width = 11.693  # 297 mm
    fig_height = 8.268  # 210 mm

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="#F8FAFC")

    # -------------------------------------------------------------------------
    # 1. BÖLÜM: ÜST BAŞLIK ALANI (HEADER)
    # [left, bottom, width, height]
    # -------------------------------------------------------------------------
    ax_header = fig.add_axes([0.040, 0.865, 0.920, 0.100])
    ax_header.axis("off")

    # Sol Başlık Grubu
    ax_header.text(
        0.0, 0.96,
        "CHURN INTELLIGENCE",
        fontsize=18,
        fontweight="bold",
        color="#0F172A",
        va="top",
        ha="left",
    )
    ax_header.text(
        0.0, 0.46,
        "B2B SaaS Müşteri Kaybı Analizi",
        fontsize=11.5,
        fontweight="bold",
        color="#1E3A8A",
        va="top",
        ha="left",
    )
    ax_header.text(
        0.0, 0.08,
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
    ax_kpi = fig.add_axes([0.040, 0.745, 0.920, 0.098])
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
            "sub": f"%{metrics['overall_kpis']['active_pct']:.2f} Sürdüren Müşteri".replace(".", ","),
            "accent": "#16A34A",  # Pozitif Yeşil
        },
        {
            "title": "İPTAL EDİLMİŞ ABONELİK",
            "value": f"{metrics['overall_kpis']['churned_subscriptions']:,}".replace(",", "."),
            "sub": f"%{metrics['overall_kpis']['churn_pct']:.2f} Sözleşmesi Sonlanan".replace(".", ","),
            "accent": "#DC2626",  # Uyarıcı Kırmızı
        },
        {
            "title": "GENEL MÜŞTERİ KAYBI ORANI",
            "value": f"%{metrics['overall_kpis']['churn_pct']:.2f}".replace(".", ","),
            "sub": "Tüm Portföy Kıyas Eşiği",
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
    # Kart 1 Arka Planı (Sol Üst)
    ax_card1 = fig.add_axes([0.040, 0.450, 0.445, 0.270])
    draw_card_background(ax_card1, "Genel Abonelik Durumu Dağılımı")

    # Kart 2 Arka Planı (Sağ Üst)
    ax_card2 = fig.add_axes([0.515, 0.450, 0.445, 0.270])
    draw_card_background(ax_card2, "Abonelik Planlarına Göre Müşteri Kaybı Oranı")

    # Kart 3 Arka Planı (Sol Alt)
    ax_card3 = fig.add_axes([0.040, 0.165, 0.445, 0.260])
    draw_card_background(ax_card3, "Müşteri Segmentlerine Göre Müşteri Kaybı Oranı")

    # Kart 4 Arka Planı (Sağ Alt)
    ax_card4 = fig.add_axes([0.515, 0.165, 0.445, 0.260])
    draw_card_background(ax_card4, "Müşteri Edinme Kanalına Göre Müşteri Kaybı Oranı")

    # --- GRAFİK 1: Genel Abonelik Durumu Dağılımı (Donut) ---
    ax_g1 = fig.add_axes([0.050, 0.460, 0.425, 0.205])
    render_chart_1_donut(ax_g1, metrics)

    # --- GRAFİK 2: Abonelik Planlarına Göre Müşteri Kaybı Oranı (Yatay Çubuk) ---
    ax_g2 = fig.add_axes([0.640, 0.485, 0.295, 0.180])
    render_chart_2_plans(ax_g2, metrics)

    # --- GRAFİK 3: Müşteri Segmentlerine Göre Müşteri Kaybı Oranı (Yatay Çubuk) ---
    ax_g3 = fig.add_axes([0.225, 0.195, 0.240, 0.175])
    render_chart_3_segments(ax_g3, metrics)

    # --- GRAFİK 4: Müşteri Edinme Kanalına Göre Müşteri Kaybı Oranı (Yatay Çubuk) ---
    ax_g4 = fig.add_axes([0.705, 0.195, 0.230, 0.175])
    render_chart_4_channels(ax_g4, metrics)

    # -------------------------------------------------------------------------
    # 4. BÖLÜM: ALT ALAN (VERİ KALİTESİ, ANALİTİK SINIRLILIKLAR VE KAYNAKÇA)
    # -------------------------------------------------------------------------
    ax_footer = fig.add_axes([0.040, 0.015, 0.920, 0.138])
    render_footer_section(ax_footer, metrics)

    return fig


def draw_card_background(ax: plt.Axes, title: str) -> None:
    """Grafikler için kurumsal kart arka planı ve başlık çizer."""
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
        fontsize=9.5,
        fontweight="bold",
        color="#0F172A",
        va="top",
        ha="left",
        transform=ax.transAxes,
    )


# -----------------------------------------------------------------------------
# GRAFİK ÇİZİM YARDIMCILARI
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

    # Donut ortası metni (iki satır, taşma olmadan kusursuz ortalanmış)
    ax.text(
        -0.30, 0.08,
        f"{total_cnt:,}".replace(",", "."),
        ha="center",
        va="center",
        fontsize=13.0,
        fontweight="bold",
        color="#0F172A",
    )
    ax.text(
        -0.30, -0.12,
        "Toplam\nAbonelik",
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        color="#64748B",
        linespacing=1.1,
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


def render_chart_2_plans(ax: plt.Axes, metrics: Dict[str, Any]) -> None:
    """
    Grafik 2: Abonelik Planlarına Göre Müşteri Kaybı Oranı (Yatay Çubuk Grafik)
    Sıralama: Enterprise (%25.00) -> Growth (%20.63) -> Starter (%20.31) -> Scale (%19.12)
    """
    ax.set_facecolor("#FFFFFF")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#CBD5E1")
        ax.spines[spine].set_linewidth(0.8)

    plan_data = []
    for p_name, vals in metrics["plan_analysis"].items():
        plan_data.append({
            "name": p_name,
            "rate": vals["churn_rate_pct"],
            "total": vals["total"],
            "churned": vals["churned"],
            "label": f"{p_name} (n={vals['total']:,})".replace(",", "."),
        })

    plan_data.sort(key=lambda x: x["rate"], reverse=False)

    y_pos = np.arange(len(plan_data))
    rates = [d["rate"] for d in plan_data]
    labels = [d["label"] for d in plan_data]

    bars = ax.barh(y_pos, rates, height=0.52, color="#3B82F6", edgecolor="none", zorder=3)

    for idx, (bar, d) in enumerate(zip(bars, plan_data)):
        val_str = f"%{d['rate']:.2f}".replace(".", ",")
        ax.text(
            d["rate"] + 0.6, idx,
            val_str,
            va="center",
            ha="left",
            fontsize=8.2,
            fontweight="bold",
            color="#1E3A8A",
        )

    # Genel Ortalama Referans Çizgisi (%20,50)
    baseline = metrics["overall_kpis"]["churn_pct"]
    ax.axvline(
        baseline,
        color="#DC2626",
        linestyle="--",
        linewidth=1.2,
        zorder=4,
    )
    # Referans etiketi
    ax.text(
        baseline, 3.60,
        f"Genel Ortalama: %{baseline:.2f}".replace(".", ","),
        color="#DC2626",
        fontsize=6.9,
        fontweight="bold",
        ha="center",
        va="bottom",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.0, fontweight="bold", color="#1E293B")
    ax.set_xlim(0, 31)
    ax.set_ylim(-0.6, 4.0)
    ax.set_xlabel("Müşteri Kaybı Oranı (%)", fontsize=7.5, fontweight="bold", color="#475569", labelpad=2)
    ax.grid(axis="x", linestyle=":", alpha=0.6, color="#E2E8F0", zorder=1)
    ax.set_axisbelow(True)


def render_chart_3_segments(ax: plt.Axes, metrics: Dict[str, Any]) -> None:
    """
    Grafik 3: Müşteri Segmentlerine Göre Müşteri Kaybı Oranı (Yatay Çubuk Grafik)
    Sıralama: SMB (%21.54) -> Enterprise (%21.49) -> Orta Ölçekli Pazar (%17.16)
    """
    ax.set_facecolor("#FFFFFF")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#CBD5E1")
        ax.spines[spine].set_linewidth(0.8)

    seg_data = []
    for s_name, vals in metrics["segment_analysis"].items():
        if "Mid-market" in s_name:
            label_name = f"Orta Ölçekli Pazar (Mid-market)\n(n={vals['total']:,})".replace(",", ".")
        else:
            label_name = f"{s_name} (n={vals['total']:,})".replace(",", ".")

        seg_data.append({
            "name": s_name,
            "rate": vals["churn_rate_pct"],
            "total": vals["total"],
            "churned": vals["churned"],
            "label": label_name,
        })

    seg_data.sort(key=lambda x: x["rate"], reverse=False)

    y_pos = np.arange(len(seg_data))
    rates = [d["rate"] for d in seg_data]
    labels = [d["label"] for d in seg_data]

    bars = ax.barh(y_pos, rates, height=0.48, color="#8B5CF6", edgecolor="none", zorder=3)

    for idx, (bar, d) in enumerate(zip(bars, seg_data)):
        val_str = f"%{d['rate']:.2f}".replace(".", ",")
        ax.text(
            d["rate"] + 0.6, idx,
            val_str,
            va="center",
            ha="left",
            fontsize=8.2,
            fontweight="bold",
            color="#5B21B6",
        )

    baseline = metrics["overall_kpis"]["churn_pct"]
    ax.axvline(
        baseline,
        color="#DC2626",
        linestyle="--",
        linewidth=1.2,
        zorder=4,
    )
    ax.text(
        baseline, 2.60,
        f"Genel Ortalama: %{baseline:.2f}".replace(".", ","),
        color="#DC2626",
        fontsize=6.9,
        fontweight="bold",
        ha="center",
        va="bottom",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7.4, fontweight="bold", color="#1E293B")
    ax.set_xlim(0, 31)
    ax.set_ylim(-0.6, 3.0)
    ax.set_xlabel("Müşteri Kaybı Oranı (%)", fontsize=7.5, fontweight="bold", color="#475569", labelpad=2)
    ax.grid(axis="x", linestyle=":", alpha=0.6, color="#E2E8F0", zorder=1)
    ax.set_axisbelow(True)


def render_chart_4_channels(ax: plt.Axes, metrics: Dict[str, Any]) -> None:
    """
    Grafik 4: Müşteri Edinme Kanalına Göre Müşteri Kaybı Oranı (Yatay Çubuk Grafik)
    Sıralama: Organik (%22.60) -> Giden Pazarlama (%22.37) -> Ücretli Arama (%21.29)
              -> Tavsiye (%18.48) -> İş Ortağı (%18.00)
    """
    ax.set_facecolor("#FFFFFF")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#CBD5E1")
        ax.spines[spine].set_linewidth(0.8)

    ch_data = []
    for c_name, vals in metrics["channel_analysis"].items():
        if "Outbound" in c_name:
            label_name = f"Giden Pazarlama (Outbound)\n(n={vals['total']:,})".replace(",", ".")
        else:
            label_name = f"{c_name} (n={vals['total']:,})".replace(",", ".")

        ch_data.append({
            "name": c_name,
            "rate": vals["churn_rate_pct"],
            "total": vals["total"],
            "churned": vals["churned"],
            "label": label_name,
        })

    ch_data.sort(key=lambda x: x["rate"], reverse=False)

    y_pos = np.arange(len(ch_data))
    rates = [d["rate"] for d in ch_data]
    labels = [d["label"] for d in ch_data]

    bars = ax.barh(y_pos, rates, height=0.55, color="#0EA5E9", edgecolor="none", zorder=3)

    for idx, (bar, d) in enumerate(zip(bars, ch_data)):
        val_str = f"%{d['rate']:.2f}".replace(".", ",")
        ax.text(
            d["rate"] + 0.6, idx,
            val_str,
            va="center",
            ha="left",
            fontsize=8.2,
            fontweight="bold",
            color="#0369A1",
        )

    baseline = metrics["overall_kpis"]["churn_pct"]
    ax.axvline(
        baseline,
        color="#DC2626",
        linestyle="--",
        linewidth=1.2,
        zorder=4,
    )
    ax.text(
        baseline, 4.60,
        f"Genel Ortalama: %{baseline:.2f}".replace(".", ","),
        color="#DC2626",
        fontsize=6.9,
        fontweight="bold",
        ha="center",
        va="bottom",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7.4, fontweight="bold", color="#1E293B")
    ax.set_xlim(0, 31)
    ax.set_ylim(-0.6, 5.0)
    ax.set_xlabel("Müşteri Kaybı Oranı (%)", fontsize=7.5, fontweight="bold", color="#475569", labelpad=2)
    ax.grid(axis="x", linestyle=":", alpha=0.6, color="#E2E8F0", zorder=1)
    ax.set_axisbelow(True)


def render_footer_section(ax: plt.Axes, metrics: Dict[str, Any]) -> None:
    """
    Alt Bölüm: Veri Kalitesi, Analitik Sınırlılıklar ve Kaynakça
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
        0.018, 0.92,
        "VERİ KALİTESİ VE ANALİTİK SINIRLILIKLAR",
        fontsize=8.2,
        fontweight="bold",
        color="#9F1239",
        va="top",
        ha="left",
        transform=ax.transAxes,
    )

    limitations_text = (
        "1. Coğrafi bölge bilgisi 2.000 gözlemin 479 tanesinde eksiktir (%23,95). Bu değerler rastgele doldurulmamış, eksik olarak korunmuştur.\n"
        "2. Fatura geçmişi yalnızca 2.000 aboneliğin 120 tanesinde mevcuttur (%6,00).\n"
        "3. Ödeme geçmişi 2.000 aboneliğin 118 tanesinde mevcuttur (%5,90).\n\n"
        "Önemli Metodolojik Uyarı: Özellikle fatura ve ödeme verilerine dayanan bulgular yalnızca sınırlı bir alt grubu (%6)\n"
        "temsil etmekte olup tüm müşteri popülasyonuna genellenemez. Verinin doğrudan desteklemediği neden-sonuç\n"
        "ilişkileri kurulmamalıdır."
    )
    ax.text(
        0.018, 0.73,
        limitations_text,
        fontsize=6.7,
        color="#334155",
        va="top",
        ha="left",
        linespacing=1.24,
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
        0.652, 0.92,
        "KAYNAK GÖSTERİMİ VE AKADEMİK BİLDİRİM",
        fontsize=8.2,
        fontweight="bold",
        color="#1E3A8A",
        va="top",
        ha="left",
        transform=ax.transAxes,
    )

    provenance_text = (
        f"Veri Seti: {DATA_SOURCE_NAME}\n"
        f"Bağlantı: {DATA_SOURCE_URL}\n"
        f"Lisans: {DATA_LICENSE}\n\n"
        f"Veri Kapsamı Beyanı: 2023–2025 veri kapsamı kaynak tarafından\n"
        f"beyan edilmiş olup, bağımsız olarak doğrulanmamıştır.\n"
        f"Hazırlayan: {STUDENT_NAME}  |  Öğrenci No: {STUDENT_ID}"
    )
    ax.text(
        0.652, 0.73,
        provenance_text,
        fontsize=6.6,
        color="#334155",
        va="top",
        ha="left",
        linespacing=1.21,
        transform=ax.transAxes,
    )


# -----------------------------------------------------------------------------
# ANA ÇALIŞTIRMA VE ÇIKTI ÜRETİM FONKSİYONU
# -----------------------------------------------------------------------------
def build_vize_dashboard() -> Tuple[Path, Path, Path]:
    """
    Tüm dashboard üretim sürecini koordine eder:
    Veriyi yükler, doğrular, grafikleri çizer ve çıktı dosyalarını kaydeder.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Veri yükleme ve doğrulama
    df, metrics = load_and_validate_data(DATASET_PATH)

    # 2. JSON metrik raporunu kaydet
    logger.info("Doğrulanmış metrikler JSON formatında kaydediliyor: %s", OUTPUT_JSON_PATH)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # 3. Dashboard figürünü oluştur
    logger.info("A4 yatay profesyonel dashboard figürü çiziliyor...")
    fig = build_full_dashboard(df, metrics)

    # 4. PDF ve PNG olarak kaydet
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
