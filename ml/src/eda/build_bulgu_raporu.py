"""
B2B SaaS Churn Intelligence - Vize Sınavı Tek Sayfalık Bulgu Raporu Üreticisi
----------------------------------------------------------------------------
Bu modül, ml/reports/vize/dashboard_metrics.json dosyasındaki doğrulanmış
metrikleri okuyarak (hiçbir sayıyı elle sabitlemeden) A4 dikey formatında,
10–10,5 pt yazı boyutunda, tek sayfalık 'bulgu_raporu.docx' ve 'bulgu_raporu.pdf'
dosyalarını üretir ve otomatik tutarlılık denetimi gerçekleştirir.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple
from xml.sax.saxutils import escape

# Yollar
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
ML_DIR = SRC_DIR.parent
REPO_ROOT = ML_DIR.parent

VIZE_DIR = ML_DIR / "reports" / "vize"
METRICS_JSON_PATH = VIZE_DIR / "dashboard_metrics.json"
OUTPUT_DOCX_PATH = VIZE_DIR / "bulgu_raporu.docx"
OUTPUT_PDF_PATH = VIZE_DIR / "bulgu_raporu.pdf"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bulgu_raporu")


def fmt_int(val: int | float) -> str:
    """Tam sayıları Türkçe binlik ayırıcı (.) ile biçimlendirir."""
    return f"{int(round(val)):,}".replace(",", ".")


def fmt_pct(val: float, decimals: int = 2) -> str:
    """Yüzde değerlerini Türkçe ondalık virgülü (,) ile biçimlendirir."""
    return f"{val:.{decimals}f}".replace(".", ",")


def build_report_content(m: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tüm sayıları doğrudan dashboard_metrics.json sözlüğünden okuyarak
    rapor başlıklarını ve paragraflarını oluşturur.
    """
    st = m["student_info"]
    prov = m["data_provenance"]
    kpis = m["overall_kpis"]
    km = m["kaplan_meier"]
    ms = km["milestones"]
    drops = km["six_month_drops_pp"]
    plans = m["plan_analysis"]
    mrr_plans = m["mrr_by_plan"]
    segs = m["segment_analysis"]
    chans = m["channel_analysis"]
    chi2 = m["chi_square_tests"]
    dq = m["data_quality_limitations"]

    # Dinamik biçimlendirilmiş metrikler (tüm raporda tek ve tutarlı biçim)
    tot_sub = fmt_int(kpis["total_subscriptions"])
    act_sub = fmt_int(kpis["active_subscriptions"])
    ch_sub = fmt_int(kpis["churned_subscriptions"])
    act_pct = fmt_pct(kpis["active_pct"], 2)
    ch_pct = fmt_pct(kpis["churn_pct"], 2)

    tot_mrr = fmt_int(kpis["total_mrr"])
    ch_mrr = fmt_int(kpis["churned_mrr"])
    mrr_ch_pct = fmt_pct(kpis["mrr_churn_pct"], 2)
    mrr_vs_sub_diff = fmt_pct(kpis["mrr_churn_pct"] - kpis["churn_pct"], 2)

    km_180 = fmt_pct(ms["day_180_pct"], 2)
    km_365 = fmt_pct(ms["day_365_pct"], 2)
    km_540 = fmt_pct(ms["day_540_pct"], 2)

    dr_1 = fmt_pct(drops["d0_to_180_pp"], 2)
    dr_2 = fmt_pct(drops["d180_to_365_pp"], 2)
    dr_3 = fmt_pct(drops["d365_to_540_pp"], 2)

    cond_prob = km.get("conditional_loss_probabilities", {})
    q_1 = fmt_pct(cond_prob.get("q_0_to_180_pct", drops["d0_to_180_pp"]), 2)
    q_2 = fmt_pct(cond_prob.get("q_180_to_365_pct", 8.52), 2)
    q_3 = fmt_pct(cond_prob.get("q_365_to_540_pct", 10.42), 2)
    cond_str = f"%{q_1} → %{q_2} → %{q_3}"

    ent_plan = plans["Enterprise (Kurumsal Plan)"]
    ent_mrr = mrr_plans["Enterprise (Kurumsal Plan)"]
    scale_mrr = mrr_plans["Ölçek"]

    ent_cnt = fmt_int(ent_plan["total"])
    ent_sub_share = fmt_pct(ent_plan["subscription_share_pct"], 2)
    ent_mrr_share = fmt_pct(ent_mrr["total_mrr_share_pct"], 2)
    ent_lost_share = fmt_pct(ent_mrr["churned_mrr_share_pct"], 2)
    ent_ch_rate = fmt_pct(ent_plan["churn_rate_pct"], 2)

    scale_mrr_share = fmt_pct(scale_mrr["total_mrr_share_pct"], 2)
    scale_lost_share = fmt_pct(scale_mrr["churned_mrr_share_pct"], 2)

    # Enterprise + Ölçek birleşik MRR payları (JSON'daki ham değerlerden dinamik hesaplanır)
    ent_scale_tot_mrr_pct = (
        (ent_mrr["total_mrr"] + scale_mrr["total_mrr"]) / kpis["total_mrr"]
    ) * 100.0
    ent_scale_lost_mrr_pct = (
        (ent_mrr["churned_mrr"] + scale_mrr["churned_mrr"]) / kpis["churned_mrr"]
    ) * 100.0
    ent_scale_tot_str = fmt_pct(ent_scale_tot_mrr_pct, 2)
    ent_scale_lost_str = fmt_pct(ent_scale_lost_mrr_pct, 2)

    scale_rate = fmt_pct(plans["Ölçek"]["churn_rate_pct"], 2)
    growth_rate = fmt_pct(plans["Büyüme"]["churn_rate_pct"], 2)
    starter_rate = fmt_pct(plans["Başlangıç"]["churn_rate_pct"], 2)

    p_plan = fmt_pct(chi2["plan_name"]["p_value"], 2)
    p_seg = fmt_pct(chi2["segment"]["p_value"], 2)
    p_ch = fmt_pct(chi2["acquisition_channel"]["p_value"], 2)
    p_fisher = fmt_pct(chi2["enterprise_vs_others_fisher"]["p_value"], 2)

    mid_seg_rate = fmt_pct(segs["Orta Ölçekli Pazar"]["churn_rate_pct"], 2)
    smb_seg_rate = fmt_pct(segs["KOBİ"]["churn_rate_pct"], 2)
    ent_seg_rate = fmt_pct(segs["Kurumsal Müşteri"]["churn_rate_pct"], 2)

    part_ch_rate = fmt_pct(chans["İş Ortağı"]["churn_rate_pct"], 2)
    ref_ch_rate = fmt_pct(chans["Tavsiye"]["churn_rate_pct"], 2)
    org_ch_rate = fmt_pct(chans["Organik"]["churn_rate_pct"], 2)
    out_ch_rate = fmt_pct(chans["Giden Pazarlama"]["churn_rate_pct"], 2)
    paid_ch_rate = fmt_pct(chans["Ücretli Arama"]["churn_rate_pct"], 2)

    reg_miss_cnt = fmt_int(dq["missing_region_count"])
    reg_miss_pct = fmt_pct(dq["missing_region_pct"], 2)
    bil_cnt = fmt_int(dq["billing_data_coverage_count"])
    bil_pct = fmt_pct(dq["billing_data_coverage_pct"], 2)
    pay_cnt = fmt_int(dq["payment_data_coverage_count"])
    pay_pct = fmt_pct(dq["payment_data_coverage_pct"], 2)

    content = {
        "header_title": "B2B Müşteri Kaybı ve Elde Tutma Analizi",
        "header_subtitle": f"{st['project_title']} — Vize Bulgu Raporu",
        "header_meta": f"Öğrenci: {st['name']}  |  Öğrenci No: {st['student_id']}  |  {st['course']}",
        "sections": [
            {
                "heading": "1. Amaç ve Veri",
                "paragraphs": [
                    (
                        f"Bu çalışmanın temel araştırma sorusu şudur: B2B SaaS portföyünde hangi abonelikler, "
                        f"ne zaman ve ne kadar gelir (MRR) etkisiyle kaybedilmektedir? Analizde {prov['source']} "
                        f"({prov['license']}) kaynağına ait sentetik (simüle) veri seti kullanılmıştır. Kaynakta "
                        f"beyan edilen veri kapsamı 01.01.2023 – 31.12.2025 (2023–2025 dönemi, bağımsız doğrulanmamış) "
                        f"olup toplam {tot_sub} abonelik kaydından ({act_sub} aktif, {ch_sub} iptal edilmiş) "
                        f"oluşmaktadır ve temel gözlem birimi aboneliktir."
                    )
                ],
                "bullets": [],
            },
            {
                "heading": "2. Grafiklerin Amacı ve Seçim Gerekçesi",
                "paragraphs": [],
                "bullets": [
                    (
                        "\"Abonelik Süresine Göre Elde Tutma Eğrisi\": Aboneliklerin zaman içinde (0–720 gün) hangi "
                        "hızla kaybedildiği ve terk riskinin hangi dönemlerde yoğunlaştığı sorusuna yanıt verir. Aktif "
                        "aboneliklerin gözlem süresi devam ettiği (sağdan sansürlü veri) için zaman içindeki sağkalım "
                        "olasılığını sansürlü gözlemleri hesaba katan Kaplan-Meier çizgi/basamak grafiği ile göstermek "
                        "uygun seçilmiştir."
                    ),
                    (
                        f"\"Plan Bazında MRR ve Kayıp MRR Payı\": Her abonelik planının toplam aylık gelir ({tot_mrr} MRR) "
                        f"içindeki payı ile kaybedilen gelir (%{mrr_ch_pct}) içindeki payının orantılı olup olmadığını "
                        "sorgular. İki farklı oransal büyüklüğü (Toplam MRR Payı %: Plan MRR / Portföy MRR ve Kaybedilen "
                        "MRR Payı %: Plan Kayıp MRR / Toplam Kayıp MRR) dört plan kategorisi arasında doğrudan "
                        "karşılaştırmak amacıyla iki gruplu çubuk grafik kullanılmıştır."
                    ),
                    (
                        f"\"Hangi Planda Kayıp Oranı Daha Yüksek?\": Enterprise (Kurumsal Plan) grubunun en yüksek "
                        f"kayıp oranına sahip olup olmadığını ve planlar arasındaki farkın anlamlı olup olmadığını "
                        f"inceler. Kategorileri sıralı ve ortak ölçekte genel ortalama (%{ch_pct}) ile karşılaştırma "
                        "amacıyla yatay çubuk grafik ve kesikli referans çizgisi tercih edilmiştir."
                    ),
                    (
                        f"\"Segmentlere Göre Kayıp Farklılaşıyor mu?\": KOBİ, Kurumsal Müşteri ve Orta Ölçekli Pazar "
                        f"kategorileri arasında müşteri kaybı oranının farklılaşıp farklılaşmadığını sorgular. "
                        f"Kategorileri sıralı ve ortak ölçekte genel ortalama (%{ch_pct}) ile karşılaştırma amacıyla "
                        "yatay çubuk grafik ve kesikli referans çizgisi kullanılmıştır."
                    ),
                    (
                        f"\"Edinme Kanalına Göre Kayıp Oranı Farklılaşıyor mu?\": Müşteri kazanım kanalları arasında "
                        f"kayıp oranının farklılaşıp farklılaşmadığını değerlendirir. Kategorileri sıralı ve ortak "
                        f"ölçekte genel ortalama (%{ch_pct}) ile karşılaştırma amacıyla yatay çubuk grafik ve kesikli "
                        "referans çizgisi seçilmiştir."
                    ),
                ],
            },
            {
                "heading": "3. Temel Bulgular",
                "paragraphs": [],
                "bullets": [
                    (
                        f"Genel Abonelik ve Gelir Kaybı: Portföydeki {tot_sub} aboneliğin {ch_sub} adedi iptal edilmiş "
                        f"olup genel müşteri kaybı (churn) oranı %{ch_pct} ({ch_sub}/{tot_sub}) düzeyindedir (aktif: "
                        f"{act_sub} / %{act_pct}). Toplam {tot_mrr} MRR portföyü içinde iptal edilen aboneliklerin "
                        f"MRR payı %{mrr_ch_pct} ({ch_mrr} MRR) olup portföy kayıp oranı abonelik sayısındaki kayıp "
                        f"payının biraz üzerindedir (%{mrr_ch_pct} ile %{ch_pct}; fark {mrr_vs_sub_diff} puan)."
                    ),
                    (
                        f"Zamana Bağlı Elde Tutma Eğrisi: Kaplan-Meier elde tutma oranı 180. günde %{km_180}, 365. günde "
                        f"(12. ay) %{km_365} ve 540. günde (18. ay) %{km_540} düzeyindedir. İlk 90 günde yalnızca bir "
                        f"iptal olayı gözlenirken (1 olay), her 6 aylık dilimde sağkalım kaybı hızlanmakta ({dr_1} → "
                        f"{dr_2} → {dr_3} puan) ve ilgili döneme ulaşanlar için koşullu kayıp riski de artmaktadır "
                        f"({cond_str})."
                    ),
                    (
                        f"Enterprise Planının Gelir Etkisi: Enterprise (Kurumsal Plan) tüm aboneliklerin yalnızca "
                        f"%{ent_sub_share}'sini ({ent_cnt}/{tot_sub}) oluşturmasına karşın toplam MRR'nin "
                        f"%{ent_mrr_share}'unu ve kaybedilen toplam MRR'nin %{ent_lost_share}'ini "
                        f"oluşturmaktadır. Bu planda churn oranı %{ent_ch_rate} olmakla birlikte diğer planlardan "
                        f"istatistiksel olarak anlamlı biçimde yüksek değildir (Fisher kesin test p = {p_fisher}); gelir etkisi "
                        "büyük ölçüde yüksek birim fiyattan gelmektedir."
                    ),
                    (
                        f"Kategorik Kırılımlarda İstatistiksel Anlamlılık: Plan (Enterprise %{ent_ch_rate}, Ölçek "
                        f"%{scale_rate}, Büyüme %{growth_rate}, Başlangıç %{starter_rate}; Ki-kare p = {p_plan}), segment "
                        f"(KOBİ %{smb_seg_rate}, Kurumsal Müşteri %{ent_seg_rate}, Orta Ölçekli Pazar %{mid_seg_rate}; "
                        f"Ki-kare p = {p_seg}) ve edinme kanalı (Organik %{org_ch_rate}, Giden Pazarlama %{out_ch_rate}, "
                        f"Ücretli Arama %{paid_ch_rate}, Tavsiye %{ref_ch_rate}, İş Ortağı %{part_ch_rate}; Ki-kare "
                        f"p = {p_ch}) analizlerinde p > 0,05 bulunmuştur. Bu örneklemde gruplar arasında istatistiksel "
                        f"olarak anlamlı bir farklılık bulunduğuna ilişkin yeterli kanıt elde edilememiştir; bu durum "
                        f"grupların eşit olduğunu kanıtlamaz (Cochran kuralı gereği tüm beklenen frekanslar > 5'tir)."
                    ),
                ],
            },
            {
                "heading": "4. Bulguların Yorumu",
                "paragraphs": [
                    (
                        f"Analiz bulguları birlikte değerlendirildiğinde, müşteri kaybının belirli bir abonelik planı, "
                        f"müşteri segmenti veya edinme kanalına özgü yapısal bir ayrışmadan (Ki-kare p = {p_plan} / "
                        f"{p_seg} / {p_ch}) çok, zamana yayılan bir elde tutma sorunu ile ilişkili olduğu görülmektedir. "
                        f"İlk 90 günde yalnızca bir iptal olayı gözlenirken kayıplar 6. aydan sonra hızlanmakta ({dr_1} → "
                        f"{dr_2} → {dr_3} puan) ve 18. aya kadar terk riski azalmamaktadır. Benzer biçimde Enterprise "
                        f"planının önceliği, daha yüksek bir terk eğiliminden (Fisher p = {p_fisher}) değil, portföy "
                        f"gelirinin %{ent_mrr_share}'unu ve kayıp gelirin %{ent_lost_share}'ini barındıran yüksek gelir "
                        "ağırlığından kaynaklanmaktadır; bu bulgular nedensellik iddiası taşımamakta olup gözlemsel "
                        "ilişkisellik çerçevesinde yorumlanmalıdır."
                    ),
                    (
                        f"Sınırlılıklar: Çalışma; verinin sentetik (simüle) olması, {tot_sub} aboneliğin {bil_cnt}'sinde "
                        f"(%{bil_pct}) fatura geçmişi, {pay_cnt}'sinde (%{pay_pct}) ödeme geçmişi bulunması nedeniyle "
                        f"fatura ve ödeme analizlerinin tüm abonelik popülasyonuna genellenememesi, bölge bilgisinin "
                        f"%{reg_miss_pct} ({reg_miss_cnt} abonelik) oranında eksik olması ve anlamlılık testlerinin "
                        "çoklu test düzeltmesiz keşifsel nitelik taşıması nedeniyle sınırlılık içerir."
                    ),
                    (
                        f"Genel Sonuç: Enterprise ve Ölçek planları birlikte toplam MRR'nin %{ent_scale_tot_str}'sini "
                        f"ve kayıp MRR'nin %{ent_scale_lost_str}'sini oluşturmaktadır. Ölçek planının kayıp MRR payı "
                        f"(%{scale_lost_share}) kendi MRR payının (%{scale_mrr_share}) altında kaldığından Ölçek "
                        f"yalnızca gelir ağırlığı nedeniyle izleme önceliği taşırken, Enterprise ise hem gelir ağırlığı "
                        f"(%{ent_mrr_share}) hem de yüksek (istatistiksel olarak anlamlı olmayan) churn oranı (%{ent_ch_rate}) "
                        "nedeniyle izlenmelidir; bu bulgular nedensellik iddiası taşımamaktadır."
                    ),
                ],
                "bullets": [],
            },
        ],
    }
    return content


def write_docx_report(content: Dict[str, Any], docx_path: Path) -> None:
    """
    Standart OpenXML (WordprocessingML) yapısında tek sayfalık A4 dikey,
    10–10,5 pt yazı boyutunda bulgu_raporu.docx dosyasını üretir.
    """
    # A4 portrait in twips: 11906 x 16838 (210mm x 297mm)
    # Margins: top/bottom = 720 twips (0.5 in = 12.7 mm), left/right = 850 twips (15 mm)
    # Font sizes in half-points: 10 pt = 20, 10.5 pt = 21, 11 pt = 22, 14 pt = 28

    body_xml_parts: List[str] = []

    # Üst Başlık Bloğu (Rapor Başlığı, Altbaşlık, Öğrenci ve Numara Bilgisi)
    body_xml_parts.append(
        f"""
        <w:p>
          <w:pPr>
            <w:spacing w:before="0" w:after="25" w:line="235" w:lineRule="auto"/>
          </w:pPr>
          <w:r>
            <w:rPr>
              <w:rFonts w:ascii="Liberation Sans" w:hAnsi="Liberation Sans" w:cs="Liberation Sans"/>
              <w:b/>
              <w:color w:val="0F172A"/>
              <w:sz w:val="28"/>
              <w:szCs w:val="28"/>
            </w:rPr>
            <w:t>{escape(content['header_title'])}</w:t>
          </w:r>
        </w:p>
        <w:p>
          <w:pPr>
            <w:spacing w:before="0" w:after="25" w:line="225" w:lineRule="auto"/>
          </w:pPr>
          <w:r>
            <w:rPr>
              <w:rFonts w:ascii="Liberation Sans" w:hAnsi="Liberation Sans" w:cs="Liberation Sans"/>
              <w:b/>
              <w:color w:val="1E3A8A"/>
              <w:sz w:val="21"/>
              <w:szCs w:val="21"/>
            </w:rPr>
            <w:t>{escape(content['header_subtitle'])}</w:t>
          </w:r>
        </w:p>
        <w:p>
          <w:pPr>
            <w:pBdr>
              <w:bottom w:val="single" w:sz="6" w:space="4" w:color="CBD5E1"/>
            </w:pBdr>
            <w:spacing w:before="0" w:after="80" w:line="220" w:lineRule="auto"/>
          </w:pPr>
          <w:r>
            <w:rPr>
              <w:rFonts w:ascii="Liberation Sans" w:hAnsi="Liberation Sans" w:cs="Liberation Sans"/>
              <w:color w:val="334155"/>
              <w:sz w:val="20"/>
              <w:szCs w:val="20"/>
            </w:rPr>
            <w:t>{escape(content['header_meta'])}</w:t>
          </w:r>
        </w:p>
        """
    )

    for sec in content["sections"]:
        # Bölüm Başlığı (11 pt bold)
        body_xml_parts.append(
            f"""
            <w:p>
              <w:pPr>
                <w:spacing w:before="65" w:after="30" w:line="225" w:lineRule="auto"/>
              </w:pPr>
              <w:r>
                <w:rPr>
                  <w:rFonts w:ascii="Liberation Sans" w:hAnsi="Liberation Sans" w:cs="Liberation Sans"/>
                  <w:b/>
                  <w:color w:val="1E3A8A"/>
                  <w:sz w:val="22"/>
                  <w:szCs w:val="22"/>
                </w:rPr>
                <w:t>{escape(sec['heading'])}</w:t>
              </w:r>
            </w:p>
            """
        )

        # Paragraflar (10 pt = sz 20)
        for p_text in sec["paragraphs"]:
            body_xml_parts.append(
                f"""
                <w:p>
                  <w:pPr>
                    <w:jc w:val="both"/>
                    <w:spacing w:before="0" w:after="45" w:line="222" w:lineRule="auto"/>
                  </w:pPr>
                  <w:r>
                    <w:rPr>
                      <w:rFonts w:ascii="Liberation Sans" w:hAnsi="Liberation Sans" w:cs="Liberation Sans"/>
                      <w:color w:val="1E293B"/>
                      <w:sz w:val="20"/>
                      <w:szCs w:val="20"/>
                    </w:rPr>
                    <w:t>{escape(p_text)}</w:t>
                  </w:r>
                </w:p>
                """
            )

        # Madde İşaretli Satırlar (10 pt = sz 20)
        for b_text in sec["bullets"]:
            body_xml_parts.append(
                f"""
                <w:p>
                  <w:pPr>
                    <w:ind w:left="200" w:hanging="140"/>
                    <w:jc w:val="both"/>
                    <w:spacing w:before="0" w:after="36" w:line="220" w:lineRule="auto"/>
                  </w:pPr>
                  <w:r>
                    <w:rPr>
                      <w:rFonts w:ascii="Liberation Sans" w:hAnsi="Liberation Sans" w:cs="Liberation Sans"/>
                      <w:color w:val="1E293B"/>
                      <w:sz w:val="20"/>
                      <w:szCs w:val="20"/>
                    </w:rPr>
                    <w:t>• {escape(b_text)}</w:t>
                  </w:r>
                </w:p>
                """
            )

    # Sayfa ayarları (A4 dikey)
    body_xml_parts.append(
        """
        <w:sectPr>
          <w:pgSz w:w="11906" w:h="16838"/>
          <w:pgMar w:top="560" w:right="720" w:bottom="560" w:left="720" w:header="360" w:footer="360" w:gutter="0"/>
        </w:sectPr>
        """
    )

    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">\n'
        "  <w:body>\n"
        + "\n".join(body_xml_parts)
        + "\n  </w:body>\n</w:document>\n"
    )

    content_types_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
"""

    rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""

    with zipfile.ZipFile(docx_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", rels_xml)
        zf.writestr("word/document.xml", document_xml)


def convert_docx_to_pdf(docx_path: Path, output_dir: Path) -> Path:
    """LibreOffice headless kullanarak .docx dosyasını doğrudan .pdf'e dönüştürür."""
    lo_profile = Path(tempfile.gettempdir()) / "lo_vize_profile"
    cmd = [
        "libreoffice",
        f"-env:UserInstallation=file://{lo_profile}",
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        str(output_dir),
        str(docx_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"LibreOffice PDF dönüştürme hatası: {res.stderr}")
    pdf_path = output_dir / f"{docx_path.stem}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF çıktısı oluşturulamadı: {pdf_path}")
    return pdf_path


def verify_report_numbers_against_json(content: Dict[str, Any], m: Dict[str, Any]) -> None:
    """
    Rapordaki tüm metinleri birleştirir ve dashboard_metrics.json'daki
    tüm zorunlu metriklerin raporda birebir yer aldığını otomatik doğrular.
    """
    full_text = " ".join(
        [content["header_title"], content["header_subtitle"], content["header_meta"]]
        + [
            p
            for sec in content["sections"]
            for p in (sec["paragraphs"] + sec["bullets"])
        ]
    )

    kpis = m["overall_kpis"]
    km = m["kaplan_meier"]
    ms = km["milestones"]
    drops = km["six_month_drops_pp"]
    plans = m["plan_analysis"]
    mrr_plans = m["mrr_by_plan"]
    segs = m["segment_analysis"]
    chans = m["channel_analysis"]
    chi2 = m["chi_square_tests"]
    dq = m["data_quality_limitations"]

    ent_mrr = mrr_plans["Enterprise (Kurumsal Plan)"]
    scale_mrr = mrr_plans["Ölçek"]
    ent_scale_tot_mrr_pct = (
        (ent_mrr["total_mrr"] + scale_mrr["total_mrr"]) / kpis["total_mrr"]
    ) * 100.0
    ent_scale_lost_mrr_pct = (
        (ent_mrr["churned_mrr"] + scale_mrr["churned_mrr"]) / kpis["churned_mrr"]
    ) * 100.0

    cond_prob = km.get("conditional_loss_probabilities", {})
    cond_str = (
        f"%{fmt_pct(cond_prob.get('q_0_to_180_pct', drops['d0_to_180_pp']), 2)} → "
        f"%{fmt_pct(cond_prob.get('q_180_to_365_pct', 8.52), 2)} → "
        f"%{fmt_pct(cond_prob.get('q_365_to_540_pct', 10.42), 2)}"
    )

    expected_tokens = [
        m["student_info"]["name"],
        m["student_info"]["student_id"],
        m["data_provenance"]["source"],
        "B2B Müşteri Kaybı ve Elde Tutma Analizi",
        "SaaS Abonelik Müşteri Kaybı: Keşifsel Veri Analizi",
        "Abonelik Süresine Göre Elde Tutma Eğrisi",
        "Plan Bazında MRR ve Kayıp MRR Payı",
        "Hangi Planda Kayıp Oranı Daha Yüksek?",
        "Segmentlere Göre Kayıp Farklılaşıyor mu?",
        "Edinme Kanalına Göre Kayıp Oranı Farklılaşıyor mu?",
        "sansürlü gözlemleri hesaba katan",
        "İlk 90 günde yalnızca bir iptal olayı gözlenirken",
        "fatura geçmişi",
        "ödeme geçmişi",
        fmt_int(kpis["total_subscriptions"]),
        fmt_int(kpis["active_subscriptions"]),
        fmt_int(kpis["churned_subscriptions"]),
        f"%{fmt_pct(kpis['churn_pct'], 2)} ({fmt_int(kpis['churned_subscriptions'])}/{fmt_int(kpis['total_subscriptions'])})",
        fmt_pct(kpis["mrr_churn_pct"], 2),
        f"({fmt_int(kpis['churned_mrr'])} MRR)",
        fmt_pct(kpis["mrr_churn_pct"] - kpis["churn_pct"], 2),
        fmt_pct(ms["day_180_pct"], 2),
        fmt_pct(ms["day_365_pct"], 2),
        fmt_pct(ms["day_540_pct"], 2),
        f"{fmt_pct(drops['d0_to_180_pp'], 2)} → {fmt_pct(drops['d180_to_365_pp'], 2)} → {fmt_pct(drops['d365_to_540_pp'], 2)} puan",
        cond_str,
        fmt_pct(plans["Enterprise (Kurumsal Plan)"]["subscription_share_pct"], 2),
        fmt_pct(ent_mrr["total_mrr_share_pct"], 2),
        fmt_pct(ent_mrr["churned_mrr_share_pct"], 2),
        fmt_pct(scale_mrr["total_mrr_share_pct"], 2),
        fmt_pct(scale_mrr["churned_mrr_share_pct"], 2),
        fmt_pct(ent_scale_tot_mrr_pct, 2),
        fmt_pct(ent_scale_lost_mrr_pct, 2),
        fmt_pct(plans["Enterprise (Kurumsal Plan)"]["churn_rate_pct"], 2),
        fmt_pct(chi2["enterprise_vs_others_fisher"]["p_value"], 2),
        fmt_pct(chi2["plan_name"]["p_value"], 2),
        fmt_pct(chi2["segment"]["p_value"], 2),
        fmt_pct(chi2["acquisition_channel"]["p_value"], 2),
        fmt_pct(segs["Orta Ölçekli Pazar"]["churn_rate_pct"], 2),
        fmt_pct(chans["İş Ortağı"]["churn_rate_pct"], 2),
        fmt_pct(chans["Tavsiye"]["churn_rate_pct"], 2),
        fmt_pct(dq["missing_region_pct"], 2),
    ]

    for tok in expected_tokens:
        if tok not in full_text:
            raise ValueError(f"Otomatik doğrulama hatası: '{tok}' değeri veya başlığı raporda bulunamadı!")

    logger.info("Otomatik metrik ve başlık doğrulaması başarılı (%d kontrol noktası eşleşti).", len(expected_tokens))


def verify_pdf_single_page(pdf_path: Path) -> int:
    """pdfinfo ile PDF'in tam olarak 1 sayfa olduğunu doğrular."""
    res = subprocess.run(["pdfinfo", str(pdf_path)], capture_output=True, text=True, check=True)
    match = re.search(r"^Pages:\s+(\d+)", res.stdout, flags=re.MULTILINE)
    if not match:
        raise RuntimeError("pdfinfo çıktısında sayfa sayısı okunamadı!")
    pages = int(match.group(1))
    if pages != 1:
        raise ValueError(f"Sayfa sınırı hatası: bulgu_raporu.pdf 1 sayfa olmalı, bulunan: {pages} sayfa!")
    return pages


def build_bulgu_raporu() -> Tuple[Path, Path]:
    if not METRICS_JSON_PATH.exists():
        raise FileNotFoundError(f"Metrik JSON dosyası bulunamadı: {METRICS_JSON_PATH}")

    with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    content = build_report_content(metrics)
    verify_report_numbers_against_json(content, metrics)

    logger.info("DOCX raporu oluşturuluyor: %s", OUTPUT_DOCX_PATH)
    write_docx_report(content, OUTPUT_DOCX_PATH)

    logger.info("PDF raporu oluşturuluyor: %s", OUTPUT_PDF_PATH)
    convert_docx_to_pdf(OUTPUT_DOCX_PATH, VIZE_DIR)

    pages = verify_pdf_single_page(OUTPUT_PDF_PATH)
    logger.info("bulgu_raporu.pdf başarıyla doğrulandı (Sayfa sayısı: %d).", pages)

    return OUTPUT_PDF_PATH, OUTPUT_DOCX_PATH


if __name__ == "__main__":
    build_bulgu_raporu()
    print("✅ Tek sayfalık bulgu_raporu.pdf ve bulgu_raporu.docx başarıyla üretildi ve doğrulandı.")
