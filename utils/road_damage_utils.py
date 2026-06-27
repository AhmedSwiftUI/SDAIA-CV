"""
أدوات نظام رصد تلف الطريق — قاعدة بيانات المواقع، رسم الكشف، إنشاء الخريطة
"""

import json
import math
import uuid
from pathlib import Path
from typing import Optional

import arabic_reshaper
import cv2
import folium
import numpy as np
from bidi.algorithm import get_display
from folium.plugins import MiniMap
from PIL import Image, ImageDraw, ImageFont

import config

# خط عربي من النظام
_ARABIC_FONT_PATH = "/System/Library/Fonts/SFArabic.ttf"


def _pil_text_ar(img_bgr, text: str, pos: tuple, font_size: int = 22,
                 bg_color=(30, 30, 220), text_color=(255, 255, 255)) -> None:
    """ارسم نصاً عربياً على صورة OpenCV BGR باستخدام PIL (in-place)."""
    reshaped = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped)

    pil_img  = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw     = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype(_ARABIC_FONT_PATH, font_size)
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), bidi_text, font=font)
    tw   = bbox[2] - bbox[0]
    th   = bbox[3] - bbox[1]

    x, y = pos
    # خلفية مستطيلة
    draw.rectangle([x, y - th - 8, x + tw + 10, y + 2],
                   fill=tuple(reversed(bg_color)))   # BGR→RGB
    draw.text((x + 5, y - th - 4), bidi_text, font=font,
              fill=tuple(reversed(text_color)))      # BGR→RGB

    # نسخ النتيجة back to img_bgr
    result_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    np.copyto(img_bgr, result_bgr)

DAMAGE_DB_PATH        = config.OUTPUT_DIR / "damage_locations.json"
ROAD_DAMAGE_MODEL_DIR = config.MODELS_DIR / "road_damage" / "weights"

# أصناف التلف — أسماء النموذج الحقيقية + Roboflow + CRDDC2022
# المفاتيح lowercase دائماً (lookup يُحوّل الاسم لـ lower)
DAMAGE_INFO = {
    # أسماء النموذج المستخدم (Longitudinal Crack, Transverse Crack, Alligator Crack, Potholes)
    "longitudinal crack":  {"label": "تشقق طولي",     "color": "orange", "severity": "متوسطة"},
    "transverse crack":    {"label": "تشقق عرضي",     "color": "orange", "severity": "متوسطة"},
    "alligator crack":     {"label": "تشقق تمساحي",   "color": "red",    "severity": "عالية"},
    "potholes":            {"label": "حفرة",           "color": "red",    "severity": "عالية"},
    # تنويعات شائعة
    "pothole":             {"label": "حفرة",           "color": "red",    "severity": "عالية"},
    "crack":               {"label": "تشقق",           "color": "orange", "severity": "متوسطة"},
    "longitudinal-crack":  {"label": "تشقق طولي",     "color": "orange", "severity": "متوسطة"},
    "longitudinal_crack":  {"label": "تشقق طولي",     "color": "orange", "severity": "متوسطة"},
    "transverse-crack":    {"label": "تشقق عرضي",     "color": "orange", "severity": "متوسطة"},
    "transverse_crack":    {"label": "تشقق عرضي",     "color": "orange", "severity": "متوسطة"},
    "alligator-crack":     {"label": "تشقق تمساحي",   "color": "red",    "severity": "عالية"},
    "alligator_crack":     {"label": "تشقق تمساحي",   "color": "red",    "severity": "عالية"},
    # CRDDC2022
    "d00":                 {"label": "تشقق طولي",     "color": "orange", "severity": "متوسطة"},
    "d10":                 {"label": "تشقق عرضي",     "color": "orange", "severity": "متوسطة"},
    "d20":                 {"label": "تشقق تمساحي",   "color": "red",    "severity": "عالية"},
    "d40":                 {"label": "حفرة",           "color": "red",    "severity": "عالية"},
}

SEVERITY_ORDER = {"عالية": 0, "متوسطة": 1, "غير محدد": 2}


def get_damage_info(class_name: str) -> dict:
    return DAMAGE_INFO.get(class_name.lower(), {
        "label": class_name, "color": "gray", "severity": "غير محدد"
    })


def find_road_model() -> Optional[str]:
    for name in ("best.pt", "last.pt", "road_damage.pt"):
        p = ROAD_DAMAGE_MODEL_DIR / name
        if p.exists():
            return str(p)
    return None


# ══════════════════════════════════════════
#  قاعدة البيانات (JSON)
# ══════════════════════════════════════════

def load_damage_db() -> list:
    if not DAMAGE_DB_PATH.exists():
        return []
    with open(DAMAGE_DB_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_damage_records(new_records: list):
    db = load_damage_db()
    db.extend(new_records)
    with open(DAMAGE_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def clear_damage_db():
    with open(DAMAGE_DB_PATH, "w", encoding="utf-8") as f:
        json.dump([], f)


# ══════════════════════════════════════════
#  رسم نتائج الكشف على الصورة
# ══════════════════════════════════════════

def draw_damage_detections(image_bgr, results):
    """ارسم بوكسات الكشف وارجع (صورة RGB، قائمة الكشوفات)."""
    out = image_bgr.copy()
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        name   = results.names[cls_id]
        conf   = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        info  = get_damage_info(name)
        color = (30, 30, 220) if info["color"] == "red" else (0, 140, 255)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{info['label']}  {conf:.0%}"
        _pil_text_ar(out, label, (x1, y1), font_size=20, bg_color=color)

        detections.append({"type": name, "conf": round(conf, 4),
                            "bbox": [x1, y1, x2, y2], "info": info})

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), detections


def draw_demo_detections(image_bgr):
    """نمط محاكاة — يرسم كشوفات وهمية للعرض التوضيحي."""
    h, w = image_bgr.shape[:2]
    out  = image_bgr.copy()
    detections = []

    demo_types = [
        ("pothole",            0.91),
        ("longitudinal-crack", 0.83),
        ("pothole",            0.77),
    ]
    for i, (dtype, conf) in enumerate(demo_types):
        x1 = int(w * (0.1 + i * 0.28))
        y1 = int(h * 0.30)
        x2 = x1 + int(w * 0.18)
        y2 = y1 + int(h * 0.25)

        info  = get_damage_info(dtype)
        color = (30, 30, 220) if info["color"] == "red" else (0, 140, 255)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{info['label']} {conf:.0%}"
        _pil_text_ar(out, label, (x1, y1), font_size=20, bg_color=color)

        detections.append({"type": dtype, "conf": conf,
                            "bbox": [x1, y1, x2, y2], "info": info})

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), detections


# ══════════════════════════════════════════
#  إنشاء الخريطة التراكمية
# ══════════════════════════════════════════

def create_damage_map(records: list) -> str:
    if records:
        center_lat = sum(r["lat"] for r in records) / len(records)
        center_lon = sum(r["lon"] for r in records) / len(records)
    else:
        center_lat, center_lon = config.DEMO_GPS_LAT, config.DEMO_GPS_LON

    m = folium.Map(location=[center_lat, center_lon],
                   zoom_start=14, tiles="CartoDB dark_matter")
    MiniMap(toggle_display=True).add_to(m)

    for r in records:
        info = get_damage_info(r.get("damage_type", ""))
        lat, lon = r["lat"], r["lon"]

        clr_hex = ("#ef4444" if info["color"] == "red"
                   else "#f97316" if info["color"] == "orange"
                   else "#94a3b8")

        popup_html = f"""
        <div dir='rtl' style='font-family:Arial;font-size:13px;line-height:2;width:210px'>
          <b style='color:{clr_hex};font-size:15px'>⚠️ {info["label"]}</b><br>
          <b>📅</b> {r.get("timestamp","")}<br>
          <b>🎯 الثقة:</b> {r.get("confidence", 0):.0%}<br>
          <b>🔴 الخطورة:</b> {info["severity"]}<br>
          <b>📍</b> {r.get("location","")}<br>
          <a href="https://www.google.com/maps?q={lat},{lon}" target="_blank"
             style='color:#60a5fa'>📌 افتح في Google Maps</a>
        </div>"""

        folium.Marker(
            [lat, lon],
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"⚠️ {info['label']} — {r.get('timestamp','')}",
            icon=folium.Icon(color=info["color"], icon="warning-sign",
                             prefix="glyphicon"),
        ).add_to(m)

        folium.Circle(
            [lat, lon], radius=300,
            color=clr_hex, fill=True, fill_opacity=0.08, weight=1.5,
        ).add_to(m)

    legend_html = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:9999;
                background:#1e293b;padding:14px 18px;border-radius:10px;
                border:1px solid #334155;font-family:Arial;font-size:13px;
                color:#e2e8f0;box-shadow:2px 2px 8px rgba(0,0,0,.5)">
      <div dir="rtl">
        <b style="font-size:14px;color:#f1f5f9">📍 مفتاح الخريطة</b><br><br>
        🔴 حفرة / تشقق تمساحي<br>
        🟠 تشقق طولي / عرضي<br>
        ⭕ منطقة تنبيه (300 م)<br><br>
        <b style="color:#94a3b8">إجمالي المواقع: {len(records)}</b>
      </div>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    out_path = config.OUTPUT_DIR / "damage_map.html"
    m.save(str(out_path))
    return str(out_path)
