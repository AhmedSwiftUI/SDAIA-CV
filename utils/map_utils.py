"""
إنشاء خريطة تفاعلية تُظهر موقع الحادث مع:
  - علامة حمراء على مكان الحادث
  - دائرة تنبيه نصف قطرها 1 كم
  - محاكاة سائقين قادمين تم تنبيههم
  - علامات الإسعاف والمرور
"""

import math
import folium
from folium.plugins import MiniMap
import config


def _offset(lat: float, lon: float, dx_m: float, dy_m: float):
    """إزاحة إحداثيات بمسافة بالأمتار."""
    dlat = dy_m / 111_320
    dlon = dx_m / (111_320 * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon


def create_accident_map(lat: float, lon: float, location_name: str, timestamp: str) -> str:
    m = folium.Map(location=[lat, lon], zoom_start=15, tiles="OpenStreetMap")
    MiniMap(toggle_display=True).add_to(m)

    # ── علامة الحادث ──
    popup_html = f"""
    <div dir="rtl" style="font-family:Arial;font-size:14px;width:220px;line-height:1.8">
        <b style="color:red;font-size:16px;">🚨 حادث مروري</b><br>
        <b>📍 الموقع:</b> {location_name}<br>
        <b>🕐 الوقت:</b> {timestamp}<br>
        <b>🌐 الإحداثيات:</b><br>
        &nbsp;&nbsp;{lat:.6f}°N &nbsp; {lon:.6f}°E<br>
        <a href="https://www.google.com/maps?q={lat},{lon}" target="_blank">
        📌 افتح في Google Maps</a>
    </div>"""
    folium.Marker(
        [lat, lon],
        popup=folium.Popup(popup_html, max_width=300),
        tooltip="🚨 موقع الحادث — انقر للتفاصيل",
        icon=folium.Icon(color="red", icon="exclamation-triangle", prefix="fa"),
    ).add_to(m)

    # ── دائرة التنبيه 1 كم ──
    folium.Circle(
        location=[lat, lon],
        radius=config.ALERT_RADIUS_KM * 1000,
        color="#FF6600",
        weight=2.5,
        fill=True,
        fill_color="#FF6600",
        fill_opacity=0.12,
        tooltip=f"⚠️ منطقة التنبيه — {config.ALERT_RADIUS_KM} كم",
        popup="<b>⚠️ منطقة التنبيه</b><br>جميع السائقين داخل هذا النطاق تلقّوا إشعاراً",
    ).add_to(m)

    # ── سائقون قادمون تم تنبيههم (محاكاة) ──
    approaching = [
        {"dx":  320, "dy":  680, "dir": "الشمال"},
        {"dx": -450, "dy":  250, "dir": "الغرب"},
        {"dx":  580, "dy": -310, "dir": "الجنوب الشرقي"},
        {"dx": -200, "dy": -700, "dir": "الجنوب"},
    ]
    for d in approaching:
        dlat, dlon = _offset(lat, lon, d["dx"], d["dy"])
        folium.Marker(
            [dlat, dlon],
            popup=(
                f"<div dir='rtl'>"
                f"<b>🚗 سائق قادم من {d['dir']}</b><br>"
                f"✅ تم إرسال تنبيه: <i>⚠️ حادث أمامك!</i>"
                f"</div>"
            ),
            tooltip=f"🚗 سائق — {d['dir']} (تم تنبيهه)",
            icon=folium.Icon(color="blue", icon="car", prefix="fa"),
        ).add_to(m)

    # ── الإسعاف ──
    alat, alon = _offset(lat, lon, 900, 1100)
    folium.Marker(
        [alat, alon],
        popup="<div dir='rtl'><b>🚑 الإسعاف</b><br>✅ تم الإبلاغ — في الطريق</div>",
        tooltip="🚑 الإسعاف",
        icon=folium.Icon(color="green", icon="plus-square", prefix="fa"),
    ).add_to(m)

    # ── المرور ──
    tlat, tlon = _offset(lat, lon, -750, -900)
    folium.Marker(
        [tlat, tlon],
        popup="<div dir='rtl'><b>🚔 إدارة المرور</b><br>✅ تم الإبلاغ — جاري التحرك</div>",
        tooltip="🚔 المرور",
        icon=folium.Icon(color="purple", icon="shield", prefix="fa"),
    ).add_to(m)

    # ── مفتاح الخريطة ──
    legend = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:9999;
                background:white;padding:14px 18px;border-radius:10px;
                border:2px solid #ccc;font-family:Arial;font-size:13px;
                box-shadow:2px 2px 6px rgba(0,0,0,0.2)">
        <div dir="rtl">
        <b style="font-size:15px">📍 مفتاح الخريطة</b><br><br>
        🔴 موقع الحادث<br>
        🟠 منطقة التنبيه (1 كم)<br>
        🚗 سائق تم تنبيهه<br>
        🚑 الإسعاف (في الطريق)<br>
        🚔 المرور (في الطريق)
        </div>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))

    out_path = str(config.OUTPUT_DIR / "accident_map.html")
    m.save(out_path)
    print(f"🗺️  الخريطة محفوظة: {out_path}")
    return out_path
