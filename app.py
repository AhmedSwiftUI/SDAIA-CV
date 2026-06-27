"""
نظام رصد الحوادث المرورية + رصد تلف الطريق
streamlit run app.py
"""

import sys, os, time, json, tempfile, uuid
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

import config
from utils.map_utils import create_accident_map
from utils.notify_utils import send_alert
from utils.road_damage_utils import (
    load_damage_db, save_damage_records, clear_damage_db,
    create_damage_map, draw_damage_detections, draw_demo_detections,
    find_road_model, get_damage_info,
)

# ══════════════════════════════════════════
#  إعداد الصفحة
# ══════════════════════════════════════════
st.set_page_config(
    page_title="نظام الرصد الذكي للطرق",
    page_icon="🚦", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700;900&display=swap');
*, html, body { font-family: 'Tajawal', sans-serif !important; }
.main { background: #0f172a !important; }
.stApp { background: #0f172a !important; }
.block-container { padding-top: 1rem !important; }
section[data-testid="stSidebar"] { background: #1e293b !important; }

.kpi-card {
    background: #1e293b; border-radius: 14px; padding: 18px 14px;
    text-align: center; border-top: 4px solid var(--c); margin-bottom: 8px;
}
.kpi-val  { font-size: 30px; font-weight: 900; color: var(--c); line-height: 1; }
.kpi-lbl  { font-size: 13px; color: #94a3b8; margin-top: 6px; }

.notify-card {
    border-radius: 14px; padding: 20px 22px; direction: rtl; line-height: 2;
}
.notify-card.green { background: #052e16; border: 1.5px solid #16a34a; }
.notify-card.blue  { background: #0c1a2e; border: 1.5px solid #3b82f6; }
.notify-title { font-size: 20px; font-weight: 800; margin-bottom: 6px; }
.notify-row   { font-size: 14px; color: #cbd5e1; }
.notify-row b { color: #f1f5f9; }

.section-hdr {
    color: #64748b; font-size: 13px; font-weight: 700; letter-spacing: 1px;
    border-bottom: 1px solid #1e293b; padding-bottom: 4px; margin: 18px 0 10px;
    direction: rtl;
}
.damage-card {
    background: #1e293b; border-radius: 12px; padding: 14px 16px;
    border-right: 4px solid var(--c); margin-bottom: 8px; direction: rtl;
}
.damage-card.high { --c: #ef4444; }
.damage-card.med  { --c: #f97316; }
.damage-card.low  { --c: #94a3b8; }
.damage-type { font-size: 15px; font-weight: 800; color: var(--c); }
.damage-meta { font-size: 12px; color: #94a3b8; margin-top: 3px; }
footer { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════
#  الثوابت
# ══════════════════════════════════════════
ACCIDENT_CLASSES = {
    "accident","car-accident","car-crash","colision",
    "unitario","volcadura","moderate","moderate-accident",
    "severe","severe-accident","object-accident",
}
CONF_THRESHOLD = 0.80
FRAMES_CONFIRM = 4
SKIP_FRAMES    = 2

# ══════════════════════════════════════════
#  دوال مساعدة
# ══════════════════════════════════════════
def find_accident_model():
    p = config.MODELS_DIR / "accident_detector" / "weights" / "best.pt"
    if not p.exists():
        st.error("❌ نموذج كشف الحوادث غير موجود — شغّل 1_train.py أولاً")
        st.stop()
    return str(p)


def draw_frame(frame, results, alert_confirmed, frame_num, total):
    out = frame.copy()
    acc_boxes = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        name   = results.names[cls_id]
        conf   = float(box.conf[0])
        if name.lower() not in ACCIDENT_CLASSES:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        acc_boxes.append({"conf": conf, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        clr = (0, 0, 210)
        cv2.rectangle(out, (x1, y1), (x2, y2), clr, 2)
        lbl = f"Accident  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1)
        cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 6, y1), clr, -1)
        cv2.putText(out, lbl, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1)

    h, w = out.shape[:2]
    if alert_confirmed:
        cv2.rectangle(out, (0, 0), (w, 50), (0, 0, 190), -1)
        cv2.putText(out, "ACCIDENT DETECTED",
                    (12, 34), cv2.FONT_HERSHEY_DUPLEX, 0.82, (255, 255, 255), 2)
    else:
        ov = out.copy()
        cv2.rectangle(ov, (0, 0), (w, 48), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.55, out, 0.45, 0, out)
        bw = int((frame_num / max(total, 1)) * (w - 20))
        cv2.rectangle(out, (10, 30), (w - 10, 42), (40, 40, 40), -1)
        if bw > 0:
            cv2.rectangle(out, (10, 30), (10 + bw, 42), (0, 170, 255), -1)
        pct = int(100 * frame_num / max(total, 1))
        cv2.putText(out, f"Scanning... {pct}%  Frame {frame_num}/{total}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), acc_boxes


def chart_confidence(df):
    fig = go.Figure()
    if not df.empty:
        fig.add_trace(go.Scatter(
            x=df["time"], y=df["conf"],
            mode="lines", line=dict(color="#38bdf8", width=2.5),
            fill="tozeroy", fillcolor="rgba(56,189,248,0.12)",
            name="درجة الثقة",
        ))
    fig.update_layout(
        title=dict(text="📈 درجة ثقة الكشف أثناء الفيديو", font=dict(size=15)),
        paper_bgcolor="#1e293b", plot_bgcolor="#0f172a", font_color="#e2e8f0",
        xaxis=dict(title="الزمن (ثانية)", gridcolor="#1e293b", zerolinecolor="#1e293b"),
        yaxis=dict(title="الثقة", range=[0, 1], gridcolor="#1e293b"),
        margin=dict(l=10, r=10, t=44, b=10), height=280,
    )
    return fig

# ══════════════════════════════════════════
#  الشريط الجانبي
# ══════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:10px 0 20px'>
      <div style='font-size:48px'>🚦</div>
      <div style='font-size:18px;font-weight:900;color:#f8fafc'>نظام الرصد الذكي</div>
      <div style='font-size:12px;color:#475569;margin-top:4px'>Smart Road Monitoring</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-hdr">🎯 وضع الكشف</p>', unsafe_allow_html=True)
    mode = st.radio(
        "وضع الكشف",
        ["🚨  كشف الحوادث المرورية", "🕳️  رصد تلف الطريق"],
        label_visibility="collapsed",
    )
    is_accident_mode = mode.startswith("🚨")

    st.markdown("---")

    if is_accident_mode:
        st.markdown('<p class="section-hdr">📁 فيديو الداش كام</p>',
                    unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "رفع فيديو", type=["mp4", "avi", "mov", "mkv"],
            label_visibility="collapsed", key="acc_upload"
        )
        rd_file = None
    else:
        st.markdown('<p class="section-hdr">📂 صورة أو فيديو الطريق</p>',
                    unsafe_allow_html=True)
        rd_file = st.file_uploader(
            "رفع ملف", type=["jpg", "jpeg", "png", "webp", "mp4", "avi", "mov", "mkv"],
            label_visibility="collapsed", key="rd_upload"
        )
        uploaded = None

        st.markdown('<p class="section-hdr">📍 الموقع الجغرافي</p>',
                    unsafe_allow_html=True)
        rd_lat = st.number_input("خط العرض", value=config.DEMO_GPS_LAT,
                                 format="%.6f", key="rd_lat")
        rd_lon = st.number_input("خط الطول", value=config.DEMO_GPS_LON,
                                 format="%.6f", key="rd_lon")
        rd_loc = st.text_input("اسم الموقع", value=config.LOCATION_NAME, key="rd_loc")

    st.markdown("---")
    file_ready = (uploaded is not None) if is_accident_mode else (rd_file is not None)
    start_btn = st.button(
        "▶️  ابدأ الكشف",
        type="primary",
        disabled=not file_ready,
        use_container_width=True,
    )

    st.markdown("---")
    if is_accident_mode:
        st.markdown('<p class="section-hdr">📊 إحصائيات مباشرة</p>',
                    unsafe_allow_html=True)
        ph_frame = st.empty()
        ph_count = st.empty()
        ph_conf  = st.empty()
        ph_alert = st.empty()
        ph_frame.metric("الإطار الحالي",  "—")
        ph_count.metric("حوادث مرصودة",  "0")
        ph_conf.metric("آخر درجة ثقة",   "—")
        ph_alert.metric("وقت أول حادث",  "—")

# ══════════════════════════════════════════
#  الترويسة الرئيسية
# ══════════════════════════════════════════
if is_accident_mode:
    st.markdown("""
    <h1 style='text-align:center;color:#f8fafc;direction:rtl;margin-bottom:2px;font-size:28px'>
    🚨 نظام رصد الحوادث المرورية
    </h1>
    <p style='text-align:center;color:#475569;margin-bottom:16px;font-size:14px'>
    Dashcam Real-Time Accident Detection — YOLOv8
    </p>""", unsafe_allow_html=True)
else:
    st.markdown("""
    <h1 style='text-align:center;color:#f8fafc;direction:rtl;margin-bottom:2px;font-size:28px'>
    🕳️ نظام رصد تلف الطريق
    </h1>
    <p style='text-align:center;color:#475569;margin-bottom:16px;font-size:14px'>
    Road Damage Detection — Potholes &amp; Cracks — YOLOv8
    </p>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PLACEHOLDER CONTAINERS  (يُعرَّف دائماً بغض النظر عن الوضع)
# ══════════════════════════════════════════════════════════════

# ── حوادث ──
alert_ph   = st.empty()
col_v, col_m = st.columns([1, 1], gap="medium")
with col_v:
    st.markdown('<p class="section-hdr">📹 بث الفيديو المباشر</p>',
                unsafe_allow_html=True)
    frame_ph = st.empty()
    frame_ph.markdown(
        "<div style='background:#1e293b;height:340px;border-radius:12px;"
        "display:flex;align-items:center;justify-content:center;"
        "color:#334155;font-size:16px'>ارفع ملفاً وابدأ الكشف</div>",
        unsafe_allow_html=True)
with col_m:
    st.markdown('<p class="section-hdr">🗺️ الخريطة التفاعلية</p>',
                unsafe_allow_html=True)
    map_ph = st.empty()
    map_ph.markdown(
        "<div style='background:#1e293b;height:340px;border-radius:12px;"
        "display:flex;align-items:center;justify-content:center;"
        "color:#334155;font-size:16px'>الخريطة تظهر بعد الكشف</div>",
        unsafe_allow_html=True)

# شريط التقدم (رصد الطريق)
rd_prog_ph   = st.empty()
rd_stats_ph  = st.empty()

st.markdown("---")
kpi_ph     = st.empty()
notify_ph  = st.empty()
chart_ph   = st.empty()
table_ph   = st.empty()
export_ph  = st.empty()

# ══════════════════════════════════════════
#  دالة عرض نتائج الحوادث
# ══════════════════════════════════════════
def render_accident_results(df, alert_sec, notify_ts):
    if df.empty:
        return
    conf_arr = df["conf"].values

    with kpi_ph.container():
        st.markdown('<p class="section-hdr">📊 ملخص الحادث</p>', unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4, gap="small")
        k1.markdown(f"""<div class="kpi-card" style="--c:#ef4444">
            <div class="kpi-val">{alert_sec:.1f}s</div>
            <div class="kpi-lbl">🚨 وقت أول حادث</div></div>""", unsafe_allow_html=True)
        k2.markdown(f"""<div class="kpi-card" style="--c:#38bdf8">
            <div class="kpi-val">{len(df)}</div>
            <div class="kpi-lbl">📊 إجمالي الكشوفات</div></div>""", unsafe_allow_html=True)
        k3.markdown(f"""<div class="kpi-card" style="--c:#a78bfa">
            <div class="kpi-val">{np.max(conf_arr):.0%}</div>
            <div class="kpi-lbl">🎯 أعلى ثقة</div></div>""", unsafe_allow_html=True)
        k4.markdown(f"""<div class="kpi-card" style="--c:#34d399">
            <div class="kpi-val">{np.mean(conf_arr):.0%}</div>
            <div class="kpi-lbl">📈 متوسط الثقة</div></div>""", unsafe_allow_html=True)

    with notify_ph.container():
        st.markdown('<p class="section-hdr">📡 حالة الإبلاغ</p>', unsafe_allow_html=True)
        n1, n2 = st.columns(2, gap="medium")
        n1.markdown(f"""
        <div class="notify-card green">
          <div class="notify-title" style="color:#4ade80">🚑 إبلاغ الإسعاف</div>
          <div class="notify-row">✅ &nbsp;<b>تم الإبلاغ بنجاح</b></div>
          <div class="notify-row">🕐 &nbsp;<b>وقت الإبلاغ:</b> {notify_ts}</div>
          <div class="notify-row">📍 &nbsp;<b>الموقع:</b> {config.LOCATION_NAME}</div>
        </div>""", unsafe_allow_html=True)
        n2.markdown(f"""
        <div class="notify-card blue">
          <div class="notify-title" style="color:#60a5fa">🚔 إبلاغ المرور</div>
          <div class="notify-row">✅ &nbsp;<b>تم الإبلاغ بنجاح</b></div>
          <div class="notify-row">🕐 &nbsp;<b>وقت الإبلاغ:</b> {notify_ts}</div>
          <div class="notify-row">📍 &nbsp;<b>الموقع:</b> {config.LOCATION_NAME}</div>
        </div>""", unsafe_allow_html=True)

    with chart_ph.container():
        st.markdown('<p class="section-hdr">📈 مسار ثقة الكشف</p>', unsafe_allow_html=True)
        st.plotly_chart(chart_confidence(df), use_container_width=True)

    with table_ph.container():
        st.markdown('<p class="section-hdr">🕐 سجل الحوادث</p>', unsafe_allow_html=True)
        disp = df.copy()
        disp["conf"] = disp["conf"].apply(lambda x: f"{x:.1%}")
        disp.columns = ["الإطار", "الوقت (ث)", "درجة الثقة",
                        "وقت الرصد", "الموقع", "خط العرض", "خط الطول"]
        st.dataframe(disp, use_container_width=True, hide_index=True)

    with export_ph.container():
        st.markdown('<p class="section-hdr">⬇️ تصدير النتائج</p>', unsafe_allow_html=True)
        e1, e2 = st.columns(2)
        e1.download_button("📄 CSV", df.to_csv(index=False).encode("utf-8-sig"),
                           "accident_report.csv", "text/csv", use_container_width=True)
        e2.download_button("📦 JSON",
                           df.to_json(orient="records", force_ascii=False,
                                      indent=2).encode("utf-8"),
                           "accident_report.json", "application/json",
                           use_container_width=True)


# ══════════════════════════════════════════
#  الخريطة التراكمية لتلف الطريق (تُعرض دائماً في وضع الطريق)
# ══════════════════════════════════════════
def render_damage_summary():
    db = load_damage_db()
    if not db:
        return

    with kpi_ph.container():
        st.markdown('<p class="section-hdr">📊 إحصائيات التلف</p>', unsafe_allow_html=True)
        high_c = sum(1 for r in db if get_damage_info(r["damage_type"])["color"] == "red")
        med_c  = sum(1 for r in db if get_damage_info(r["damage_type"])["color"] == "orange")
        k1, k2, k3, k4 = st.columns(4, gap="small")
        k1.markdown(f"""<div class="kpi-card" style="--c:#38bdf8">
            <div class="kpi-val">{len(db)}</div>
            <div class="kpi-lbl">📍 إجمالي المواقع</div></div>""", unsafe_allow_html=True)
        k2.markdown(f"""<div class="kpi-card" style="--c:#ef4444">
            <div class="kpi-val">{high_c}</div>
            <div class="kpi-lbl">🔴 خطورة عالية</div></div>""", unsafe_allow_html=True)
        k3.markdown(f"""<div class="kpi-card" style="--c:#f97316">
            <div class="kpi-val">{med_c}</div>
            <div class="kpi-lbl">🟠 خطورة متوسطة</div></div>""", unsafe_allow_html=True)
        k4.markdown(f"""<div class="kpi-card" style="--c:#94a3b8">
            <div class="kpi-val">{len(db) - high_c - med_c}</div>
            <div class="kpi-lbl">⚪ غير محدد</div></div>""", unsafe_allow_html=True)

    with table_ph.container():
        st.markdown('<p class="section-hdr">📋 سجل المواقع المحفوظة</p>',
                    unsafe_allow_html=True)
        df_db = pd.DataFrame(db)
        df_show = df_db.copy()
        df_show["damage_type"] = df_show["damage_type"].apply(
            lambda x: get_damage_info(x)["label"])
        df_show["confidence"] = df_show["confidence"].apply(lambda x: f"{x:.0%}")
        cols = [c for c in ["timestamp", "damage_type", "confidence", "location", "lat", "lon"]
                if c in df_show.columns]
        df_show = df_show[cols]
        df_show.columns = ["التاريخ والوقت", "نوع التلف", "الثقة",
                           "الموقع", "خط العرض", "خط الطول"][:len(cols)]
        st.dataframe(df_show, use_container_width=True, hide_index=True)

    with export_ph.container():
        st.markdown('<p class="section-hdr">⬇️ تصدير | مسح</p>', unsafe_allow_html=True)
        ex1, ex2, ex3 = st.columns(3)
        ex1.download_button("📄 CSV", df_db.to_csv(index=False).encode("utf-8-sig"),
                            "road_damage.csv", "text/csv", use_container_width=True)
        ex2.download_button("📦 JSON",
                            json.dumps(db, ensure_ascii=False, indent=2).encode("utf-8"),
                            "road_damage.json", "application/json",
                            use_container_width=True)
        if ex3.button("🗑️ مسح البيانات", type="secondary", use_container_width=True):
            clear_damage_db()
            st.rerun()


# ══════════════════════════════════════════
#  عرض الخريطة التراكمية للطريق تلقائياً
# ══════════════════════════════════════════
if not is_accident_mode:
    db_now = load_damage_db()
    if db_now:
        map_html = create_damage_map(db_now)
        with open(map_html, encoding="utf-8") as mf:
            with map_ph:
                components.html(mf.read(), height=340, scrolling=False)
    render_damage_summary()

# ══════════════════════════════════════════
#  حلقة الكشف الرئيسية
# ══════════════════════════════════════════
if start_btn and file_ready:
    from ultralytics import YOLO

    # ──────────────────────────────────────
    #  وضع كشف الحوادث
    # ──────────────────────────────────────
    if is_accident_mode and uploaded:
        alert_ph.markdown(
            "<div style='background:#1e3a5f;border:1px solid #3b82f6;"
            "border-radius:10px;padding:10px 20px;text-align:center;"
            "color:#93c5fd;direction:rtl'>🔄 جاري تحميل الفيديو وبدء الكشف...</div>",
            unsafe_allow_html=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        model = YOLO(find_accident_model())
        cap   = cv2.VideoCapture(tmp_path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        records     = []
        consecutive = 0
        alert_sent  = False
        alert_sec   = None
        notify_ts   = None
        frame_num   = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1
            if frame_num % SKIP_FRAMES != 0 and not alert_sent:
                continue

            results        = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]
            rgb, acc_boxes = draw_frame(frame, results, alert_sent, frame_num, total)
            detected_now   = len(acc_boxes) > 0
            consecutive    = consecutive + 1 if detected_now else 0

            if detected_now:
                best = max(b["conf"] for b in acc_boxes)
                records.append({
                    "frame": frame_num, "time": round(frame_num / fps, 2),
                    "conf": round(best, 4), "timestamp": time.strftime("%H:%M:%S"),
                    "location": config.LOCATION_NAME,
                    "lat": config.DEMO_GPS_LAT, "lon": config.DEMO_GPS_LON,
                })

            if consecutive >= FRAMES_CONFIRM and not alert_sent:
                alert_sent = True
                alert_sec  = frame_num / fps
                notify_ts  = time.strftime("%Y-%m-%d  %H:%M:%S")

                create_accident_map(config.DEMO_GPS_LAT, config.DEMO_GPS_LON,
                                    config.LOCATION_NAME, notify_ts)
                send_alert(config.DEMO_GPS_LAT, config.DEMO_GPS_LON,
                           config.LOCATION_NAME, notify_ts)

                alert_ph.markdown(f"""
                <div style='background:#7f1d1d;border:2px solid #ef4444;
                     border-radius:12px;padding:14px 24px;text-align:center;
                     direction:rtl;margin-bottom:8px'>
                  <span style='font-size:28px'>🚨</span>
                  <span style='font-size:20px;font-weight:900;color:#fca5a5;
                               margin-right:10px'>تم رصد حادث مروري!</span><br>
                  <span style='color:#fecaca;font-size:14px'>
                    {config.LOCATION_NAME} | {notify_ts} | الثانية {alert_sec:.1f}
                  </span>
                </div>""", unsafe_allow_html=True)

                map_path = config.OUTPUT_DIR / "accident_map.html"
                if map_path.exists():
                    with open(map_path, encoding="utf-8") as f:
                        with map_ph:
                            components.html(f.read(), height=340, scrolling=False)

            frame_ph.image(rgb, use_container_width=True)
            ph_frame.metric("الإطار الحالي", f"{frame_num}/{total}")
            ph_count.metric("حوادث مرصودة", str(len(records)))
            if records:
                ph_conf.metric("آخر درجة ثقة", f"{records[-1]['conf']:.0%}")
            if alert_sec:
                ph_alert.metric("وقت أول حادث", f"{alert_sec:.1f}s")

            if frame_num % 60 == 0 and records and alert_sent:
                render_accident_results(pd.DataFrame(records), alert_sec, notify_ts)

        cap.release()
        os.unlink(tmp_path)

        df_final = pd.DataFrame(records)
        if not df_final.empty:
            render_accident_results(df_final, alert_sec, notify_ts)
            df_final.to_csv(config.OUTPUT_DIR / "accident_report.csv",
                            index=False, encoding="utf-8-sig")
        else:
            alert_ph.empty()
            st.warning("⚠️ لم يُرصد أي حادث في هذا الفيديو")

    # ──────────────────────────────────────
    #  وضع رصد تلف الطريق
    # ──────────────────────────────────────
    elif not is_accident_mode and rd_file:
        road_model_path = find_road_model()
        demo_mode       = road_model_path is None
        now_ts          = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        suffix          = Path(rd_file.name).suffix.lower()
        is_video        = suffix in {".mp4", ".avi", ".mov", ".mkv"}

        if not demo_mode:
            rd_model = YOLO(road_model_path)

        alert_ph.markdown(
            "<div style='background:#1a2e1a;border:1px solid #16a34a;"
            "border-radius:10px;padding:10px 20px;text-align:center;"
            "color:#86efac;direction:rtl'>🔄 جاري تحليل الملف...</div>",
            unsafe_allow_html=True)

        # ── فيديو ──
        if is_video:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(rd_file.read())
                tmp_path = tmp.name

            cap   = cv2.VideoCapture(tmp_path)
            fps   = cap.get(cv2.CAP_PROP_FPS) or 30
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            skip  = 3

            best_per_type = {}
            frame_num = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_num += 1
                if frame_num % skip != 0:
                    continue

                if demo_mode:
                    rgb_out, detections = draw_demo_detections(frame)
                else:
                    res = rd_model.predict(frame, conf=0.35, verbose=False)[0]
                    rgb_out, detections = draw_damage_detections(frame, res)

                for d in detections:
                    dt = d["type"]
                    if dt not in best_per_type or d["conf"] > best_per_type[dt]["conf"]:
                        best_per_type[dt] = {"conf": d["conf"], "info": d["info"]}

                frame_ph.image(rgb_out, use_container_width=True)

                pct = frame_num / max(total, 1)
                rd_prog_ph.markdown(
                    f"<div style='color:#94a3b8;font-size:13px;direction:rtl;"
                    f"padding:6px 0'>⏳ معالجة الإطار {frame_num} / {total} "
                    f"— {pct*100:.0f}%</div>",
                    unsafe_allow_html=True)

                if best_per_type:
                    cards = ""
                    for dtype, info_d in best_per_type.items():
                        info = info_d["info"]
                        sev_cls = ("high" if info["color"] == "red"
                                   else "med" if info["color"] == "orange" else "low")
                        cards += f"""
                        <div class="damage-card {sev_cls}" style="margin-bottom:6px">
                          <div class="damage-type">⚠️ {info["label"]}</div>
                          <div class="damage-meta">🎯 {info_d["conf"]:.0%} | {info["severity"]}</div>
                        </div>"""
                    rd_stats_ph.markdown(cards, unsafe_allow_html=True)

            cap.release()
            os.unlink(tmp_path)
            rd_prog_ph.empty()

            new_records = [
                {"id": str(uuid.uuid4())[:8], "damage_type": dtype,
                 "confidence": round(info_d["conf"], 4),
                 "lat": rd_lat, "lon": rd_lon, "location": rd_loc,
                 "timestamp": now_ts, "demo": demo_mode}
                for dtype, info_d in best_per_type.items()
            ]

        # ── صورة ──
        else:
            img_bytes = rd_file.read()
            nparr     = np.frombuffer(img_bytes, np.uint8)
            img_bgr   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if demo_mode:
                rgb_out, detections = draw_demo_detections(img_bgr)
            else:
                res = rd_model.predict(img_bgr, conf=0.35, verbose=False)[0]
                rgb_out, detections = draw_damage_detections(img_bgr, res)

            frame_ph.image(rgb_out, use_container_width=True)

            new_records = [
                {"id": str(uuid.uuid4())[:8], "damage_type": d["type"],
                 "confidence": round(d["conf"], 4),
                 "lat": rd_lat, "lon": rd_lon, "location": rd_loc,
                 "timestamp": now_ts, "demo": demo_mode}
                for d in detections
            ]

        # حفظ وعرض النتائج
        if new_records:
            save_damage_records(new_records)
            alert_ph.markdown(
                f"<div style='background:#052e16;border:1.5px solid #16a34a;"
                f"border-radius:10px;padding:10px 20px;text-align:center;"
                f"color:#4ade80;direction:rtl'>"
                f"✅ تم رصد <b>{len(new_records)}</b> نوع تلف وحفظ موقعه في قاعدة البيانات"
                f"</div>",
                unsafe_allow_html=True)
        else:
            alert_ph.markdown(
                "<div style='background:#1c1917;border:1px solid #78716c;"
                "border-radius:10px;padding:10px 20px;text-align:center;"
                "color:#a8a29e;direction:rtl'>"
                "⚠️ لم يُرصد أي تلف في هذا الملف</div>",
                unsafe_allow_html=True)

        # تحديث الخريطة
        db_updated = load_damage_db()
        if db_updated:
            map_html = create_damage_map(db_updated)
            with open(map_html, encoding="utf-8") as mf:
                with map_ph:
                    components.html(mf.read(), height=340, scrolling=False)

        render_damage_summary()
