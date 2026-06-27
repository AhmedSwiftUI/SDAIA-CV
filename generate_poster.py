"""
يولّد ملف PDF إنفوغرافيك لمشروع AccidentDetector
python3 generate_poster.py
"""

import math
from pathlib import Path

import arabic_reshaper
from bidi.algorithm import get_display
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

# ── خطوط عربية ──────────────────────────────────────────────
FONT_PATH    = "/System/Library/Fonts/SFArabic.ttf"
pdfmetrics.registerFont(TTFont("Arabic",       FONT_PATH))
pdfmetrics.registerFont(TTFont("ArabicBold",   "/System/Library/Fonts/SFArabicRounded.ttf"))

# ── ألوان النظام ──────────────────────────────────────────────
BG_DARK   = colors.HexColor("#0f172a")
BG_CARD   = colors.HexColor("#1e293b")
BG_CARD2  = colors.HexColor("#162032")
ACCENT    = colors.HexColor("#6366f1")   # بنفسجي
ACCENT2   = colors.HexColor("#22d3ee")   # سماوي
GREEN     = colors.HexColor("#10b981")
ORANGE    = colors.HexColor("#f59e0b")
RED       = colors.HexColor("#ef4444")
MUTED     = colors.HexColor("#94a3b8")
WHITE     = colors.white
YELLOW    = colors.HexColor("#fbbf24")

OUT = Path(__file__).parent / "output" / "project_poster.pdf"
OUT.parent.mkdir(exist_ok=True)

W, H = A4          # 595 × 842 pt
PAD  = 14 * mm


# ─────────────────────────────────────────────
#  مساعدات
# ─────────────────────────────────────────────
def ar(text: str) -> str:
    """تحويل النص العربي لعرض صحيح RTL."""
    return get_display(arabic_reshaper.reshape(text))


def draw_rect(c: canvas.Canvas, x, y, w, h, fill=BG_CARD, radius=8, stroke=None, stroke_w=1):
    c.saveState()
    c.setFillColor(fill)
    if stroke:
        c.setStrokeColor(stroke)
        c.setLineWidth(stroke_w)
        c.roundRect(x, y, w, h, radius, fill=1, stroke=1)
    else:
        c.setStrokeColor(fill)
        c.roundRect(x, y, w, h, radius, fill=1, stroke=0)
    c.restoreState()


def draw_text(c: canvas.Canvas, text: str, x, y, font="Arabic", size=10,
              color=WHITE, align="right"):
    c.saveState()
    c.setFont(font, size)
    c.setFillColor(color)
    t = ar(text)
    if align == "right":
        c.drawRightString(x, y, t)
    elif align == "center":
        c.drawCentredString(x, y, t)
    else:
        c.drawString(x, y, t)
    c.restoreState()


def draw_circle_badge(c, cx, cy, r, fill, text, tsize=9):
    c.saveState()
    c.setFillColor(fill)
    c.circle(cx, cy, r, fill=1, stroke=0)
    c.setFont("ArabicBold", tsize)
    c.setFillColor(WHITE)
    c.drawCentredString(cx, cy - tsize * 0.35, text)
    c.restoreState()


def gradient_rect(c, x, y, w, h, c1, c2, steps=30):
    """تدرج لوني أفقي."""
    sw = w / steps
    r1, g1, b1 = c1.red, c1.green, c1.blue
    r2, g2, b2 = c2.red, c2.green, c2.blue
    for i in range(steps):
        t = i / steps
        col = colors.Color(r1 + t*(r2-r1), g1 + t*(g2-g1), b1 + t*(b2-b1))
        c.setFillColor(col)
        c.rect(x + i*sw, y, sw+1, h, fill=1, stroke=0)


def section_header(c, num: str, title: str, y_top: float, accent=ACCENT):
    """رسم رأس القسم برقم ملون وخط عنوان."""
    # خلفية شريط
    c.saveState()
    c.setFillColor(accent)
    c.roundRect(PAD, y_top - 10, 7, 24, 3, fill=1, stroke=0)
    c.restoreState()

    draw_circle_badge(c, PAD + 18, y_top + 2, 10, accent, num, tsize=9)
    draw_text(c, title, W - PAD, y_top, font="ArabicBold", size=13, color=WHITE)
    # خط فاصل
    c.saveState()
    c.setStrokeColor(accent)
    c.setLineWidth(0.5)
    c.line(PAD + 32, y_top + 2, W - PAD - 5, y_top + 2)
    c.restoreState()


# ─────────────────────────────────────────────
#  البناء الرئيسي
# ─────────────────────────────────────────────
def build():
    c = canvas.Canvas(str(OUT), pagesize=A4)

    # ══ خلفية الصفحة ══════════════════════════
    c.setFillColor(BG_DARK)
    c.rect(0, 0, W, H, fill=1, stroke=0)

    # ══ نقاط زخرفية ════════════════════════════
    c.saveState()
    c.setFillColor(colors.HexColor("#1e293b"))
    for i in range(6):
        for j in range(4):
            c.circle(W - 35 - i*12, H - 30 - j*12, 1.5, fill=1, stroke=0)
    c.restoreState()

    # ══ تدرج خلفية الهيدر ══════════════════════
    gradient_rect(c, 0, H - 80, W, 80,
                  colors.HexColor("#1e1b4b"), colors.HexColor("#0f172a"))

    # ── خط علوي بنفسجي ──
    c.saveState()
    c.setFillColor(ACCENT)
    c.rect(0, H - 4, W, 4, fill=1, stroke=0)
    c.restoreState()

    # ── أيقونة الكاميرا (دائرة) ──
    c.saveState()
    c.setFillColor(colors.HexColor("#312e81"))
    c.circle(PAD + 18, H - 40, 18, fill=1, stroke=0)
    c.setStrokeColor(ACCENT)
    c.setLineWidth(2)
    c.circle(PAD + 18, H - 40, 18, fill=0, stroke=1)
    # رمز الكاميرا
    c.setFillColor(WHITE)
    c.roundRect(PAD + 8, H - 48, 20, 14, 3, fill=1, stroke=0)
    c.setFillColor(colors.HexColor("#312e81"))
    c.circle(PAD + 18, H - 41, 4, fill=1, stroke=0)
    c.restoreState()

    # ── اسم المشروع ──
    draw_text(c, "نظام الرصد الذكي للطرق", W - PAD, H - 28,
              font="ArabicBold", size=20, color=WHITE)
    draw_text(c, "كشف الحوادث المرورية وتلف الطريق بالذكاء الاصطناعي",
              W - PAD, H - 48, font="Arabic", size=11, color=colors.HexColor("#a5b4fc"))
    draw_text(c, "أحمد الحربي  •  رؤية الحاسب للمطورين",
              W - PAD, H - 63, font="Arabic", size=9, color=MUTED)

    y = H - 92

    # ══ 1. نظرة عامة ════════════════════════════
    section_header(c, "١", "نظرة عامة", y)
    y -= 14

    draw_rect(c, PAD, y - 68, W - 2*PAD, 70, fill=BG_CARD, radius=10,
              stroke=colors.HexColor("#334155"), stroke_w=0.7)

    overview_lines = [
        "نظام ذكي مبني على YOLOv8 يعمل على تحليل مقاطع الداش كام",
        "للكشف الفوري عن الحوادث المرورية وتلف الطريق، مع إرسال تنبيهات",
        "تلقائية للإسعاف والمرور وعرض موقع الحادث على خريطة تفاعلية.",
    ]
    for i, line in enumerate(overview_lines):
        draw_text(c, line, W - PAD - 8, y - 15 - i * 14, size=9.5, color=WHITE)

    # ميزات سريعة
    features = [
        (ACCENT,  "🎯", "كشف الحوادث",       "YOLOv8 مدرّب على بيانات حقيقية"),
        (GREEN,   "🛣️",  "رصد تلف الطريق",   "حفر، تشققات طولية وعرضية"),
        (ACCENT2, "📍", "خريطة تفاعلية",     "Folium مع دائرة تنبيه 1 كم"),
    ]
    fw = (W - 2*PAD - 12) / 3
    for i, (col, icon, title, desc) in enumerate(features):
        bx = PAD + i * (fw + 6)
        by = y - 68
        draw_rect(c, bx, by, fw, 28, fill=colors.HexColor("#0f1f3d"), radius=6,
                  stroke=col, stroke_w=1)
        draw_text(c, title, bx + fw - 6, by + 16, font="ArabicBold", size=8, color=col)
        draw_text(c, desc, bx + fw - 6, by + 5, size=7, color=MUTED)
        c.saveState()
        c.setFont("Arabic", 10)
        c.drawString(bx + 6, by + 8, icon)
        c.restoreState()

    y -= 80

    # ══ 2. المنهجية ══════════════════════════════
    section_header(c, "٢", "المنهجية — خط سير العمل", y, accent=ACCENT2)
    y -= 14

    steps = [
        ("١", ACCENT,  "تحميل البيانات",    "Roboflow Dataset"),
        ("٢", ACCENT2, "تدريب النموذج",     "YOLOv8s • 30 epoch"),
        ("٣", GREEN,   "معالجة الفيديو",    "OpenCV إطار بإطار"),
        ("٤", ORANGE,  "الكشف والتأكيد",    "5 إطارات متتالية"),
        ("٥", RED,     "الإنذار والخريطة",  "Folium + Email"),
    ]

    sw2 = (W - 2*PAD - 8) / 5
    bh  = 66
    by  = y - bh - 4

    for i, (num, col, title, sub) in enumerate(steps):
        bx = PAD + i * (sw2 + 2)
        draw_rect(c, bx, by, sw2, bh, fill=BG_CARD, radius=8)
        # شريط علوي ملون
        c.saveState()
        c.setFillColor(col)
        c.roundRect(bx, by + bh - 6, sw2, 6, 4, fill=1, stroke=0)
        c.restoreState()
        draw_circle_badge(c, bx + sw2/2, by + bh - 18, 9, col, num, tsize=8)
        draw_text(c, title, bx + sw2 - 4, by + 28, font="ArabicBold", size=7.5, color=WHITE)
        draw_text(c, sub,   bx + sw2 - 4, by + 14, size=7, color=MUTED)

        # سهم
        if i < 4:
            c.saveState()
            c.setFillColor(col)
            c.setStrokeColor(col)
            ax = bx + sw2 + 1
            ay = by + bh/2
            c.setLineWidth(1)
            c.line(ax - 2, ay, ax + 2, ay)
            c.restoreState()

    y = by - 10

    # ══ 3. التقنيات ══════════════════════════════
    section_header(c, "٣", "الأدوات والمكتبات المستخدمة", y, accent=ORANGE)
    y -= 14

    techs = [
        (ACCENT,  "🤖", "YOLOv8",       "نموذج الكشف"),
        (ACCENT2, "👁️",  "OpenCV",       "معالجة الفيديو"),
        (GREEN,   "📦", "Roboflow",     "Dataset وAPI"),
        (ORANGE,  "🗺️", "Folium",       "الخرائط التفاعلية"),
        (RED,     "⚡", "Streamlit",    "واجهة لوحة التحكم"),
        (YELLOW,  "🐍", "Python",       "لغة البرمجة"),
        (colors.HexColor("#ec4899"), "📊", "Plotly",  "الرسوم البيانية"),
        (MUTED,   "✉️",  "smtplib",     "الإشعارات بالإيميل"),
    ]

    tw2 = (W - 2*PAD - 14) / 4
    rows = [techs[:4], techs[4:]]
    for row_i, row in enumerate(rows):
        for col_i, (col, icon, name, desc) in enumerate(row):
            bx = PAD + col_i * (tw2 + 4)
            tby = y - 36 - row_i * 42
            draw_rect(c, bx, tby, tw2, 36, fill=BG_CARD2, radius=7,
                      stroke=col, stroke_w=0.8)
            c.saveState()
            c.setFont("Arabic", 12)
            c.drawString(bx + 6, tby + 18, icon)
            c.restoreState()
            draw_text(c, name, bx + tw2 - 6, tby + 22, font="ArabicBold", size=8.5, color=col)
            draw_text(c, desc, bx + tw2 - 6, tby + 9, size=7, color=MUTED)

    y -= (36 + 42 + 12)

    # ══ 4. مكونات المشروع ════════════════════════
    section_header(c, "٤", "مكونات المشروع", y, accent=GREEN)
    y -= 14

    files = [
        ("1_train.py",           ACCENT,  "تحميل Dataset من Roboflow وتدريب YOLOv8"),
        ("2_detect.py",          ACCENT2, "معالجة فيديو الداش كام والكشف الآني"),
        ("app.py",               GREEN,   "لوحة تحكم Streamlit بالعربي"),
        ("utils/map_utils.py",   ORANGE,  "إنشاء خريطة Folium مع دائرة التنبيه"),
        ("utils/road_damage_utils.py", RED, "كشف تلف الطريق وعرضه بالعربي"),
        ("utils/notify_utils.py",YELLOW,  "إرسال تنبيهات البريد الإلكتروني"),
    ]

    bh2 = 16
    for i, (fname, col, desc) in enumerate(files):
        fy = y - 8 - i * (bh2 + 4)
        draw_rect(c, PAD, fy, W - 2*PAD, bh2, fill=BG_CARD, radius=5)
        c.saveState()
        c.setFillColor(col)
        c.roundRect(PAD, fy, 4, bh2, 2, fill=1, stroke=0)
        c.restoreState()
        # اسم الملف
        c.saveState()
        c.setFont("ArabicBold", 7.5)
        c.setFillColor(col)
        c.drawString(PAD + 10, fy + 5, fname)
        c.restoreState()
        draw_text(c, desc, W - PAD - 8, fy + 5, size=7.5, color=WHITE)

    y -= (len(files) * (bh2 + 4) + 14)

    # ══ 5. الخلاصة والأعمال المستقبلية ══════════
    half_w = (W - 2*PAD - 6) / 2

    # الخلاصة
    section_header(c, "٥", "الخلاصة", y, accent=ACCENT)
    y -= 12

    conc_h = 62
    draw_rect(c, PAD, y - conc_h, half_w, conc_h, fill=BG_CARD, radius=8,
              stroke=ACCENT, stroke_w=0.8)
    conc_lines = [
        "يُقدّم هذا المشروع حلاً عملياً متكاملاً",
        "لرصد حوادث الطرق وتلفها باستخدام",
        "الذكاء الاصطناعي، ويُعدّ نقطة انطلاق",
        "لأنظمة مدن ذكية أكثر أماناً.",
    ]
    for i, line in enumerate(conc_lines):
        draw_text(c, line, PAD + half_w - 8, y - 16 - i * 12, size=8, color=WHITE)

    # الأعمال المستقبلية
    fx = PAD + half_w + 6
    draw_rect(c, fx, y - conc_h, half_w, conc_h, fill=BG_CARD, radius=8,
              stroke=GREEN, stroke_w=0.8)

    future_title_y = y - 10
    draw_text(c, "أعمال مستقبلية", fx + half_w - 8, future_title_y,
              font="ArabicBold", size=9, color=GREEN)
    future = [
        "• دعم الكاميرات اللحظية (Live Stream)",
        "• تكامل مع GPS حقيقي للمركبات",
        "• نشر النظام كـ API سحابي",
    ]
    for i, line in enumerate(future):
        draw_text(c, line, fx + half_w - 8, y - 24 - i * 13, size=8, color=WHITE)

    y -= conc_h + 10

    # ══ فوتر ══════════════════════════════════════
    # خط فاصل
    c.saveState()
    c.setStrokeColor(colors.HexColor("#334155"))
    c.setLineWidth(0.5)
    c.line(PAD, y - 2, W - PAD, y - 2)
    c.restoreState()

    draw_text(c, "نظام الرصد الذكي للطرق — مشروع دورة رؤية الحاسب للمطورين  •  ٢٠٢٦",
              W/2, y - 12, font="Arabic", size=7, color=MUTED, align="center")

    # ── شريط سفلي ──
    c.saveState()
    c.setFillColor(ACCENT)
    c.rect(0, 0, W, 3, fill=1, stroke=0)
    c.restoreState()

    c.save()
    print(f"✅ تم إنشاء الملف: {OUT}")


if __name__ == "__main__":
    build()
