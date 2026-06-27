"""
نظام إشعارات الحوادث
  - إرسال بريد إلكتروني للإسعاف والمرور
  - في حالة عدم ضبط البريد الحقيقي يعمل بوضع المحاكاة (MOCK)
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import config


def _is_mock_mode() -> bool:
    return (
        config.EMAIL_SENDER == "your_email@gmail.com"
        or not config.EMAIL_PASSWORD
        or config.EMAIL_PASSWORD == "your_app_password"
    )


def _build_email_body(lat, lon, location_name, timestamp) -> str:
    maps_link = f"https://www.google.com/maps?q={lat},{lon}"
    return f"""
<div dir="rtl" style="font-family:Arial;font-size:15px;line-height:2;
     max-width:600px;margin:auto;border:1px solid #ddd;
     border-radius:10px;padding:24px">

  <h2 style="color:#cc0000;border-bottom:2px solid #cc0000;padding-bottom:8px">
    🚨 تنبيه عاجل: حادث مروري
  </h2>

  <table style="width:100%;border-collapse:collapse">
    <tr><td style="padding:6px 0"><b>📍 الموقع</b></td>
        <td>{location_name}</td></tr>
    <tr><td style="padding:6px 0"><b>🌐 الإحداثيات</b></td>
        <td>{lat:.6f}°N, {lon:.6f}°E</td></tr>
    <tr><td style="padding:6px 0"><b>🕐 وقت الرصد</b></td>
        <td>{timestamp}</td></tr>
  </table>

  <br>
  <a href="{maps_link}"
     style="background:#cc0000;color:white;padding:10px 20px;
            border-radius:6px;text-decoration:none;font-weight:bold">
    📌 افتح الموقع في Google Maps
  </a>

  <br><br>
  <p style="color:#888;font-size:12px">
    تم الإرسال تلقائياً بواسطة نظام رصد الحوادث بالداش كام.<br>
    الرجاء التحرك الفوري للموقع المحدد.
  </p>
</div>"""


def _send_single_email(to: str, subject: str, body: str):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = config.EMAIL_SENDER
    msg["To"]      = to
    msg.attach(MIMEText(body, "html", "utf-8"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as server:
            server.login(config.EMAIL_SENDER, config.EMAIL_PASSWORD)
            server.sendmail(config.EMAIL_SENDER, to, msg.as_string())
        return True
    except Exception as e:
        print(f"     ⚠️ خطأ في الإرسال: {e}")
        return False


def send_alert(lat: float, lon: float, location_name: str, timestamp: str, map_path: str = ""):
    subject = "🚨 تنبيه عاجل: حادث مروري رُصد تلقائياً"
    body    = _build_email_body(lat, lon, location_name, timestamp)
    mock    = _is_mock_mode()

    print("\n📧 إرسال الإشعارات...")

    targets = {
        "الإسعاف":       config.RECIPIENTS["ambulance"],
        "إدارة المرور":  config.RECIPIENTS["traffic"],
    }

    for name, email in targets.items():
        if mock or "mock" in email:
            print(f"   [MOCK] ✅ إشعار الـ{name} — {email}")
            print(f"          الموضوع: {subject}")
            print(f"          الموقع:  {location_name} ({lat:.4f}, {lon:.4f})")
        else:
            ok = _send_single_email(email, f"{subject} ({name})", body)
            status = "✅ أُرسل" if ok else "❌ فشل"
            print(f"   {status} — {name} ({email})")

    print("📧 اكتمل إرسال جميع الإشعارات\n")
