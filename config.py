from pathlib import Path

# ───── Paths ─────
BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
VIDEOS_DIR = BASE_DIR / "videos"
OUTPUT_DIR = BASE_DIR / "output"

for _d in [DATA_DIR, MODELS_DIR, VIDEOS_DIR, OUTPUT_DIR]:
    _d.mkdir(exist_ok=True)

# ───── YOLO Training ─────
BASE_MODEL        = "yolov8s.pt"   # YOLOv8-small: توازن جيد بين الدقة والسرعة
EPOCHS            = 30
IMAGE_SIZE        = 640
BATCH_SIZE        = 8
CONF_THRESHOLD    = 0.45           # عتبة الثقة للكشف
FRAMES_TO_CONFIRM = 5              # عدد الإطارات المتتالية لتأكيد الحادث قبل الإنذار

# ───── Roboflow Dataset ─────
# 1. سجّل مجاناً على roboflow.com
# 2. اذهب إلى: Account Settings → API Key
# 3. ابحث في universe.roboflow.com عن "accident detection" واختر dataset
# 4. افتح الـ dataset → Versions → Export → YOLOv8 → Show Download Code
ROBOFLOW_API_KEY   = "UzmR4p7roKdyscn5K16N"
ROBOFLOW_WORKSPACE = "victor-bxgzd"
ROBOFLOW_PROJECT   = "accident-detection-uevan"
ROBOFLOW_VERSION   = 2

# ───── GPS (محاكاة للعرض التوضيحي) ─────
# غيّر هذه الإحداثيات لموقع فعلي إذا أردت
DEMO_GPS_LAT  = 24.7136
DEMO_GPS_LON  = 46.6753
LOCATION_NAME = "طريق الملك فهد، الرياض، المملكة العربية السعودية"
ALERT_RADIUS_KM = 1.0

# ───── إشعارات البريد الإلكتروني ─────
# استخدم Gmail App Password من: myaccount.google.com → Security → App Passwords
# اتركها كما هي للعمل بوضع المحاكاة (MOCK)
EMAIL_SENDER   = "your_email@gmail.com"
EMAIL_PASSWORD = "your_app_password"
RECIPIENTS = {
    "ambulance": "ambulance@mock.com",
    "traffic":   "traffic@mock.com",
}
