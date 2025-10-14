import os, json, threading, time, re
import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont

# ============== Optional libs ==============
ROBOWFLOW_ENABLED = True
try:
    from inference_sdk import InferenceHTTPClient
except Exception:
    ROBOWFLOW_ENABLED = False

try:
    from pynput import mouse, keyboard
    PYNPUT_OK = True
except Exception:
    PYNPUT_OK = False

TESS_OK = False
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:/Users/user/AppData/Local/Programs/Tesseract-OCR/tesseract.exe"
    print("✅ pytesseract version:", pytesseract.get_tesseract_version())
    TESS_OK = True
except Exception as e:
    print("❌ pytesseract import failed:", e)
    TESS_OK = False

# ============== Paths & assets ==============
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
MACRO_DIR  = os.path.join(ASSETS_DIR, "macros")
CONFIG_DIR = os.path.join(ASSETS_DIR, "configs")
os.makedirs(MACRO_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

MACRO_FILE = os.path.join(MACRO_DIR, "main_macro.json")
ROI_FILE   = os.path.join(CONFIG_DIR, "roi_config.json")
ALERT_IMAGE= os.path.join(ASSETS_DIR, "alert.png")

# ============== Default stream URLs ==============
ROI_STREAM_URL = "rtsp://nthummrl@gmail.com:nthummrl110@140.114.56.28:554/stream1"
OXY_STREAM_URL = "http://140.114.56.28:8080/video?dummy=param.mjpg"

# ============== Globals ==============
monitoring = False
pause_for_alert = False
macro_play_stop = False
recording = False

roi_main = None
roi_trigger = None
oxy_roi = None

_main_stop_event = threading.Event()  # 🛑 主程序 ESC 停止事件

debug_cond_bad = False
debug_cond_high_oxy = False
oxy_threshold = 1.0  # 可自行設定臨界值

# === OXY 前處理參數 ===
oxy_otsu_threshold = 0  # 0 = 自動 OTSU 模式
oxy_brightness = 50   # 0~100, 50 = 原圖
oxy_contrast   = 50   # 0~100, 50 = 原圖
oxy_gamma      = 50   # 0~100, 50 = γ=1.0
oxy_saturation = 50   # 0~100, 50 = 原圖

trigger_update_interval = 0.15
trigger_delay_after_gray = 1.2
g_drop_threshold = 5.0
gray_increase_threshold = 10.0
trigger_mode = "color"
interval_seconds = 2.0
OXY_WARN_IF_EMPTY = True

last_predict_text = "—"
last_predict_conf = 0.0
last_oxy_text     = "—"
last_oxy_ok       = False

SCROLL_SCALE = 120.0  # 🧭 補償倍率，可依實際手感微調

if ROBOWFLOW_ENABLED:
    try:
        client = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key="RnGFo8AzPLZNtcop9YZ0")
        WORKSPACE   = "semantic-segmentation-9o2cy"
        WORKFLOW_ID = "custom-workflow-2"
    except Exception:
        ROBOWFLOW_ENABLED = False

# ============== Utils ==============
def ts(): return time.strftime("%H:%M:%S")
def log(msg):
    print(msg)
    try:
        console.insert(tk.END, f"{ts()} | {msg}\n"); console.see(tk.END)
    except Exception:
        pass
def set_status(active:bool):
    global recording
    recording = active
    if status_label.winfo_exists():
        status_label.config(text=("🟢 Active" if active else "🔴 Idle"),
                            fg=("lime" if active else "red"))
def save_roi_config():
    data = {
        "roi_main": list(roi_main) if roi_main else None,          # 粉床 ROI
        "roi_trigger": list(roi_trigger) if roi_trigger else None,  # 螢幕監聽 ROI
        "oxy_roi": list(oxy_roi) if oxy_roi else None,
        "oxy_otsu_threshold": oxy_otsu_threshold,
        "oxy_brightness": oxy_brightness,
        "oxy_contrast": oxy_contrast,
        "oxy_gamma": oxy_gamma,
        "oxy_saturation": oxy_saturation,
        "roi_stream_url": ROI_STREAM_URL,
        "oxy_stream_url": OXY_STREAM_URL
    }
    json.dump(data, open(ROI_FILE, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    log("✅ ROI/Stream + OXY 設定已儲存")
def load_roi_config():
    global roi_main, roi_trigger, oxy_roi
    global oxy_otsu_threshold, oxy_brightness, oxy_contrast, oxy_gamma, oxy_saturation
    global ROI_STREAM_URL, OXY_STREAM_URL, ROI_USERNAME, ROI_PASSWORD

    if os.path.exists(ROI_FILE):
        d = json.load(open(ROI_FILE, "r", encoding="utf-8"))
        roi_main    = tuple(d.get("roi_main")) if d.get("roi_main") else None
        roi_trigger = tuple(d.get("roi_trigger")) if d.get("roi_trigger") else None
        oxy_roi     = tuple(d.get("oxy_roi")) if d.get("oxy_roi") else None

        oxy_otsu_threshold = int(d.get("oxy_otsu_threshold", 0))
        oxy_brightness     = int(d.get("oxy_brightness", 50))
        oxy_contrast       = int(d.get("oxy_contrast", 50))
        oxy_gamma          = int(d.get("oxy_gamma", 50))
        oxy_saturation     = int(d.get("oxy_saturation", 50))

        ROI_STREAM_URL = d.get("roi_stream_url", ROI_STREAM_URL)
        OXY_STREAM_URL = d.get("oxy_stream_url", OXY_STREAM_URL)
        ROI_USERNAME   = d.get("roi_username", "")
        ROI_PASSWORD   = d.get("roi_password", "")

        log(f"載入設定 | OTSU={oxy_otsu_threshold}, Bright={oxy_brightness}, Contrast={oxy_contrast}, Gamma={oxy_gamma}, Satur={oxy_saturation}")

        # ✅ 若有螢幕監聽 ROI，顯示在螢幕預覽畫面中
        if roi_trigger:
            try:
                import pyautogui
                screenshot = pyautogui.screenshot()
                frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                x, y, w, h = roi_trigger
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 215, 0), 2)
                cv2.putText(frame, "螢幕監聽 ROI", (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 215, 0), 2)
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                tkimg = to_tk(pil, size=(480, 270))
                left_preview.configure(image=tkimg)
                left_preview.image = tkimg
                log(f"🟦 已載入螢幕監聽 ROI 並顯示於畫面：{roi_trigger}")
            except Exception as e:
                log(f"⚠️ 無法更新螢幕監聽 ROI 畫面：{e}")
    else:
        log("（尚未有 ROI/Stream 設定）")

# ============== Image utils ==============
def to_tk(pil_img, size=None):
    if size: pil_img = pil_img.resize(size, Image.LANCZOS)
    return ImageTk.PhotoImage(pil_img)
def gray_frame(w=640, h=480):
    """🩶 Return a plain gray fallback frame."""
    return np.full((h, w, 3), 80, dtype=np.uint8)

# 初始化 ROI buffer
roi_frame_buffer = gray_frame(640, 480)

# ============== Stream wrappers ==============
def init_gray_preview():
    g1 = to_tk(Image.fromarray(gray_frame(480, 270)))
    g2 = to_tk(Image.fromarray(gray_frame(480, 100)))
    left_preview.configure(image=g1); left_preview.image = g1
    right_preview.configure(image=g2); right_preview.image = g2
    log("🩶 初始灰底畫面已載入")
def read_stream(url):
    """
    永遠回傳一張畫面 (串流失敗時為灰底)
    強制使用 TCP 傳輸並支援 @ → %40 編碼。
    """
    try:
        # 自動把帳號中的 @ 轉成 %40
        if "@" in url.split("://",1)[-1].split("@")[0]:
            parts = url.split("://", 1)
            prefix = parts[0] + "://"
            body = parts[1]
            user_part, rest = body.split("@", 1)
            if "@" in user_part:
                user_part = user_part.replace("@", "%40")
            url = prefix + user_part + "@" + rest
            log(f"🔧 已自動修正 RTSP URL: {url}")

        # 強制用 FFmpeg backend (TCP)
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)

        time.sleep(0.1)
        ok, frame = cap.read()
        cap.release()

        if not ok or frame is None:
            log("⚠️ RTSP 連線成功但無畫面，改回灰底")
            frame = gray_frame()
        return frame

    except Exception as e:
        log(f"❌ RTSP 讀取錯誤: {e}")
        return gray_frame()
def get_roi_frame():
    try:
        return read_stream(ROI_STREAM_URL, ROI_USERNAME, ROI_PASSWORD)
    except:
        return gray_frame()
def get_oxy_frame():
    try:
        frame = read_stream(OXY_STREAM_URL)
        if frame is None or frame.size == 0:
            return gray_frame(480, 270)
        return frame
    except:
        return gray_frame(480, 270)
# ============== Visual overlay ==============
def overlay_powbed(frame_bgr, pred_text=None, pred_conf=None):
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    try: font = ImageFont.truetype("arial.ttf", 60)
    except: font = ImageFont.load_default()

    if roi_main:
        x,y,w,h = roi_main
        draw.rectangle([x, y, x+w, y+h], outline=(0,255,0), width=15)
        draw.text((x, max(0,y-80)), "PowBed ROI", fill=(0,255,0), font=font)
    return pil

# ============== ROI Trigger & Inference ==============
def detect_green_to_gray(prev_img, curr_img, g_drop=5, gray_increase=10):
    prev_mean = np.mean(prev_img, axis=(0,1))
    curr_mean = np.mean(curr_img, axis=(0,1))
    delta_g = curr_mean[1] - prev_mean[1]
    prev_gray = np.mean(cv2.cvtColor(prev_img.astype(np.uint8), cv2.COLOR_BGR2GRAY))
    curr_gray = np.mean(cv2.cvtColor(curr_img.astype(np.uint8), cv2.COLOR_BGR2GRAY))
    delta_gray = curr_gray - prev_gray
    trigger = (delta_g > g_drop) and (delta_gray > gray_increase)
    return trigger, float(delta_g), float(delta_gray)
import tempfile
def do_inference_on_roi_frame(frame_bgr):
    """
    在 ROI 區域進行推論，並於 App 介面下方更新辨識結果 Label。
    不在畫面上疊文字。
    """
    global last_predict_text, last_predict_conf

    if roi_main is None:
        log("⚠️ 尚未設定 ROI_Main，無法推論")
        return None, None

    x, y, w, h = roi_main
    crop = frame_bgr[y:y+h, x:x+w].copy()

    # ✅ 建立臨時英文安全路徑
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg", prefix="roi_tmp_", dir=None)
    os.close(tmp_fd)  # 關閉檔案描述符，讓 cv2 可寫入
    try:
        ok = cv2.imwrite(tmp_path, crop)
        if not ok:
            log(f"❌ 無法寫入暫存圖檔: {tmp_path}")
            return None, None

        if ROBOWFLOW_ENABLED:
            try:
                # ✅ Roboflow 只接受「檔案路徑」或 URL
                result = client.run_workflow(
                    workspace_name=WORKSPACE,
                    workflow_id=WORKFLOW_ID,
                    images={"image": tmp_path},
                    use_cache=True
                )

                pred_class = result[0]["predictions"]["top"]
                conf = float(result[0]["predictions"]["confidence"])
                last_predict_text, last_predict_conf = pred_class, conf
                log(f"[ROI] Prediction: {pred_class} ({conf:.3f})")

                # 更新顯示結果
                color = "lime" if pred_class.lower() == "good" else "orange"
                roi_result_label.config(
                    text=f"辨識結果：{pred_class} ({conf:.2f})",
                    fg=color
                )

                # 成功推論後才更新 powder_bed_roi 預覽畫面
                roi_crop = crop.copy()
                tkroi = to_tk(Image.fromarray(cv2.cvtColor(roi_crop, cv2.COLOR_BGR2RGB)), size=(360, 200))
                root.after(0, lambda img=tkroi: powder_bed_roi_preview.configure(image=img))
                root.after(0, lambda img=tkroi: setattr(powder_bed_roi_preview, "image", img))

                # 若非 good 則觸發警示
                if pred_class.lower() != "good":
                    handle_emergency("ROI_BAD")

                return pred_class, conf

            except Exception as e:
                log(f"❌ Inference error: {e}")

        else:
            last_predict_text, last_predict_conf = "mock_good", 0.99
            roi_result_label.config(text="辨識結果：mock_good (0.99)", fg="lime")
            return "good", 0.99

    finally:
        # ✅ 刪除臨時檔案
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                # log("🧹 已清理暫存檔")
            except Exception as e:
                log(f"⚠️ 無法刪除暫存檔: {e}")

    return None, None

# ============== Emergency handling ==============
alert_win = None
def show_alert(msg="⚠️ 異常偵測"):
    global alert_win, pause_for_alert
    if alert_win is not None: return
    pause_for_alert = True
    alert_win = tk.Toplevel(root)
    alert_win.attributes("-fullscreen", True)
    alert_win.configure(bg="black")
    if os.path.exists(ALERT_IMAGE):
        img = Image.open(ALERT_IMAGE)
        sw, sh = alert_win.winfo_screenwidth(), alert_win.winfo_screenheight()
        img = img.resize((sw, sh), Image.LANCZOS)
        tkimg = ImageTk.PhotoImage(img)
        lbl = tk.Label(alert_win, image=tkimg, bg="black")
        lbl.image = tkimg
        lbl.pack(fill="both", expand=True)
    else:
        tk.Label(alert_win, text=msg, fg="red", bg="black", font=("Arial", 80, "bold")).pack(expand=True)
    alert_win.bind("<Button-1>", lambda e: close_alert())
def close_alert():
    global alert_win, pause_for_alert
    if alert_win is not None:
        alert_win.destroy()
        alert_win = None
    pause_for_alert = False
def handle_emergency(source="ROI_BAD"):
    log(f"🛑 Emergency from {source}")
    stop_macro_play()
    stop_all_monitoring(silent=True)
    show_alert(f"⚠️ 異常：{source}")

# ============== Monitor & Preview Loops ==============
oxy_frame_buffer = gray_frame(480, 100)     # 🔹 OXY 畫面緩衝區（共享）
def powbed_monitor_loop():
    """🔶 專職 ROI 觸發監測與推論（不更新畫面）"""
    prev_trig = None
    while monitoring and not _main_stop_event.is_set():
        frame = get_roi_frame()
        if frame is None or frame.size == 0:
            time.sleep(0.5)
            continue

        global roi_frame_buffer
        roi_frame_buffer = frame.copy()  # ✅ 更新最新 buffer 給 preview loop 用

        # === Trigger ROI 檢測 ===
        if roi_trigger is not None:
            x, y, w, h = roi_trigger
            trig = frame[y:y+h, x:x+w].copy()

            if prev_trig is not None:
                triggered, dG, dGray = detect_green_to_gray(
                    prev_trig, trig, g_drop_threshold, gray_increase_threshold
                )
                if triggered:
                    log(f"[ROI Trigger] 綠→灰觸發 ΔG={dG:.1f}, ΔGray={dGray:.1f}")
                    time.sleep(trigger_delay_after_gray)
                    do_inference_on_roi_frame(frame)

            prev_trig = trig

        time.sleep(trigger_update_interval)
def oxy_monitor_loop():
    """🧠 OXY 背景 OCR 偵測（永遠運行，無論 monitoring 狀態）"""
    global oxy_frame_buffer, last_oxy_text, last_oxy_ok, debug_cond_high_oxy

    last_oxy_value = None
    log("🧠 OXY 偵測執行中（持續運行模式）")

    while True and not _main_stop_event.is_set():
        try:
            frame = oxy_frame_buffer.copy()
            if frame is None or frame.size == 0:
                time.sleep(0.3)
                continue

            # === 前處理 ===
            img = frame.astype(np.float32)
            brightness = (oxy_brightness - 50) * 2.5
            contrast = (oxy_contrast / 50.0)
            img = np.clip((img - 128) * contrast + 128 + brightness, 0, 255).astype(np.uint8)

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 1] *= (oxy_saturation / 50.0)
            hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gamma = oxy_gamma / 50.0
            gray = np.uint8(np.clip(np.power(gray / 255.0, 1.0 / gamma) * 255, 0, 255))

            if oxy_otsu_threshold <= 0:
                _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                _, th = cv2.threshold(gray, oxy_otsu_threshold, 255, cv2.THRESH_BINARY)

            # === OCR 辨識 ===
            raw_text = pytesseract.image_to_string(
                th, config="--psm 7 -c tessedit_char_whitelist=0123456789."
            ).strip()
            match = re.findall(r"[0-9.]+", raw_text)
            text = match[0] if match else ""

            # === 更新顯示（即使 monitoring=False 也更新）===
            if text and text != last_oxy_value:
                last_oxy_value = text
                last_oxy_text = text
                root.after(0, lambda t=text: oxy_value_label.config(
                    text=f"OCR 結果：{t}"
                ))

                try:
                    val_num = float(text)
                    last_oxy_ok = val_num <= oxy_threshold
                    if val_num > oxy_threshold:
                        debug_cond_high_oxy = True
                        log(f"🧨 OXY 值過高：{val_num} > {oxy_threshold}")
                        if monitoring:  # 只有在主程序執行時才觸發緊急停止
                            handle_emergency("OXY HIGH")
                except ValueError:
                    last_oxy_ok = False

        except Exception as e:
            log(f"⚠️ OXY OCR 錯誤: {e}")

        time.sleep(0.5)

def powbed_preview_loop():
    """🖼 專職 ROI 畫面顯示更新（FFmpeg + thread-safe GUI 更新）"""
    global roi_frame_buffer
    url = ROI_STREAM_URL
    cap = None
    reconnecting = False

    log("📡 ROI 預覽執行中（FFMPEG backend）（常駐）")

    while True:
        try:
            # === 重新連線處理 ===
            if cap is None or not cap.isOpened():
                if not reconnecting:
                    reconnecting = True
                    log("🔴 ROI 串流中斷，重新連線中...")
                    gray = gray_frame(640, 480)
                    pil = Image.fromarray(gray)
                    draw = ImageDraw.Draw(pil)
                    font = ImageFont.truetype("arial.ttf", 48)
                    draw.text((60, 230), "PowBed Reconnecting...", fill=(255, 0, 0), font=font)
                    tkimg = to_tk(pil, size=(480, 270))
                    root.after(0, lambda img=tkimg: left_preview.configure(image=img))
                    root.after(0, lambda img=tkimg: setattr(left_preview, "image", img))
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                time.sleep(2)
                continue

            # === 正常取 frame ===
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                cap = None
                time.sleep(1)
                continue

            reconnecting = False
            roi_frame_buffer = frame.copy()

            # === 主畫面更新 ===
            pil = overlay_powbed(frame, last_predict_text, last_predict_conf)
            tkimg = to_tk(pil, size=(480, 270))
            root.after(0, lambda img=tkimg: left_preview.configure(image=img))
            root.after(0, lambda img=tkimg: setattr(left_preview, "image", img))

            time.sleep(0.05)

        except Exception as e:
            log(f"⚠️ 粉床預覽錯誤：{e}")
            cap = None
            time.sleep(1)
def oxy_preview_loop():
    """🖼 OXY 畫面預覽（只負責顯示，不做 OCR）"""
    global oxy_frame_buffer, OXY_STREAM_URL, oxy_roi
    url = OXY_STREAM_URL
    cap = None
    reconnecting = False

    log(f"📡 啟動 OXY 預覽（FFMPEG backend）（常駐）: {url}")

    while True:
        try:
            # === 若尚未開啟或中斷 → 重新連線 ===
            if cap is None or not cap.isOpened():
                if not reconnecting:
                    reconnecting = True
                    log("🟡 OXY 串流中斷，重新連線中...")
                    gray = gray_frame(480, 100)
                    pil = Image.fromarray(gray)
                    draw = ImageDraw.Draw(pil)
                    font = ImageFont.truetype("arial.ttf", 30)
                    draw.text((40, 40), "OXY Reconnecting...", fill=(255, 255, 0), font=font)
                    tkimg = to_tk(pil)
                    root.after(0, lambda img=tkimg: right_preview.configure(image=img))
                    root.after(0, lambda img=tkimg: setattr(right_preview, "image", img))

                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
                time.sleep(1)
                continue

            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                log("⚠️ OXY 無法讀取 frame，嘗試重連...")
                cap.release()
                cap = None
                time.sleep(1)
                continue

            reconnecting = False

            # === ROI 裁切（防呆）===
            if oxy_roi and frame is not None and frame.size > 0:
                x, y, w, h = oxy_roi
                h_max, w_max = frame.shape[:2]
                x = max(0, min(x, w_max - 1))
                y = max(0, min(y, h_max - 1))
                w = min(w, w_max - x)
                h = min(h, h_max - y)
                frame = frame[y:y+h, x:x+w].copy()

            # === 更新共享 buffer ===
            oxy_frame_buffer = frame.copy()

            # === 顯示畫面（thread-safe）===
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            tkimg = to_tk(pil)
            root.after(0, lambda img=tkimg: right_preview.configure(image=img))
            root.after(0, lambda img=tkimg: setattr(right_preview, "image", img))

            time.sleep(0.05)

        except Exception as e:
            log(f"❌ OXY 預覽錯誤: {e}")
            if cap:
                cap.release()
            cap = None
            time.sleep(1)

    if cap:
        cap.release()
    log("🟥 OXY 預覽結束")
def screen_roi_preview_loop():
    """🖥 永久運行的螢幕監聽 ROI 即時預覽"""
    import pyautogui
    global roi_trigger

    log("📺 螢幕監聽 ROI 預覽執行中（持續更新）")

    while True:
        try:
            if roi_trigger is None:
                # 沒設定 ROI，就顯示提示畫面
                gray = gray_frame(200, 120)
                pil = Image.fromarray(gray)
                draw = ImageDraw.Draw(pil)
                draw.text((30, 50), "No Screen ROI", fill=(255, 255, 0))
                tkimg = to_tk(pil, size=(120, 120))
                root.after(0, lambda img=tkimg: trigger_roi_preview.configure(image=img))
                root.after(0, lambda img=tkimg: setattr(trigger_roi_preview, "image", img))
                time.sleep(1.0)
                continue

            # 擷取整個螢幕畫面
            screenshot = pyautogui.screenshot()
            frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

            # 確保 ROI 合法範圍
            x, y, w, h = roi_trigger
            h_max, w_max = frame.shape[:2]
            x = max(0, min(x, w_max - 1))
            y = max(0, min(y, h_max - 1))
            w = min(w, w_max - x)
            h = min(h, h_max - y)

            roi_crop = frame[y:y+h, x:x+w].copy()

            # 若 ROI 區域過小（例如 0x0），跳過
            if roi_crop.size == 0:
                time.sleep(1.0)
                continue

            # 轉換成 Tk 影像顯示
            pil = Image.fromarray(cv2.cvtColor(roi_crop, cv2.COLOR_BGR2RGB))
            tkimg = to_tk(pil, size=(120, 120))
            root.after(0, lambda img=tkimg: trigger_roi_preview.configure(image=img))
            root.after(0, lambda img=tkimg: setattr(trigger_roi_preview, "image", img))

        except Exception as e:
            log(f"⚠️ 螢幕 ROI 預覽錯誤: {e}")
            gray = gray_frame(200, 120)
            tkimg = to_tk(Image.fromarray(gray), size=(120, 120))
            root.after(0, lambda img=tkimg: trigger_roi_preview.configure(image=img))
            root.after(0, lambda img=tkimg: setattr(trigger_roi_preview, "image", img))
            time.sleep(2)

        time.sleep(0.5)

# ============== Macro (Enhanced Loop + Scroll Support + ESC Safety) ==============
import ctypes
import threading

macro_events = []
macro_loop_delay = 3.0  # 🕒 每輪播放間隔秒數
_macro_stop_event = threading.Event()
_macro_thread = None
_macro_lock = threading.Lock()  # 保護避免多重啟動

def _normalize_key_name(k_str: str):
    """將 pynput 記錄的 key 字串轉為 pyautogui 可處理名稱。"""
    k_str = k_str.replace("'", "").strip()
    if len(k_str) == 1:
        return k_str, True
    mapping = {
        "Key.enter": "enter", "Key.tab": "tab", "Key.backspace": "backspace",
        "Key.delete": "delete", "Key.space": "space", "Key.esc": "esc", "Key.escape": "esc",
        "Key.up": "up", "Key.down": "down", "Key.left": "left", "Key.right": "right",
        "Key.home": "home", "Key.end": "end", "Key.page_up": "pageup", "Key.page_down": "pagedown",
        "Key.shift": "shift", "Key.ctrl": "ctrl", "Key.alt": "alt", "Key.cmd": "win",
        "Key.caps_lock": "capslock", "Key.print_screen": "printscreen", "Key.num_lock": "numlock",
        "Key.scroll_lock": "scrolllock",
        "Key.f1": "f1", "Key.f2": "f2", "Key.f3": "f3", "Key.f4": "f4", "Key.f5": "f5",
        "Key.f6": "f6", "Key.f7": "f7", "Key.f8": "f8", "Key.f9": "f9", "Key.f10": "f10",
        "Key.f11": "f11", "Key.f12": "f12",
    }
    if k_str in mapping:
        return mapping[k_str], False
    return None, False
def record_main_macro():
    """開始錄製（按 ESC 結束），輸出至 MACRO_FILE。"""
    if not PYNPUT_OK:
        messagebox.showwarning("提示", "未安裝 pynput。")
        return

    def _record_thread():
        global macro_events
        macro_events = []
        set_status(True)
        log("🎬 開始錄製（按 ESC 結束）")

        start = time.time()

        def on_click(x, y, btn, pressed):
            macro_events.append({
                "t": time.time() - start,
                "type": "click",
                "x": int(x), "y": int(y),
                "btn": str(btn),
                "pressed": bool(pressed)
            })

        def on_scroll(x, y, dx, dy):
            macro_events.append({
                "t": time.time() - start,
                "type": "scroll",
                "x": int(x), "y": int(y),
                "dx": int(dx), "dy": int(dy)
            })

        def on_key(k):
            if k == keyboard.Key.esc:
                return False
            macro_events.append({
                "t": time.time() - start,
                "type": "key",
                "key": str(k)
            })

        ml = mouse.Listener(on_click=on_click, on_scroll=on_scroll)
        kl = keyboard.Listener(on_press=on_key)
        ml.start()
        kl.start()
        kl.join()
        ml.stop()

        json.dump(macro_events, open(MACRO_FILE, "w", encoding="utf-8"),
                  indent=2, ensure_ascii=False)
        set_status(False)
        log("✅ 錄製完成，已儲存巨集事件")

    threading.Thread(target=_record_thread, daemon=True).start()

# def _esc_safety_listener():
#     """監聽 ESC 鍵，作為保險開關停止巨集。"""
#     from pynput import keyboard
#     def on_press(key):
#         if key == keyboard.Key.esc:
#             stop_macro_play()
#             return False
#     try:
#         with keyboard.Listener(on_press=on_press) as listener:
#             listener.join()
#     except Exception:
#         pass

def play_main_macro():
    """無限循環播放巨集，按 ESC 停止。"""
    global _macro_thread

    with _macro_lock:
        if _macro_thread and _macro_thread.is_alive():
            log("ℹ️ 巨集已在播放中，忽略本次啟動請求")
            return

        if not os.path.exists(MACRO_FILE):
            messagebox.showwarning("提示", "沒有可播放的巨集。")
            return

        try:
            import pyautogui
        except Exception as e:
            log(f"❌ 缺少 pyautogui，無法播放巨集：{e}")
            return

        try:
            with open(MACRO_FILE, "r", encoding="utf-8") as f:
                events = json.load(f)
        except Exception as e:
            log(f"❌ 讀取巨集檔失敗：{e}")
            return

        _macro_stop_event.clear()
        set_status(True)
        try:
            status_label.config(text="🔵 巨集執行中（按 ESC 停止）", fg="cyan")
        except Exception:
            pass

        def _run():
            try:
                log(f"▶ 無限播放巨集，間隔 {macro_loop_delay:.1f} 秒")
                # 啟動安全監聽 ESC
                threading.Thread(target=_esc_safety_main_listener, daemon=True).start()

                while not _macro_stop_event.is_set():
                    t0 = time.time()
                    for e in events:
                        if _macro_stop_event.is_set():
                            break

                        delay = e.get("t", 0) - (time.time() - t0)
                        if delay > 0:
                            waited = 0.0
                            while waited < delay and not _macro_stop_event.is_set():
                                time.sleep(min(0.01, delay - waited))
                                waited += 0.01
                            if _macro_stop_event.is_set():
                                break

                        etype = e.get("type")
                        try:
                            if etype == "click":
                                x, y = int(e.get("x", 0)), int(e.get("y", 0))
                                pressed = bool(e.get("pressed", True))
                                if pressed:
                                    pyautogui.mouseDown(x, y)
                                else:
                                    pyautogui.mouseUp(x, y)

                            elif etype == "scroll":
                                dy = int(e.get("dy", 0))
                                pyautogui.scroll(int(dy * SCROLL_SCALE))

                            elif etype == "key":
                                key_raw = e.get("key", "")
                                key_name, is_text = _normalize_key_name(key_raw)
                                if not key_name:
                                    continue
                                if is_text:
                                    pyautogui.typewrite(key_name)
                                else:
                                    pyautogui.press(key_name)

                        except Exception as ie:
                            log(f"⚠️ 巨集事件執行失敗：{ie}")

                    if _macro_stop_event.is_set():
                        break

                    log(f"⏸ 等待 {macro_loop_delay:.1f}s 後重播")
                    waited = 0.0
                    while waited < macro_loop_delay and not _macro_stop_event.is_set():
                        time.sleep(min(0.05, macro_loop_delay - waited))
                        waited += 0.05

            finally:
                stop_macro_play(force=True)

        _macro_thread = threading.Thread(target=_run, daemon=True, name="macro_player")
        _macro_thread.start()
def stop_macro_play(force=False):
    """停止巨集播放（支援外部 ESC 停止）。"""
    _macro_stop_event.set()
    try:
        if force:
            log("🟥 巨集播放結束")
        else:
            log("🟥 使用者停止巨集")
        set_status(False)
        status_label.config(text="🔴 Idle", fg="red")
    except Exception:
        pass
def set_macro_delay():
    """設定播放間隔。"""
    global macro_loop_delay
    try:
        val = tk.simpledialog.askfloat("設定播放間隔", "請輸入每次巨集播放間隔（秒）",
                                       minvalue=0.2, initialvalue=macro_loop_delay)
        if val is not None:
            macro_loop_delay = float(val)
            log(f"⚙️ 已設定播放間隔：{macro_loop_delay:.1f} 秒")
    except Exception as e:
        log(f"⚠️ 設定播放間隔失敗：{e}")
# ============== End Macro =====================================================

# ============== Start/Stop ==============
def start_all():
    global monitoring
    if monitoring:
        return
    monitoring = True
    _main_stop_event.clear()
    set_status(True)

    # 狀態提示顯示 ESC 停止說明
    try:
        status_label.config(text="🔵 主程序執行中（按 ESC 停止）", fg="cyan")
    except Exception:
        pass

    # 啟動 ESC 安全監聽
    threading.Thread(target=_esc_safety_main_listener, daemon=True).start()

    # --- 分開職責 ---
    threading.Thread(target=powbed_monitor_loop, daemon=True).start()
    threading.Thread(target=oxy_monitor_loop, daemon=True).start()
    play_main_macro()

    log("✅ 開始執行（ROI + OXY 分工完成）")
def stop_all_monitoring(silent=False):
    global monitoring
    monitoring = False; set_status(False)
    if not silent: log("🟥 停止監測")
def stop_all():
    global _main_stop_event
    stop_macro_play()
    stop_all_monitoring()
    close_alert()
    _main_stop_event.set()
    try:
        status_label.config(text="🔴 Idle", fg="red")
    except Exception:
        pass
    log("🟥 主程序已結束（含巨集與監測）")
def select_roi(which="main"):
    """開啟目前 ROI buffer 讓使用者框選 ROI"""
    global roi_frame_buffer, roi_main, roi_trigger
    log(f"🟩 開始選取 ROI ({which})")

    # 確保 buffer 存在
    if roi_frame_buffer is None or roi_frame_buffer.size == 0:
        roi_frame_buffer = gray_frame(640, 480)

    frame = roi_frame_buffer.copy()  # ✅ 使用目前的 buffer 畫面

    # 使用 OpenCV 提供的 ROI 選取工具
    from_center = False
    roi = cv2.selectROI(f"Select ROI - {which}", frame, from_center)
    cv2.destroyAllWindows()

    if roi == (0, 0, 0, 0):
        log("❌ 未選取任何 ROI")
        return

    x, y, w, h = map(int, roi)
    if which == "main":
        roi_main = (x, y, w, h)
        log(f"✅ 設定 ROI_Main = {roi_main}")
    else:
        roi_trigger = (x, y, w, h)
        log(f"✅ 設定 ROI_Trigger = {roi_trigger}")

    save_roi_config()
    log("💾 ROI 設定已儲存")

    # 更新預覽畫面顯示新的 ROI（即使沒在監測）
    try:
        frame = roi_frame_buffer.copy()
        pil = overlay_powbed(frame)
        tkimg = to_tk(pil, size=(480, 270))
        left_preview.configure(image=tkimg)
        left_preview.image = tkimg
        log("🟩 ROI 已顯示於預覽畫面")
    except Exception as e:
        log(f"⚠️ 無法更新預覽畫面：{e}")
def select_screen_roi():
    """
    🖥 從螢幕截圖中選取「螢幕監聽 ROI」
    使用 roi_trigger 作為儲存欄位，以沿用原有結構。
    """
    import pyautogui
    global roi_trigger

    log("🖥 開始選取螢幕監聽 ROI")

    try:
        # 1️⃣ 擷取整個螢幕畫面
        screenshot = pyautogui.screenshot()
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

        # 2️⃣ 開啟 OpenCV 選取工具
        roi = cv2.selectROI("Select Screen ROI", frame, showCrosshair=True)
        cv2.destroyAllWindows()

        if roi == (0, 0, 0, 0):
            log("❌ 未選取任何螢幕區域")
            return

        roi_trigger = tuple(map(int, roi))
        log(f"✅ 設定螢幕監聽 ROI = {roi_trigger}")

        # 3️⃣ 儲存到現有設定檔
        save_roi_config()

        # 4️⃣ 更新主畫面預覽（僅標示範圍）
        try:
            pil = overlay_powbed(roi_frame_buffer, last_predict_text, last_predict_conf)
            tkimg = to_tk(pil, size=(480, 270))
            left_preview.configure(image=tkimg)
            left_preview.image = tkimg
            log("🟦 螢幕監聽 ROI 已顯示於預覽畫面")
        except Exception as e:
            log(f"⚠️ 無法更新預覽畫面：{e}")

    except Exception as e:
        log(f"❌ 螢幕 ROI 選取錯誤: {e}")
def select_oxy_roi():
    """開啟目前 OXY 畫面讓使用者框選顯示範圍"""
    global oxy_roi
    log("🟦 開始選取 Oxygen ROI")

    frame = get_oxy_frame()
    if frame is None or frame.size == 0:
        log("❌ 無法取得 OXY 畫面")
        return

    try:
        # ✅ 移除 from_center 參數，以相容 OpenCV 4.10+
        roi = cv2.selectROI("Select OXY ROI", frame)
        cv2.destroyAllWindows()

        if roi == (0, 0, 0, 0):
            log("❌ 未選取任何 ROI")
            return

        oxy_roi = tuple(map(int, roi))
        save_roi_config()
        log(f"✅ 設定 Oxygen ROI = {oxy_roi}")

    except Exception as e:
        log(f"❌ OXY ROI 選取錯誤: {e}")

# ============== ESC 監聽 ==============
def _esc_safety_main_listener():
    """統一 ESC 鍵監聽：僅啟動一次，用於停止主程序與巨集。"""
    from pynput import keyboard
    global _esc_listener_started

    # 若已有 listener 在跑，就直接返回避免重複監聽
    if getattr(_esc_safety_main_listener, "_running", False):
        return
    _esc_safety_main_listener._running = True

    def on_press(key):
        if key == keyboard.Key.esc:
            log("🛑 按下 ESC — 停止所有程序（主程序 + 巨集）")
            stop_all()
            _main_stop_event.set()
            # 監聽完畢後清除旗標
            _esc_safety_main_listener._running = False
            return False

    try:
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()
    except Exception:
        _esc_safety_main_listener._running = False

# ============== Debug tools ==============
def manual_predict_once():
        """手動觸發一次 ROI Predict（使用目前 ROI 畫面）"""
        global roi_frame_buffer
        if roi_frame_buffer is None or roi_frame_buffer.size == 0:
            log("⚠️ ROI buffer 為空，請確認串流畫面是否啟動")
            return
        log("🧠 手動觸發一次 Predict")
        do_inference_on_roi_frame(roi_frame_buffer.copy())
def debug_oxy_preprocess_otsu():
    """
    🧪 進階版：視覺化 OXY 前處理 + OTSU 二值化效果
    可調亮度、對比、飽和、Gamma、閾值，並儲存設定
    （與 oxy_preview_loop() 使用的處理邏輯完全一致）
    """
    global oxy_otsu_threshold, oxy_brightness, oxy_contrast, oxy_gamma, oxy_saturation
    log("🧪 開啟 OXY Preprocess + OTSU Debug 工具")

    frame = get_oxy_frame()
    if frame is None or frame.size == 0:
        log("❌ 無法取得 OXY 畫面")
        return

    # ✅ 僅顯示 ROI 區域（防呆）
    if oxy_roi and frame.size > 0:
        x, y, w, h = oxy_roi
        h_max, w_max = frame.shape[:2]
        x = max(0, min(x, w_max - 1))
        y = max(0, min(y, h_max - 1))
        w = min(w, w_max - x)
        h = min(h, h_max - y)
        frame = frame[y:y+h, x:x+w].copy()
        log(f"🟦 使用 OXY ROI 區域：{oxy_roi}")

    win_name = "OXY Preprocess + OTSU Debug"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1400, 700)

    # === 初始化 Trackbars ===
    cv2.createTrackbar("Brightness", win_name, oxy_brightness, 100, lambda x: None)
    cv2.createTrackbar("Contrast", win_name, oxy_contrast, 100, lambda x: None)
    cv2.createTrackbar("Gamma", win_name, oxy_gamma, 100, lambda x: None)
    cv2.createTrackbar("Saturation", win_name, oxy_saturation, 100, lambda x: None)
    cv2.createTrackbar("Threshold (0=OTSU auto)", win_name, oxy_otsu_threshold, 255, lambda x: None)

    log("📊 使用滑桿調整曝光/對比/閾值，按 S 儲存設定，ESC 離開")

    while True:
        # === 讀取滑桿 ===
        b = cv2.getTrackbarPos("Brightness", win_name)
        c = cv2.getTrackbarPos("Contrast", win_name)
        g = cv2.getTrackbarPos("Gamma", win_name)
        s = cv2.getTrackbarPos("Saturation", win_name)
        t = cv2.getTrackbarPos("Threshold (0=OTSU auto)", win_name)

        # === 前處理流程（與 oxy_preview_loop 相同）===
        img = frame.copy().astype(np.float32)
        brightness = (b - 50) * 2.5
        contrast = (c / 50.0)
        img = np.clip((img - 128) * contrast + 128 + brightness, 0, 255).astype(np.uint8)

        # 飽和度
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= (s / 50.0)
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # 灰階 + gamma
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gamma = g / 50.0
        gray = np.uint8(np.clip(np.power(gray / 255.0, 1.0 / gamma) * 255, 0, 255))

        # 閾值化
        if t == 0:
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, th = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)

        # === 顯示原圖 + Gray + Binary 合併視圖 ===
        merged = np.hstack([
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
        ])
        cv2.imshow(win_name, merged[:, :, ::-1])  # BGR → RGB 修正顏色

        # === 控制鍵 ===
        key = cv2.waitKey(50) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('s'):
            oxy_brightness = b
            oxy_contrast = c
            oxy_gamma = g
            oxy_saturation = s
            oxy_otsu_threshold = t
            save_roi_config()
            log(f"💾 已儲存設定 Bright={b}, Contrast={c}, Gamma={g}, Satur={s}, Th={t}")

    cv2.destroyWindow(win_name)
    log("🧪 OXY Debug 工具已關閉")

def manual_trigger_bad():
    global debug_cond_bad
    debug_cond_bad = True
    log("🧨 手動觸發條件：Predict BAD")
    handle_emergency("Manual Predict BAD")
def manual_trigger_high_oxy():
    global debug_cond_high_oxy
    debug_cond_high_oxy = True
    log("🧨 手動觸發條件：OXY 高於閾值")
    handle_emergency("Manual OXY HIGH")
def reset_safety_conditions():
    """使用者確認問題排除，清除安全旗標"""
    global debug_cond_bad, debug_cond_high_oxy
    debug_cond_bad = False
    debug_cond_high_oxy = False
    log("✅ 使用者確認問題已排除，安全狀態已恢復正常")
# ============== Main App ==============
def main():
    global root, left_preview, right_preview, powder_bed_roi_preview, trigger_roi_preview
    global oxy_value_label, roi_result_label, console, status_label

    root = tk.Tk()
    root.title("Smart ROI Monitor v13 (RTSP + OXY MJPEG)")
    root.geometry("1200x850")
    root.configure(bg="#202020")

    # === Menu ===
    menubar = tk.Menu(root)
    macro_menu = tk.Menu(menubar, tearoff=0)
    macro_menu.add_command(label="錄製巨集", command=record_main_macro)
    macro_menu.add_command(label="播放巨集", command=play_main_macro)
    macro_menu.add_command(label="停止巨集", command=stop_macro_play)
    macro_menu.add_separator()
    macro_menu.add_command(label="設定播放間隔", command=set_macro_delay)
    menubar.add_cascade(label="巨集", menu=macro_menu)

    roi_menu = tk.Menu(menubar, tearoff=0)
    roi_menu.add_command(label="選取粉床 ROI (Main)", command=lambda: select_roi("main"))
    roi_menu.add_command(label="選取螢幕監聽 ROI", command=select_screen_roi)
    roi_menu.add_command(label="選取 OXY ROI", command=select_oxy_roi)
    roi_menu.add_separator()
    roi_menu.add_command(label="重新載入 ROI 設定", command=load_roi_config)
    roi_menu.add_command(label="儲存 ROI 設定", command=save_roi_config)
    menubar.add_cascade(label="ROI 設定", menu=roi_menu)
    root.config(menu=menubar)

    # === Debug Menu ===
    debug_menu = tk.Menu(menubar, tearoff=0)
    debug_menu.add_command(label="手動推送一次 Predict", command=manual_predict_once)
    debug_menu.add_command(label="進階 OXY Preprocess + OTSU Debug", command=debug_oxy_preprocess_otsu)
    debug_menu.add_separator()
    debug_menu.add_command(label="🔴 手動觸發 Predict BAD", command=manual_trigger_bad)
    debug_menu.add_command(label="🟠 手動觸發 OXY 高於閾值", command=manual_trigger_high_oxy)
    menubar.add_cascade(label="Debug 工具", menu=debug_menu)

    # === Status bar ===
    status_frame = tk.Frame(root, bg="#202020")
    status_frame.grid(row=0, column=0, sticky="ew", pady=5)
    status_label = tk.Label(status_frame, text="🔴 Idle", fg="red", bg="#202020", font=("Arial", 14, "bold"))
    status_label.pack(side="left", padx=10)
    tk.Button(status_frame, text="▶ 開始執行", bg="#3cb371", command=start_all).pack(side="left", padx=5)
    tk.Button(status_frame, text="⏹ 結束執行", bg="#ff6347", command=stop_all).pack(side="left", padx=5)

    # === 安全狀態區 (Safety State) ===
    safety_frame = tk.Frame(status_frame, bg="#202020")
    safety_frame.pack(side="right", padx=10)

    safety_bad_label = tk.Label(safety_frame, text="⚠ BAD狀態: 🟢", fg="lime", bg="#202020", font=("Consolas", 10, "bold"))
    safety_bad_label.pack(side="left", padx=5)

    safety_oxy_label = tk.Label(safety_frame, text=f"🫁 OXY安全閾值({oxy_threshold}): 🟢", fg="lime", bg="#202020", font=("Consolas", 10, "bold"))
    safety_oxy_label.pack(side="left", padx=5)

    tk.Button(safety_frame, text="✅ 恢復運行", command=reset_safety_conditions, bg="#444", fg="white").pack(side="left", padx=8)


    # === Main layout ===
    main_frame = tk.Frame(root, bg="#202020")
    main_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
    root.rowconfigure(1, weight=1)
    root.columnconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)
    main_frame.columnconfigure(1, weight=1)
    main_frame.rowconfigure(0, weight=1)

    # ----- Left (ROI) -----
    left_frame = tk.Frame(main_frame, bg="#202020")
    left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
    left_frame.columnconfigure(0, weight=2)
    left_frame.rowconfigure(2, weight=1)

    roi_box = tk.LabelFrame(left_frame, text="粉床 Stream", fg="white", bg="#202020")
    roi_box.grid(row=0, column=0, sticky="nsew", pady=5)
    left_preview = tk.Label(roi_box, bg="black")
    left_preview.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

    roi_result_label = tk.Label(left_frame, text="辨識結果：—", fg="cyan", bg="#202020", font=("Consolas", 12))
    roi_result_label.grid(row=1, column=0, pady=5)

    roi_subframe = tk.Frame(left_frame, bg="#202020")
    roi_subframe.grid(row=2, column=0, sticky="nsew", pady=5)
    roi_subframe.columnconfigure(0, weight=1)
    roi_subframe.columnconfigure(1, weight=1)

    powder_bed_roi_box = tk.LabelFrame(roi_subframe, text="粉床 ROI Predict", fg="white", bg="#202020")
    powder_bed_roi_box.grid(row=0, column=0, sticky="nsew", padx=4)
    powder_bed_roi_preview = tk.Label(powder_bed_roi_box, bg="black")
    powder_bed_roi_preview.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

    trigger_roi_box = tk.LabelFrame(roi_subframe, text="螢幕監聽 ROI", fg="white", bg="#202020")
    trigger_roi_box.grid(row=0, column=1, sticky="nsew", padx=4)
    trigger_roi_preview = tk.Label(trigger_roi_box, bg="black")
    trigger_roi_preview.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

    # ----- Right (OXY + Console) -----
    right_frame = tk.Frame(main_frame, bg="#202020")
    right_frame.grid(row=0, column=1, sticky="nsew")
    right_frame.columnconfigure(0, weight=1)
    right_frame.rowconfigure(2, weight=1)

    oxy_wrapper = tk.Frame(right_frame, bg="#202020", width=400)
    oxy_wrapper.grid(row=0, column=0, sticky="n", pady=5)
    oxy_wrapper.grid_propagate(False)  # ✅ 固定寬度，不讓子元件自動撐開
    oxy_box = tk.LabelFrame(
        oxy_wrapper,
        text="氧氣 Stream",
        fg="white",
        bg="#202020"
    )
    oxy_box.pack(fill="both", expand=True)
    right_preview = tk.Label(oxy_box, bg="black")
    right_preview.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

    oxy_value_label = tk.Label(right_frame, text="OCR 結果：—", fg="lime", bg="#202020", font=("Consolas", 12))
    oxy_value_label.grid(row=1, column=0, pady=5, sticky="ew")

    console_box = tk.LabelFrame(right_frame, text="Console Log", fg="white", bg="#202020")
    # console_box.grid(row=2, column=0, sticky="nsew", pady=5)
    console_box.grid(row=2, column=0, sticky="n", pady=5)
    console_box.columnconfigure(0, weight=1)
    console_box.rowconfigure(0, weight=1)
    # console = tk.Text(console_box, font=("Consolas", 10), bg="#111", fg="white", wrap="word")
    console = tk.Text(console_box, font=("Consolas", 10), bg="#111", fg="white", wrap="word", height=20)
    console.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

    # === Init + Start Threads ===
    load_roi_config()
    init_gray_preview()

    try:
        pil = overlay_powbed(roi_frame_buffer, last_predict_text, last_predict_conf)
        tkimg = to_tk(pil)
        left_preview.configure(image=tkimg)
        left_preview.image = tkimg
        log("🟩 已載入並顯示 ROI 疊圖（初始化）")
    except Exception as e:
        log(f"⚠️ 初始化 ROI 疊圖失敗：{e}")

    threading.Thread(target=powbed_preview_loop, daemon=True).start()
    threading.Thread(target=oxy_preview_loop, daemon=True).start()
    threading.Thread(target=screen_roi_preview_loop, daemon=True).start()
    log("📡 粉床 & 氧氣 串流執行中、trigger螢幕監聽中")

    if not ROBOWFLOW_ENABLED:
        log("⚠ Roboflow 未啟用，將以 mock good 模式運行（不會觸發停止）")
    if not TESS_OK:
        log("⚠ 未安裝 pytesseract（OCR 無法運作）")

            
    def update_safety_status():
        """每 0.5 秒更新安全狀態指標"""
        bad_color = "lime" if not debug_cond_bad else "red"
        oxy_color = "lime" if not debug_cond_high_oxy else "red"

        safety_bad_label.config(text=f"⚠ BAD狀態: {'🟢' if not debug_cond_bad else '🔴'}", fg=bad_color)
        safety_oxy_label.config(text=f"🫁 OXY安全閾值({oxy_threshold}): {'🟢' if not debug_cond_high_oxy else '🔴'}", fg=oxy_color)

        root.after(500, update_safety_status)

    root.after(500, update_safety_status)


    root.mainloop()

if __name__ == "__main__":
    main()
