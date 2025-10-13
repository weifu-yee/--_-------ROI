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

TESS_OK = True
try:
    import pytesseract
except Exception:
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
OXY_STREAM_URL = "http://192.168.0.102:8080/video"

# ============== Globals ==============
monitoring = False
pause_for_alert = False
macro_play_stop = False
recording = False

roi_main = None
roi_trigger = None

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
    data = {"roi_main": list(roi_main) if roi_main else None,
            "roi_trigger": list(roi_trigger) if roi_trigger else None,
            "roi_stream_url": ROI_STREAM_URL,
            "oxy_stream_url": OXY_STREAM_URL}
    json.dump(data, open(ROI_FILE, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    log("✅ ROI/Stream 設定已儲存")

def load_roi_config():
    global roi_main, roi_trigger, ROI_STREAM_URL, OXY_STREAM_URL, ROI_USERNAME, ROI_PASSWORD
    if os.path.exists(ROI_FILE):
        d = json.load(open(ROI_FILE, "r", encoding="utf-8"))
        roi_main    = tuple(d.get("roi_main")) if d.get("roi_main") else None
        roi_trigger = tuple(d.get("roi_trigger")) if d.get("roi_trigger") else None
        ROI_STREAM_URL = d.get("roi_stream_url", ROI_STREAM_URL)
        OXY_STREAM_URL = d.get("oxy_stream_url", OXY_STREAM_URL)
        ROI_USERNAME = d.get("roi_username", "")
        ROI_PASSWORD = d.get("roi_password", "")
        log(f"載入設定 | ROI URL={ROI_STREAM_URL}, 帳號={ROI_USERNAME}")
    else:
        ROI_USERNAME = ROI_PASSWORD = ""
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
        return read_stream(OXY_STREAM_URL)
    except:
        return gray_frame()

# ============== Visual overlays ==============
def overlay_roi_and_badge(frame_bgr, pred_text=None, pred_conf=None):
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    try: font = ImageFont.truetype("arial.ttf", 18)
    except: font = ImageFont.load_default()

    if roi_main:
        x,y,w,h = roi_main
        draw.rectangle([x, y, x+w, y+h], outline=(0,255,0), width=3)
        draw.text((x, max(0,y-20)), "ROI1(Main)", fill=(0,255,0), font=font)
    if roi_trigger:
        x,y,w,h = roi_trigger
        draw.rectangle([x, y, x+w, y+h], outline=(255,215,0), width=3)
        draw.text((x, max(0,y-20)), "ROI2(Trigger)", fill=(255,215,0), font=font)
    return pil


def overlay_oxy(frame_bgr, oxy_text:str, ok:bool, show_warn_if_empty=True):
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    try: font = ImageFont.truetype("arial.ttf", 20)
    except: font = ImageFont.load_default()
    text = f"O2: {oxy_text}" if oxy_text else ("⚠ No reading" if show_warn_if_empty else "")
    if text:
        color = (50,205,50) if ok else (255,215,0)
        draw.rectangle([10,10,10+240,42], fill=(0,0,0,128), outline=color, width=2)
        draw.text((20,16), text, fill=color, font=font)
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
    tmp = os.path.join(ASSETS_DIR, "_tmp_roi.jpg")

    try:
        cv2.imwrite(tmp, crop)
        if ROBOWFLOW_ENABLED:
            try:
                result = client.run_workflow(
                    workspace_name=WORKSPACE,
                    workflow_id=WORKFLOW_ID,
                    images={"image": tmp},
                    use_cache=True
                )
                pred_class = result[0]["predictions"]["top"]
                conf = float(result[0]["predictions"]["confidence"])
                last_predict_text, last_predict_conf = pred_class, conf
                log(f"[ROI] Prediction: {pred_class} ({conf:.3f})")

                # ✅ 更新 ROI 結果顯示在 App 介面
                try:
                    color = "lime" if pred_class.lower() == "good" else "orange"
                    roi_result_label.config(
                        text=f"辨識結果：{pred_class} ({conf:.2f})",
                        fg=color
                    )
                except Exception:
                    pass

                # 若非 good 觸發異常處理
                if pred_class.lower() != "good":
                    handle_emergency("ROI_BAD")

                return pred_class, conf

            except Exception as e:
                log(f"❌ Inference error: {e}")

        else:
            # Mock 模式
            last_predict_text, last_predict_conf = "mock_good", 0.99
            roi_result_label.config(text="辨識結果：mock_good (0.99)", fg="lime")
            return "good", 0.99

    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

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

# ============== Monitor threads ==============
def roi_monitor_loop():
    """持續更新左側 ROI 畫面"""
    prev_trig = None
    while monitoring:
        frame = get_roi_frame()
        global roi_frame_buffer
        roi_frame_buffer = frame.copy()  # ✅ 更新 ROI buffer 內容

        # ROI 觸發監測（若有設定 Trigger ROI）
        if roi_trigger is not None:
            x, y, w, h = roi_trigger
            trig = frame[y:y+h, x:x+w].copy()
            if prev_trig is not None:
                triggered, dG, dGray = detect_green_to_gray(
                    prev_trig, trig, g_drop_threshold, gray_increase_threshold
                )
                if triggered:
                    log(f"[ROI] 綠→灰觸發 ΔG={dG:.1f}, ΔGray={dGray:.1f}")
                    time.sleep(trigger_delay_after_gray)
                    do_inference_on_roi_frame(frame)
            prev_trig = trig

        # ✅ 顯示 ROI 畫面於左側
        pil = overlay_roi_and_badge(frame, last_predict_text, last_predict_conf)
        tkimg = to_tk(pil, size=(480, 270))
        left_preview.configure(image=tkimg)
        left_preview.image = tkimg

        time.sleep(trigger_update_interval)

def oxy_monitor_loop():
    while monitoring:
        frame = get_oxy_frame()
        text = ""
        if TESS_OK:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(th, config="--psm 7 -c tessedit_char_whitelist=0123456789.%").strip()
        pil = overlay_oxy(frame, text, True)
        tkimg = to_tk(pil, size=(480,270))
        right_preview.configure(image=tkimg); right_preview.image = tkimg
        time.sleep(0.1)

# ============== Background ROI Preview (always running) ==============
def roi_preview_loop():
    """穩定的 ROI 預覽更新迴圈（可顯示 RTSP 畫面）"""
    global roi_frame_buffer
    cap = None
    url = ROI_STREAM_URL

    # 自動將帳號中的 @ 轉成 %40
    if "@" in url.split("://", 1)[-1].split("@")[0]:
        parts = url.split("://", 1)
        prefix = parts[0] + "://"
        body = parts[1]
        user_part, rest = body.split("@", 1)
        if "@" in user_part:
            user_part = user_part.replace("@", "%40")
        url = prefix + user_part + "@" + rest
        log(f"🔧 已自動修正 RTSP URL: {url}")

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        log(f"❌ 無法開啟 RTSP 串流：{url}")
        return

    log("📡 ROI 串流已開啟，開始更新畫面")
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            log("⚠️ ROI 串流中斷，重新嘗試連線中...")
            time.sleep(1)
            cap.release()
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            continue

        roi_frame_buffer = frame.copy()

        # 轉換成 Tkinter 可顯示格式
        try:
            pil = overlay_roi_and_badge(frame, last_predict_text, last_predict_conf)
            tkimg = to_tk(pil, size=(480, 270))
            # 更新到 GUI （在主執行緒排程執行以防 tkinter 崩潰）
            root.after(0, lambda img=tkimg: left_preview.configure(image=img))
            root.after(0, lambda img=tkimg: setattr(left_preview, "image", img))
        except Exception as e:
            log(f"⚠️ ROI 畫面更新失敗：{e}")

        time.sleep(0.05)  # 每秒約 20fps 更新


# ============== Macro (Enhanced Loop + Scroll Support) ==============

import ctypes

macro_events = []
macro_play_stop = False
macro_loop_delay = 3.0  # 🕒 每輪播放間隔秒數（可由使用者設定）

def get_scroll_lines():
    """取得 Windows 系統目前設定的滾動行數（預設 3）"""
    SPI_GETWHEELSCROLLLINES = 0x0068
    lines = ctypes.c_int()
    ctypes.windll.user32.SystemParametersInfoW(SPI_GETWHEELSCROLLLINES, 0, ctypes.byref(lines), 0)
    return lines.value if lines.value > 0 else 3

def record_main_macro():
    if not PYNPUT_OK:
        messagebox.showwarning("提示", "未安裝 pynput。")
        return

    def _record_thread():
        global macro_events
        macro_events = []
        set_status(True)
        log("🎬 開始錄製（按 ESC 結束）")

        start = time.time()
        scroll_lines = get_scroll_lines()

        def on_click(x, y, btn, pressed):
            macro_events.append({
                "t": time.time()-start,
                "type": "click",
                "x": x, "y": y,
                "btn": str(btn),
                "pressed": pressed
            })

        def on_scroll(x, y, dx, dy):
            macro_events.append({
                "t": time.time()-start,
                "type": "scroll",
                "x": x, "y": y,
                "dx": dx, "dy": dy * scroll_lines
            })

        def on_key(k):
            if k == keyboard.Key.esc:
                return False
            macro_events.append({
                "t": time.time()-start,
                "type": "key",
                "key": str(k)
            })

        ml = mouse.Listener(on_click=on_click, on_scroll=on_scroll)
        kl = keyboard.Listener(on_press=on_key)
        ml.start()
        kl.start()
        kl.join()  # 停止鍵盤監聽
        ml.stop()

        json.dump(macro_events, open(MACRO_FILE, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        set_status(False)
        log("✅ 錄製完成")

    threading.Thread(target=_record_thread, daemon=True).start()


def play_main_macro():
    if not os.path.exists(MACRO_FILE):
        messagebox.showwarning("提示", "沒有可播放的巨集。")
        return

    def _run():
        global macro_play_stop
        macro_play_stop = False
        scroll_lines = get_scroll_lines()
        log(f"▶ 無限播放巨集，間隔 {macro_loop_delay:.1f} 秒（每次滾動 {scroll_lines} 行）")

        import pyautogui
        while not macro_play_stop:
            ev = json.load(open(MACRO_FILE, "r", encoding="utf-8"))
            t0 = time.time()
            for e in ev:
                if macro_play_stop or not monitoring:
                    break
                delay = e["t"] - (time.time() - t0)
                if delay > 0:
                    time.sleep(delay)

                if e["type"] == "click" and e.get("pressed", True):
                    pyautogui.click(e["x"], e["y"])
                elif e["type"] == "key":
                    key = e["key"].replace("'", "")
                    if len(key) == 1:
                        pyautogui.typewrite(key)
                elif e["type"] == "scroll":
                    pyautogui.scroll(int(e["dy"]))  # 方向與 Windows 設定一致

            if macro_play_stop:
                break

            log(f"⏸ 等待 {macro_loop_delay:.1f}s 後重播")
            time.sleep(macro_loop_delay)

        log("🟥 巨集播放結束")

    threading.Thread(target=_run, daemon=True).start()


def stop_macro_play():
    global macro_play_stop
    macro_play_stop = True
    log("🟥 停止巨集")


def set_macro_delay():
    """彈出對話框讓使用者設定播放間隔秒數"""
    global macro_loop_delay
    val = tk.simpledialog.askfloat("設定播放間隔", "請輸入每次巨集播放間隔（秒）", minvalue=0.5, initialvalue=macro_loop_delay)
    if val is not None:
        macro_loop_delay = val
        log(f"⚙️ 已設定播放間隔：{macro_loop_delay:.1f} 秒")

# ============== Start/Stop ==============
def start_all():
    global monitoring
    if monitoring: return
    monitoring = True; set_status(True)
    threading.Thread(target=roi_monitor_loop, daemon=True).start()
    threading.Thread(target=oxy_monitor_loop, daemon=True).start()
    play_main_macro()
    log("✅ 開始執行")

def stop_all_monitoring(silent=False):
    global monitoring
    monitoring = False; set_status(False)
    if not silent: log("🟥 停止監測")

def stop_all():
    stop_macro_play(); stop_all_monitoring(); close_alert()

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
        pil = overlay_roi_and_badge(frame)
        tkimg = to_tk(pil, size=(480, 270))
        left_preview.configure(image=tkimg)
        left_preview.image = tkimg
        log("🟩 ROI 已顯示於預覽畫面")
    except Exception as e:
        log(f"⚠️ 無法更新預覽畫面：{e}")

# ============== GUI ==============
root = tk.Tk()
root.title("ROI Monitor v5 (灰底容錯版)")
root.geometry("1200x800")
root.configure(bg="#202020")


menubar = tk.Menu(root)
# === 巨集選單 ===
macro_menu = tk.Menu(menubar, tearoff=0)
macro_menu.add_command(label="錄製巨集", command=record_main_macro)
macro_menu.add_command(label="播放巨集", command=play_main_macro)
macro_menu.add_command(label="停止巨集", command=stop_macro_play)
macro_menu.add_separator()
macro_menu.add_command(label="設定播放間隔", command=set_macro_delay)
menubar.add_cascade(label="巨集", menu=macro_menu)
# === ROI 設定選單 ===
roi_menu = tk.Menu(menubar, tearoff=0)
roi_menu.add_command(label="選取 ROI Main", command=lambda: select_roi("main"))
roi_menu.add_command(label="選取 ROI Trigger", command=lambda: select_roi("trigger"))
roi_menu.add_separator()
roi_menu.add_command(label="重新載入 ROI 設定", command=load_roi_config)
roi_menu.add_command(label="儲存 ROI 設定", command=save_roi_config)
menubar.add_cascade(label="ROI 設定", menu=roi_menu)
root.config(menu=menubar)

status_label = tk.Label(root, text="🔴 Idle", fg="red", bg="#202020", font=("Arial", 14, "bold"))
status_label.pack(pady=6)

btn_frame = tk.Frame(root, bg="#202020"); btn_frame.pack(pady=4)
tk.Button(btn_frame, text="▶ 開始執行", width=14, bg="#3cb371", command=start_all).grid(row=0, column=0, padx=5)
tk.Button(btn_frame, text="⏹ 結束執行", width=14, bg="#ff6347", command=stop_all).grid(row=0, column=1, padx=5)

previews = tk.Frame(root, bg="#202020"); previews.pack(pady=6)
left_box  = tk.LabelFrame(previews, text="ROI Stream", fg="white", bg="#202020")
right_box = tk.LabelFrame(previews, text="Oxygen Stream (OCR)", fg="white", bg="#202020")
left_box.pack(side="left", padx=10); right_box.pack(side="left", padx=10)
left_preview = tk.Label(left_box, bg="black", width=480, height=270)
right_preview = tk.Label(right_box, bg="black", width=480, height=270)
left_preview.pack(padx=10, pady=10)
right_preview.pack(padx=10, pady=10)

# === 顯示 ROI 辨識結果與氧氣讀值 ===
roi_result_label = tk.Label(left_box, text="辨識結果：—", fg="cyan", bg="#202020", font=("Consolas", 12))
roi_result_label.pack(pady=(0,10))

oxy_value_label = tk.Label(right_box, text="氧氣值：—", fg="lime", bg="#202020", font=("Consolas", 12))
oxy_value_label.pack(pady=(0,10))

# Console log output
console = tk.Text(root, width=140, height=10, font=("Consolas", 10), bg="#111", fg="white")
console.pack(padx=10, pady=8)

# 初始化灰底畫面（避免無串流時為空白）
def init_gray_preview():
    g1 = to_tk(Image.fromarray(gray_frame(480,270)))
    g2 = to_tk(Image.fromarray(gray_frame(480,270)))
    left_preview.configure(image=g1); left_preview.image = g1
    right_preview.configure(image=g2); right_preview.image = g2
    log("🩶 初始灰底畫面已載入（等待串流）")

# ===== 啟動區 =====
load_roi_config()
init_gray_preview()

# 顯示 ROI 疊圖一次（若設定檔已有 ROI）
try:
    pil = overlay_roi_and_badge(roi_frame_buffer, last_predict_text, last_predict_conf)
    tkimg = to_tk(pil, size=(480, 270))
    left_preview.configure(image=tkimg)
    left_preview.image = tkimg
    log("🟩 已載入並顯示 ROI 疊圖（初始化）")
except Exception as e:
    log(f"⚠️ 初始化 ROI 疊圖失敗：{e}")

# 啟動 ROI 預覽背景更新
threading.Thread(target=roi_preview_loop, daemon=True).start()
log("📡 已啟動 ROI 即時預覽串流")

if not ROBOWFLOW_ENABLED:
    log("⚠ Roboflow 未啟用，將以 mock good 模式運行（不會觸發停止）")
if not TESS_OK:
    log("⚠ 未安裝 pytesseract（OCR 無法運作）")


root.mainloop()