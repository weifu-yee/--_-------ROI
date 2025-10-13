import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading, time, cv2, numpy as np, pyautogui, os
from PIL import Image, ImageTk
from inference_sdk import InferenceHTTPClient

# === Roboflow API 設定 ===
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="RnGFo8AzPLZNtcop9YZ0"
)
WORKSPACE = "semantic-segmentation-9o2cy"
WORKFLOW_ID = "custom-workflow-2"

# === 全域變數 ===
roi_main = None
roi_trigger = None
monitoring = False
alert_window = None
alert_image_path = "alert.png"
interval = 2.0
trigger_update_interval = 0.2  # ROI2 擷取畫面頻率（秒）
trigger_delay_after_gray = 1.5 # 綠變灰後延遲多少秒再預測
pause_for_alert = False

# === GUI 主窗 ===
root = tk.Tk()
root.title("ROI Dual-Trigger (Green→Gray Detection)")
root.geometry("1100x780")
root.configure(bg="#202020")

trigger_mode = tk.StringVar(value="color")  # "interval" or "color"
latest_frame_main = None
latest_frame_trigger = None

# === Log ===
def log_message(msg):
    text_log.insert(tk.END, f"{time.strftime('%H:%M:%S')} | {msg}\n")
    text_log.see(tk.END)
    print(msg)

# === 錄製狀態 ===
status_label = tk.Label(root, text="🔴 Not Recording", fg="red", bg="#202020", font=("Arial", 14, "bold"))
status_label.pack(pady=5)

def update_status(is_recording):
    if is_recording:
        status_label.config(text="🟢 Recording...", fg="lime")
    else:
        status_label.config(text="🔴 Not Recording", fg="red")

# === ROI 選擇 ===
def select_roi_main():
    global roi_main
    img = pyautogui.screenshot()
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cv2.imshow("Select ROI (Main)", frame)
    roi_main = cv2.selectROI("Select ROI (Main)", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    if roi_main == (0, 0, 0, 0):
        roi_main = None
        messagebox.showwarning("ROI Selection", "未選取主 ROI")
    else:
        log_message(f"主 ROI 選取完成: {roi_main}")

def select_roi_trigger():
    global roi_trigger
    img = pyautogui.screenshot()
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cv2.imshow("Select ROI (Trigger)", frame)
    roi_trigger = cv2.selectROI("Select ROI (Trigger)", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    if roi_trigger == (0, 0, 0, 0):
        roi_trigger = None
        messagebox.showwarning("ROI Selection", "未選取觸發 ROI")
    else:
        log_message(f"觸發 ROI 選取完成: {roi_trigger}")

# === 顯示 ROI 畫面 ===
def show_frame_in_label(frame_bgr, target_label, size=(256,256)):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    frame_pil = frame_pil.resize(size)
    tk_img = ImageTk.PhotoImage(frame_pil)
    target_label.config(image=tk_img)
    target_label.image = tk_img

# === 綠→灰 偵測（使用你提供的版本） ===
def detect_green_to_gray(prev_img, curr_img, g_drop=5, gray_increase=10):
    """返回 (bool, ΔG, ΔGray)，當綠色顯著下降 + 灰度上升時觸發"""
    prev_mean = np.mean(prev_img, axis=(0,1))
    curr_mean = np.mean(curr_img, axis=(0,1))
    delta_g = curr_mean[1] - prev_mean[1]   # G下降

    prev_gray = np.mean(cv2.cvtColor(prev_img.astype(np.uint8), cv2.COLOR_BGR2GRAY))
    curr_gray = np.mean(cv2.cvtColor(curr_img.astype(np.uint8), cv2.COLOR_BGR2GRAY))
    delta_gray = curr_gray - prev_gray  # 灰度上升

    trigger = (delta_g > g_drop) and (delta_gray > gray_increase)
    return trigger, delta_g, delta_gray

# === API 推論 ===
def predict_once(frame_bgr=None):
    global roi_main, pause_for_alert
    if roi_main is None or pause_for_alert:
        return
    x, y, w, h = roi_main
    if frame_bgr is None:
        screenshot = np.array(pyautogui.screenshot(region=(x, y, w, h)))
        frame_bgr = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    show_frame_in_label(frame_bgr, frame_label_main)
    temp_path = "_temp_capture.jpg"
    cv2.imwrite(temp_path, frame_bgr)
    try:
        result = client.run_workflow(
            workspace_name=WORKSPACE,
            workflow_id=WORKFLOW_ID,
            images={"image": temp_path},
            use_cache=True
        )
        os.remove(temp_path)
        pred_class = result[0]["predictions"]["top"]
        conf = result[0]["predictions"]["confidence"]
        log_message(f"Prediction: {pred_class} ({conf:.3f})")
        if pred_class != "good":
            show_alert()
        else:
            close_alert()
    except Exception as e:
        log_message(f"❌ API Error: {e}")

# === 警示圖 ===
def show_alert():
    global alert_window, pause_for_alert
    if alert_window is not None:
        return
    pause_for_alert = True
    alert_window = tk.Toplevel()
    alert_window.attributes("-fullscreen", True)
    alert_window.configure(bg="black")
    if os.path.exists(alert_image_path):
        img = Image.open(alert_image_path)
        sw, sh = alert_window.winfo_screenwidth(), alert_window.winfo_screenheight()
        img = img.resize((sw, sh), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)
        label = tk.Label(alert_window, image=tk_img)
        label.image = tk_img
        label.pack(fill="both", expand=True)
    else:
        tk.Label(alert_window, text="⚠️ 鋪粉失敗", fg="red", bg="black", font=("Arial", 100, "bold")).pack(expand=True)
    alert_window.bind("<Button-1>", lambda e: close_alert())
    alert_window.bind("<Escape>", lambda e: close_alert())
    log_message("🚨 Alert shown! Monitoring paused.")

def close_alert():
    global alert_window, pause_for_alert
    if alert_window is not None:
        alert_window.destroy()
        alert_window = None
        pause_for_alert = False
        log_message("✅ Alert closed. Monitoring resumed.")

# === 主監測 ===
def monitor_loop():
    global monitoring, roi_trigger, trigger_mode, interval
    prev_trigger_img = None
    while monitoring:
        if pause_for_alert:
            time.sleep(0.2)
            continue
        if trigger_mode.get() == "interval":
            predict_once()
            time.sleep(interval)
            continue
        if roi_trigger is None:
            log_message("⚠️ 尚未設定觸發 ROI！")
            time.sleep(1)
            continue

        # ROI2 擷取頻率
        x, y, w, h = roi_trigger
        trigger_img = np.array(pyautogui.screenshot(region=(x, y, w, h)))
        trigger_img = cv2.cvtColor(trigger_img, cv2.COLOR_RGB2BGR)
        show_frame_in_label(trigger_img, frame_label_trigger)

        if prev_trigger_img is not None:
            triggered, dG, dGray = detect_green_to_gray(prev_trigger_img, trigger_img)
            color_diff_label.config(text=f"ΔG={dG:.1f}, ΔGray={dGray:.1f}")
            if triggered:
                log_message(f"綠→灰 偵測到變化 (ΔG={dG:.1f}, ΔGray={dGray:.1f}) → 延遲{trigger_delay_after_gray}秒預測")
                time.sleep(trigger_delay_after_gray)  # 延遲
                predict_once()
                time.sleep(1.5)
        prev_trigger_img = trigger_img
        time.sleep(trigger_update_interval)  # ROI2更新頻率

# === 控制 ===
def start_monitoring():
    global monitoring
    if roi_main is None:
        messagebox.showwarning("Warning", "請先設定主 ROI！")
        return
    if monitoring:
        return
    monitoring = True
    update_status(True)
    t = threading.Thread(target=monitor_loop, daemon=True)
    t.start()
    log_message("✅ 開始監測...")

def stop_monitoring():
    global monitoring
    monitoring = False
    update_status(False)
    close_alert()
    log_message("🟥 停止監測。")

# === GUI 布局 ===
tk.Label(root, text="🔍 Roboflow ROI Classification System", fg="white", bg="#202020", font=("Arial", 14, "bold")).pack(pady=10)
frame_btn = tk.Frame(root, bg="#202020")
frame_btn.pack(pady=5)
tk.Button(frame_btn, text="設定主 ROI", command=select_roi_main, width=18).grid(row=0, column=0, padx=5)
tk.Button(frame_btn, text="設定觸發 ROI", command=select_roi_trigger, width=18).grid(row=0, column=1, padx=5)

frame_mode = tk.LabelFrame(root, text="觸發模式", fg="white", bg="#202020", font=("Arial", 12, "bold"))
frame_mode.pack(pady=10)
ttk.Radiobutton(frame_mode, text="顏色變化觸發", value="color", variable=trigger_mode).pack(anchor="w", padx=10)
ttk.Radiobutton(frame_mode, text="定時模式", value="interval", variable=trigger_mode).pack(anchor="w", padx=10)
frame_interval = tk.Frame(frame_mode, bg="#202020")
frame_interval.pack(pady=5)
tk.Label(frame_interval, text="間隔秒數：", fg="white", bg="#202020").grid(row=0, column=0)
entry_interval = tk.Entry(frame_interval, width=6)
entry_interval.insert(0, "2.0")
entry_interval.grid(row=0, column=1)
def update_interval():
    global interval
    try:
        interval = float(entry_interval.get())
        log_message(f"⏱️ 已更新間隔秒數: {interval}s")
    except ValueError:
        messagebox.showerror("Error", "請輸入有效數字")
tk.Button(frame_interval, text="更新", command=update_interval, width=6).grid(row=0, column=2, padx=5)

frame_ctrl = tk.Frame(root, bg="#202020")
frame_ctrl.pack(pady=10)
tk.Button(frame_ctrl, text="開始監測", command=start_monitoring, bg="#3cb371", width=15).grid(row=0, column=0, padx=5)
tk.Button(frame_ctrl, text="停止監測", command=stop_monitoring, bg="#ff6347", width=15).grid(row=0, column=1, padx=5)
tk.Button(frame_ctrl, text="離開", command=lambda: root.destroy(), width=15).grid(row=0, column=2, padx=5)

frame_preview = tk.Frame(root, bg="#202020")
frame_preview.pack(pady=5)
tk.Label(frame_preview, text="主 ROI (分析畫面)", fg="white", bg="#202020", font=("Arial", 11, "bold")).grid(row=0, column=0)
tk.Label(frame_preview, text="觸發 ROI (綠→灰監測)", fg="white", bg="#202020", font=("Arial", 11, "bold")).grid(row=0, column=1)
frame_label_main = tk.Label(frame_preview, bg="black")
frame_label_main.grid(row=1, column=0, padx=10, pady=5)
frame_label_trigger = tk.Label(frame_preview, bg="black")
frame_label_trigger.grid(row=1, column=1, padx=10, pady=5)
color_diff_label = tk.Label(frame_preview, text="ΔG=0, ΔGray=0", fg="lightgray", bg="#202020", font=("Consolas", 11))
color_diff_label.grid(row=2, column=1, pady=5)

tk.Label(root, text="🧾 狀態日誌：", fg="white", bg="#202020", font=("Arial", 12, "bold")).pack(anchor="w", padx=10)
text_log = scrolledtext.ScrolledText(root, width=120, height=15, font=("Consolas", 10))
text_log.pack(padx=10, pady=5)





from pynput import mouse, keyboard
import json, datetime

macro_recording = False
macro_events = []
macro_thread = None
macro_stop_flag = False
main_macro_path = os.path.join("macros", "main_macro.json")

# === 確保有 macros 資料夾 ===
if not os.path.exists("macros"):
    os.makedirs("macros")

# === 錄製主巨集 ===
def record_main_macro():
    global macro_recording, macro_events
    macro_events = []
    macro_recording = True
    update_status(True)
    log_message("🎬 開始錄製【主巨集】 (按 ESC 結束)")

    start_time = time.time()

    def on_click(x, y, button, pressed):
        if not macro_recording:
            return False
        macro_events.append({
            "time": time.time() - start_time,
            "type": "click",
            "x": x, "y": y,
            "button": str(button),
            "pressed": pressed
        })

    def on_scroll(x, y, dx, dy):
        if not macro_recording:
            return False
        macro_events.append({
            "time": time.time() - start_time,
            "type": "scroll",
            "x": x, "y": y,
            "dx": dx, "dy": dy
        })

    def on_key(key):
        global macro_recording
        if key == keyboard.Key.esc:
            macro_recording = False
            return False
        if macro_recording:
            macro_events.append({
                "time": time.time() - start_time,
                "type": "key",
                "key": str(key)
            })

    # 🔸 這裡在主執行緒中執行，不再開子執行緒
    ml = mouse.Listener(on_click=on_click, on_scroll=on_scroll)
    kl = keyboard.Listener(on_press=on_key)
    ml.start()
    kl.start()
    kl.join()
    ml.stop()

    json.dump(macro_events, open(main_macro_path, "w"), indent=2)
    log_message(f"✅ 主巨集錄製完成，共 {len(macro_events)} 筆事件，已儲存為 {main_macro_path}")
    update_status(False)

# === 播放主巨集 ===
def play_main_macro():
    global macro_thread, macro_stop_flag
    if not os.path.exists(main_macro_path):
        messagebox.showerror("錯誤", "尚未錄製主巨集。請先錄製。")
        return

    def run_macro():
        global macro_stop_flag
        macro_stop_flag = False
        update_status(True)
        log_message(f"▶️ 開始播放主巨集")
        events = json.load(open(main_macro_path))
        t0 = time.time()

        for e in events:
            if pause_for_alert or macro_stop_flag:
                log_message("⏸️ 巨集暫停：偵測到警示或手動停止")
                break

            delay = e["time"] - (time.time() - t0)
            if delay > 0:
                time.sleep(delay)

            if e["type"] == "click" and e["pressed"]:
                pyautogui.click(e["x"], e["y"])
            elif e["type"] == "scroll":
                pyautogui.scroll(e["dy"])  # 滾輪向上為正值，向下為負
            elif e["type"] == "key":
                key = e["key"].replace("'", "")
                if len(key) == 1:
                    pyautogui.typewrite(key)

        log_message("✅ 主巨集播放結束")
        update_status(False)

    macro_thread = threading.Thread(target=run_macro, daemon=True)
    macro_thread.start()

# === 停止播放 ===
def stop_macro():
    global macro_stop_flag
    macro_stop_flag = True
    update_status(False)
    log_message("🟥 手動停止巨集播放")

# === GUI 區塊 ===
frame_macro = tk.LabelFrame(root, text="🖱️ 主巨集控制 (支援滾輪)", fg="white", bg="#202020", font=("Arial", 12, "bold"))
frame_macro.pack(pady=10)

tk.Button(frame_macro, text="錄製主巨集", command=record_main_macro,
          bg="#4682B4", fg="white", width=15).grid(row=0, column=0, padx=5)
tk.Button(frame_macro, text="播放主巨集", command=play_main_macro,
          bg="#32CD32", fg="white", width=15).grid(row=0, column=1, padx=5)
tk.Button(frame_macro, text="停止播放", command=stop_macro,
          bg="#FF6347", fg="white", width=15).grid(row=0, column=2, padx=5)




root.mainloop()
