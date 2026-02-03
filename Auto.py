import sys
import socket
import numpy as np
import cv2
import keyboard
import time
import struct
import threading
from ultralytics import YOLO

# ===============================
# 通信設定
# ===============================
LOCAL_UDP_IP = '192.168.1.2'
SHARED_UDP_PORT = 50000
ESP32_UDP_IP = "192.168.1.1"
ESP32_UDP_PORT = 55555

# ===============================
# 制御パラメータ
# ===============================
TARGET_CLASS_ID = 0       # 検出したい物体の class id
CENTER_THRESHOLD = 50     # px
SEND_INTERVAL = 0.1       # 秒（10Hz）

# ===============================
# 状態変数
# ===============================
running = True
reverse_flag = True

image = np.zeros((240, 240, 3), dtype=np.uint8)
image_lock = threading.Lock()

target_detected = False
target_center_x = None

# ===============================
# YOLOモデル
# ===============================
model = YOLO("stool_best.pt")

# ===============================
# UDPソケット
# ===============================
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((LOCAL_UDP_IP, SHARED_UDP_PORT))
sock.settimeout(0.5)

# ===============================
# 通信関数
# ===============================
def send_command(command):
    sock.sendto(command.encode(), (ESP32_UDP_IP, ESP32_UDP_PORT))
    print(f"Sent: {command}")

def receive_telemetry():
    try:
        data, _ = sock.recvfrom(2048)
    except socket.timeout:
        return None, None, None

    packet = np.frombuffer(data, dtype=np.uint8)
    return packet[0], packet[1:-1], packet[-1]

def telemetry_reader(data):
    if len(data) != 13:
        return
    pressure, temperature, humidity = struct.unpack('<fff', data[1:13])
    print(f"P:{pressure:.2f} T:{temperature:.2f} H:{humidity:.2f}")

# ===============================
# 画像デコード
# ===============================
def image_decode(packet_number, data, img):
    index = (packet_number - 1) // 3 * 12
    rgb = 2 - (packet_number - 1) % 3
    img[index:index+12, :, rgb] = (
        np.reshape(
            np.ravel(np.array([data // 16, data % 16]).T),
            (12, 240)
        ) * 16
    )

# ===============================
# YOLO検出 + 中心取得
# ===============================
def plot_box(image_rgb):
    global target_detected, target_center_x

    results = model(image_rgb)
    boxes = results[0].boxes

    target_detected = False
    target_center_x = None

    if boxes is not None:
        for box in boxes:
            cls = int(box.cls[0])
            if cls == TARGET_CLASS_ID:
                x1, y1, x2, y2 = box.xyxy[0]
                target_center_x = int((x1 + x2) / 2)
                target_detected = True
                break

    return results[0].plot()

# ===============================
# 受信スレッド
# ===============================
def receiver_thread():
    global running
    while running:
        packet_number, data, _ = receive_telemetry()
        if packet_number is None:
            continue

        if packet_number == 0x5C:
            telemetry_reader(data)

        elif packet_number == 0xFF:
            pass  # フレーム終端

        else:
            with image_lock:
                image_decode(packet_number, data, image)

# ===============================
# 制御スレッド（自動）
# ===============================
def control_thread():
    global running
    last_send = 0
    FRAME_CENTER_X = 240 // 2

    while running:
        now = time.time()

        def send(cmd):
            nonlocal last_send
            if now - last_send > SEND_INTERVAL:
                send_command(cmd)
                last_send = now

        # -------- 自動制御ロジック --------
        if target_detected and target_center_x is not None:
            diff = abs(target_center_x - FRAME_CENTER_X)
            if diff <= CENTER_THRESHOLD:
                send("W")   # 中心が合った → 前進
            else:
                send("M")   # ズレあり → 微小回転
        else:
            send("M")       # 未検出 → 探索回転

        # -------- 終了 --------
        if keyboard.is_pressed('q'):
            running = False

        time.sleep(0.01)

# ===============================
# メインループ（表示）
# ===============================
def main():
    global running

    print("Auto control start (Q to quit)")

    threading.Thread(target=receiver_thread, daemon=True).start()
    threading.Thread(target=control_thread, daemon=True).start()

    while running:
        with image_lock:
            frame = image.copy()

        if reverse_flag:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        frame = plot_box(frame)
        cv2.imshow("ESP32S3_Sense", cv2.resize(frame, (500, 500)))

        if cv2.waitKey(1) & 0xFF == 27:
            running = False

    cv2.destroyAllWindows()
    print("Exit")

if __name__ == "__main__":
    main()
