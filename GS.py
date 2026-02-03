import sys
import socket
import numpy as np
import cv2
import keyboard
import time
import struct
import threading
from ultralytics import YOLO

LOCAL_UDP_IP = '192.168.1.2'
SHARED_UDP_PORT = 50000
ESP32_UDP_IP = "192.168.1.1"
ESP32_UDP_PORT = 55555
reverse_flag = True

running = True
image = np.zeros((240, 240, 3), dtype=np.uint8)
image_lock = threading.Lock()


model = YOLO("stool_best.pt")


# ソケット設定
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((LOCAL_UDP_IP, SHARED_UDP_PORT))
sock.settimeout(0.5)

# -------------------------------
# ESP32 通信系
# -------------------------------
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

def image_decode(packet_number, data, img):
    index = (packet_number - 1) // 3 * 12
    rgb = 2 - (packet_number - 1) % 3
    img[index:index+12, :, rgb] = (
        np.reshape(
            np.ravel(np.array([data // 16, data % 16]).T),
            (12, 240)
        ) * 16
    )


def plot_box(image_rgb):
    results = model(image_rgb)
    return results[0].plot()

 

# -------------------------------
# 受信スレッド
# -------------------------------
def receiver_thread():
    global running
    while running:
        packet_number, data, _ = receive_telemetry()
        if packet_number is None:
            continue

        if packet_number == 0x5C:
            telemetry_reader(data)

        elif packet_number == 0xFF:
            pass  # フレーム終端マーカー

        else:
            with image_lock:
                image_decode(packet_number, data, image)

# -------------------------------
# 操作スレッド
# -------------------------------
def control_thread():
    global running
    last_send = 0
    send_interval = 0.1  # 10Hz

    while running:
        now = time.time()

        def send(cmd):
            nonlocal last_send
            if now - last_send > send_interval:
                send_command(cmd)
                last_send = now

        if keyboard.is_pressed('w'):
            send("W")
        elif keyboard.is_pressed('s'):
            send("S")
        elif keyboard.is_pressed('a'):
            send("A")
        elif keyboard.is_pressed('d'):
            send("D")
        elif keyboard.is_pressed('r'):
            send("R")
        elif keyboard.is_pressed('g'):
            send("G")
        elif keyboard.is_pressed('t'):
            send("T")
        elif keyboard.is_pressed('b'):
            send("B")
        elif keyboard.is_pressed('q'):
            running = False

        time.sleep(0.01)  # CPU使用率対策

# -------------------------------
# メインループ（表示）
# -------------------------------
def main():
    global running

    print("Control start (Q to quit)")

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
