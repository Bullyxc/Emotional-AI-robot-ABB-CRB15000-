import google.generativeai as genai
import speech_recognition as sr
from gtts import gTTS
import os
import pygame
import time
import sys
import threading
import cv2
import math
import random
import re
import socket
from PIL import Image
import depthai as dai  # Import depthai for Oak-D Lite
import numpy as np
import ctypes
import ctypes.wintypes
import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import ImageTk

# --- การตั้งค่า (Configuration) ---
API_KEY = "Your AI API Key" # <--- Key ของคุณ
genai.configure(api_key=API_KEY)

# --- Global Variables ---
latest_frame = None
latest_depth_frame = None     # เก็บ depth frame ล่าสุด
center_distance_mm = 0        # ระยะทางจุดกลาง (mm)
is_ai_speaking = False        
stop_audio_event = threading.Event() 
program_running = True

# --- Chat Messages (สำหรับ Dashboard) ---
chat_messages = []  # list ของ {'role': 'user'/'ai'/'system', 'text': '...', 'time': '...'}
chat_messages_lock = threading.Lock()
new_chat_flag = False  # flag บอกว่ามีข้อความใหม่

# --- Proximity Warning ---
PROXIMITY_THRESHOLD_MM = 500  # 50 cm = 500 mm
PROXIMITY_COOLDOWN_SEC = 5    # พูดเตือนทุก 5 วินาที (ไม่ซ้ำถี่เกินไป)
last_proximity_warning_time = 0

# --- Emotion to Number Mapping ---
EMOTION_MAP = {
    "NEUTRAL":   0,
    "HAPPY":     1,
    "SAD":       2,
    "ANGRY":     3,
    "SURPRISED": 4,
    "SHY":       5,
    "POUT":      6,
    "ROLL":      7,
    "SLEEP":     8,
    "SUS":       9,
    "BLINK":    10,
    "PROXIMITY":99,   # เตือนระยะใกล้
}

# --- TCP/IP Server ---
TCP_HOST = '0.0.0.0'   # รับ connection จากทุก IP
TCP_PORT = 9999
robot_clients = []      # เก็บ list ของ client connections
robot_clients_lock = threading.Lock()
current_emotion_code = 0  # ค่าอารมณ์ปัจจุบัน


def send_emotion_to_robot(emotion_tag):
    """ส่งตัวเลขอารมณ์ไปยัง robot client ทุกตัวที่เชื่อมต่ออยู่"""
    global current_emotion_code
    emotion_code = EMOTION_MAP.get(emotion_tag, 0)
    current_emotion_code = emotion_code
    
    message = f"{emotion_code}\n"  # ส่งเป็นตัวเลข + newline
    
    with robot_clients_lock:
        disconnected = []
        for client_sock in robot_clients:
            try:
                client_sock.sendall(message.encode('utf-8'))
                print(f"📡 Sent to robot: {emotion_tag} -> {emotion_code}")
            except Exception as e:
                print(f"⚠️ Client disconnected: {e}")
                disconnected.append(client_sock)
        
        # ลบ client ที่ disconnect ออก
        for sock in disconnected:
            robot_clients.remove(sock)
            try: sock.close()
            except: pass


def tcp_server_thread():
    """TCP Server thread - รอ robot client เชื่อมต่อ"""
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.settimeout(1.0)  # timeout เพื่อให้ตรวจ program_running ได้
    
    try:
        server_sock.bind((TCP_HOST, TCP_PORT))
        server_sock.listen(5)
        print(f"🌐 TCP Server started on port {TCP_PORT}")
        print(f"   Robot client สามารถเชื่อมต่อมาที่ IP:PORT ของเครื่องนี้")
        print(robot_clients)
        
        while program_running:
            try:
                client_sock, addr = server_sock.accept()
                print(f"🤖 Robot client connected from {addr}")
                with robot_clients_lock:
                    robot_clients.append(client_sock)
                # ส่งอารมณ์ปัจจุบันให้ client ใหม่ทันที
                try:
                    client_sock.sendall(f"{current_emotion_code}\n".encode('utf-8'))
                except: pass
            except socket.timeout:
                continue
            except Exception as e:
                if program_running:
                    print(f"⚠️ TCP Server error: {e}")
    except Exception as e:
        print(f"❌ TCP Server failed to start: {e}")
    finally:
        server_sock.close()
        # ปิด client ทั้งหมด
        with robot_clients_lock:
            for sock in robot_clients:
                try: sock.close()
                except: pass
            robot_clients.clear()
        print("🌐 TCP Server stopped")

# สี (Palette)
WHITE = (255, 255, 255)
SKIN_COLOR = (255, 225, 210)    # ใช้เป็นพื้นหลัง
SHADOW_COLOR = (230, 200, 190)  # สีจมูก/เงา
IRIS_COLOR = (100, 180, 255)
PUPIL_COLOR = (20, 20, 40)
LIP_COLOR = (255, 140, 150)
EYEBROW_COLOR = (60, 40, 30)

# =============================================================
# ส่วนเลือกโมเดล (Model Selection Menu)
# =============================================================

# รายชื่อโมเดลฟรีของ Google Gemini
MODEL_CHOICES = {
    1: {
        "name": "Gemini 2.5 Flash",
        "model_id": "gemini-2.5-flash",
        "keywords": ["gemini-2.5-flash", "2.5-flash"],
        "description": "โมเดลล่าสุด เร็วและฉลาดที่สุด (แนะนำ)"
    },
    2: {
        "name": "Gemini 2.0 Flash",
        "model_id": "gemini-2.0-flash",
        "keywords": ["gemini-2.0-flash", "2.0-flash"],
        "description": "โมเดลรุ่น 2.0 เร็วและเสถียร"
    },
    3: {
        "name": "Gemini 1.5 Flash",
        "model_id": "gemini-1.5-flash",
        "keywords": ["gemini-1.5-flash", "1.5-flash"],
        "description": "โมเดลรุ่น 1.5 เร็ว รองรับ context ยาว"
    },
    4: {
        "name": "Gemini 2.5 Flash-Lite",
        "model_id": "gemini-2.5-flash-lite-preview",
        "keywords": ["gemini-2.5-flash-lite", "2.5-flash-lite"],
        "description": "รุ่นเบาของ 2.5 Flash ประหยัด token"
    },
    5: {
        "name": "Gemini 2.0 Flash-Lite",
        "model_id": "gemini-2.0-flash-lite",
        "keywords": ["gemini-2.0-flash-lite", "2.0-flash-lite"],
        "description": "รุ่นเบาของ 2.0 Flash ประหยัด token"
    },
    6: {
        "name": "Gemini 1.5 Pro",
        "model_id": "gemini-1.5-pro",
        "keywords": ["gemini-1.5-pro", "1.5-pro"],
        "description": "โมเดลรุ่น Pro ฉลาดกว่า Flash แต่ช้ากว่า"
    },
    7: {
        "name": "Gemini 2.5 Pro",
        "model_id": "gemini-2.5-pro",
        "keywords": ["gemini-2.5-pro", "2.5-pro"],
        "description": "โมเดล Pro ล่าสุด ฉลาดที่สุด (rate limit ต่ำ)"
    },
}


def show_model_menu():
    """แสดงเมนูเลือกโมเดลและรอ input จากผู้ใช้"""
    print("\n" + "=" * 60)
    print("🤖  เลือกโมเดล Gemini AI สำหรับสนทนา")
    print("=" * 60)
    
    for num, info in MODEL_CHOICES.items():
        print(f"  {num}. {info['name']:<28} - {info['description']}")
    
    print("-" * 60)
    print(f"  0. แสดงรายชื่อโมเดลทั้งหมดที่ใช้ได้ (List All)")
    print("=" * 60)
    
    while True:
        try:
            choice = input("\n👉 กรุณาเลือกหมายเลขโมเดล (1-7) [default=1]: ").strip()
            
            if choice == "" or choice == "1":
                return MODEL_CHOICES[1]
            
            if choice == "0":
                # แสดงรายชื่อโมเดลทั้งหมดจาก API
                print("\n🔄 กำลังดึงรายชื่อโมเดลจาก Google API...")
                try:
                    all_models = list(genai.list_models())
                    print(f"\n📋 พบโมเดลทั้งหมด {len(all_models)} ตัว:")
                    print("-" * 60)
                    for i, m in enumerate(all_models, 1):
                        supported = ", ".join(m.supported_generation_methods) if hasattr(m, 'supported_generation_methods') else "N/A"
                        print(f"  {i:2d}. {m.name:<45} [{supported}]")
                    print("-" * 60)
                    
                    custom_name = input("\n👉 พิมพ์ชื่อโมเดลที่ต้องการ (เช่น models/gemini-2.5-flash): ").strip()
                    if custom_name:
                        return {
                            "name": custom_name,
                            "model_id": custom_name,
                            "keywords": [custom_name],
                            "description": "Custom model"
                        }
                    else:
                        print("⚠️ ไม่ได้ระบุชื่อ ใช้ค่าเริ่มต้น (Gemini 2.5 Flash)")
                        return MODEL_CHOICES[1]
                except Exception as e:
                    print(f"❌ ไม่สามารถดึงรายชื่อโมเดลได้: {e}")
                    continue
            
            choice_num = int(choice)
            if choice_num in MODEL_CHOICES:
                return MODEL_CHOICES[choice_num]
            else:
                print(f"❌ กรุณาเลือกหมายเลข 0-7 เท่านั้น")
                
        except ValueError:
            print(f"❌ กรุณาพิมพ์ตัวเลข 0-7")
        except KeyboardInterrupt:
            print("\n\n👋 ยกเลิกโปรแกรม")
            sys.exit(0)


def get_vision_model(model_info):
    """สร้าง GenerativeModel จาก model_info ที่เลือก"""
    model_id = model_info["model_id"]
    model_name = model_info["name"]
    
    print(f"\n🔄 กำลังโหลดโมเดล: {model_name} ({model_id})...")
    
    # ลองค้นหาจาก API ก่อน
    try:
        available_models = [m.name for m in genai.list_models()]
        for m_name in available_models:
            if any(k in m_name for k in model_info["keywords"]):
                print(f"✅ พบโมเดล: {m_name}")
                return genai.GenerativeModel(m_name), m_name
    except Exception as e:
        print(f"⚠️ ไม่สามารถค้นหาจาก API: {e}")
    
    # ถ้าหาไม่เจอ ใช้ model_id โดยตรง
    print(f"⚠️ ใช้ชื่อโมเดลโดยตรง: '{model_id}'...")
    return genai.GenerativeModel(model_id), model_id


# =============================================================
# แสดงเมนูเลือกโมเดล
# =============================================================
print("\n🌟 Emotional Robot AI - Multi Model Edition 🌟")
print("   รองรับโมเดล Gemini หลายรุ่น\n")

selected_model_info = show_model_menu()
model, actual_model_name = get_vision_model(selected_model_info)

print(f"\n{'=' * 60}")
print(f"  ✅ ใช้โมเดล: {selected_model_info['name']}")
print(f"  📝 Model ID: {actual_model_name}")
print(f"{'=' * 60}\n")

system_instruction = """
คุณคือเพื่อนสาว AI ที่ร่าเริง กวนนิดๆ และขี้เล่น (อยู่บนหน้าจอหุ่นยนต์)
กฎการตอบ:
1. ตอบเป็นภาษาไทยเสมอ สั้นกระชับ
2. **สำคัญที่สุด**: ต้องเริ่มประโยคด้วย "Tag อารมณ์" เสมอ โดยเลือกให้เข้ากับบริบท หรือตามที่ผู้ใช้สั่ง:
   - [NEUTRAL] ปกติ
   - [HAPPY] ดีใจ/ยิ้ม
   - [SAD] เศร้า
   - [ANGRY] โกรธ
   - [SURPRISED] ตกใจ
   - [SHY] เขิน
   - [POUT] เบะปาก (เช่น เวลางอน หรือไม่เห็นด้วย)
   - [ROLL] มองบน (เช่น เวลาเบื่อ หรือรำคาญแบบขำๆ)
   - [SLEEP] หลับตา (เช่น เวลาทำสมาธิ หรือผู้ใช้สั่งให้หลับตา)
   - [SUS] หรี่ตามองจับผิด (Suspicious)
   - [BLINK] กะพริบตา (ถ้าผู้ใช้สั่ง)

ตัวอย่าง:
User: "ทำหน้าเบะปากใส่หน่อย" -> Model: "[POUT] เชอะ! ไม่ทำให้หรอก... ล้อเล่นน่า"
User: "เบื่อจังเลย" -> Model: "[ROLL] โอ๊ย บ่นอีกแล้วเหรอคะ"
User: "นอนได้แล้ว" -> Model: "[SLEEP] ครอกฟี่... หลับแล้วค่ะ"
"""

initial_history = [
    {"role": "user", "parts": [system_instruction]},
    {"role": "model", "parts": ["[HAPPY] พร้อมแสดงสีหน้าแล้วค่ะ! สั่งมาได้เลย"]}
]
chat = model.start_chat(history=initial_history)

# ---------------------------------------------------------
# ส่วนวาดภาพ AI (Face Only / Zoomed Version)
# ---------------------------------------------------------
class Avatar:
    def __init__(self, screen_width, screen_height):
        self.update_dimensions(screen_width, screen_height)
        
        # สถานะ
        self.emotion = "NEUTRAL"
        self.target_face_x = 0 
        self.target_face_y = 0 
        self.current_face_x = 0
        self.current_face_y = 0
        
        self.blink_timer = 0
        self.eye_open_ratio = 1.0
        self.current_mouth_h = 0
        
        # (scale ถูกคำนวณจาก update_dimensions แล้ว)

    def update_dimensions(self, width, height):
        self.width = width
        self.height = height
        self.center_x = width // 2
        self.center_y = height // 2
        # Dynamic adjust scale - ใช้เฉลี่ยจาก width และ height
        # เพื่อให้ Avatar สมส่วนกับจอทุกขนาดและทุก aspect ratio
        # (Base: 480x320 = scale 1.8)
        scale_h = (height / 320.0) * 1.8
        scale_w = (width / 480.0) * 1.8
        self.scale = (scale_h + scale_w) / 2

    def set_emotion(self, emotion_tag):
        self.emotion = emotion_tag
        
        if emotion_tag != "SLEEP" and emotion_tag != "SUS":
             pass 

        # ตั้งค่าท่าทางตามอารมณ์
        if emotion_tag == "HAPPY":
            self.target_face_x, self.target_face_y = 0, -5
        elif emotion_tag == "SAD":
            self.target_face_x, self.target_face_y = 0, 10
        elif emotion_tag == "SHY":
            self.target_face_x, self.target_face_y = 10, 5
        elif emotion_tag == "SURPRISED":
            self.target_face_x, self.target_face_y = 0, -5
        elif emotion_tag == "ANGRY":
            self.target_face_x, self.target_face_y = 0, 5
        elif emotion_tag == "POUT":
            self.target_face_x, self.target_face_y = 5, -5 
        elif emotion_tag == "ROLL":
            self.target_face_x, self.target_face_y = 0, -2
        elif emotion_tag == "BLINK":
            self.blink_timer = 200 
            self.emotion = "NEUTRAL" 
        else:
            self.target_face_x, self.target_face_y = 0, 0

    def update(self):
        # 1. Blink Logic
        self.blink_timer += 1
        blink_interval = 180 
        if self.emotion == "SURPRISED" or self.emotion == "ROLL": 
            blink_interval = 300 
        
        if self.blink_timer > blink_interval:
            self.eye_open_ratio -= 0.15
            if self.eye_open_ratio < 0:
                self.eye_open_ratio = 0
                if self.blink_timer > blink_interval + 10:
                     self.blink_timer = 0
        else:
            target_open = 1.0
            if self.emotion == "SURPRISED": target_open = 1.2
            elif self.emotion == "ANGRY" or self.emotion == "SHY": target_open = 0.8
            elif self.emotion == "SUS": target_open = 0.4 
            elif self.emotion == "SLEEP": target_open = 0.0 
            
            self.eye_open_ratio += (target_open - self.eye_open_ratio) * 0.1

        # 2. Head Movement
        idle_x = math.sin(time.time()) * 1.5
        idle_y = math.cos(time.time() * 2) * 1.5
        
        self.current_face_x += (self.target_face_x + idle_x - self.current_face_x) * 0.1
        self.current_face_y += (self.target_face_y + idle_y - self.current_face_y) * 0.1

        # 3. Mouth Animation
        target_h = 5
        if is_ai_speaking:
            target_h = (math.sin(time.time() * 18) * 0.5 + 0.5) * 30 + 5 
        
        if self.emotion == "HAPPY": target_h += 3
        
        if self.emotion == "POUT": target_h = min(target_h, 8)

        self.current_mouth_h += (target_h - self.current_mouth_h) * 0.2

    def draw_eyebrow(self, screen, x, y, is_left):
        tilt = 0
        y_offset = 0
        
        if self.emotion == "ANGRY": 
            tilt = 15 if is_left else -15
            y_offset = 8
        elif self.emotion == "SAD":
            tilt = -10 if is_left else 10
            y_offset = -4
        elif self.emotion == "SURPRISED":
            y_offset = -12
        elif self.emotion == "HAPPY":
            y_offset = -4
        elif self.emotion == "POUT" or self.emotion == "SUS":
            tilt = 5 if is_left else -5 
            y_offset = 5
        elif self.emotion == "ROLL":
            y_offset = -2

        start_x = x - 22 * self.scale
        end_x = x + 22 * self.scale
        
        y_l = y + y_offset + (tilt if not is_left else 0)
        y_r = y + y_offset + (-tilt if is_left else 0)
        
        pygame.draw.line(screen, EYEBROW_COLOR, (start_x, y_l), (end_x, y_r), int(5*self.scale))

    def draw_eye(self, screen, x, y, is_left):
        eye_w = 60 * self.scale
        eye_h = 70 * self.scale * self.eye_open_ratio
        
        # เบ้าตา
        pygame.draw.ellipse(screen, WHITE, (x - eye_w/2, y - eye_h/2, eye_w, eye_h))
        
        if self.eye_open_ratio > 0.1:
            iris_size = 28 * self.scale
            
            mx, my = pygame.mouse.get_pos()
            look_x = (mx - self.center_x) / 30 
            look_y = (my - self.center_y) / 30
            
            if self.emotion == "SHY": look_y += 8; look_x += 8
            elif self.emotion == "SAD": look_y += 5
            elif self.emotion == "ROLL": look_y = -12; look_x = 0 
            elif self.emotion == "POUT": look_x += 5 if is_left else -5 
            
            look_x = max(-10, min(10, look_x))
            look_y = max(-15, min(10, look_y))

            pygame.draw.circle(screen, IRIS_COLOR, (x + look_x, y + look_y), iris_size)
            pygame.draw.circle(screen, PUPIL_COLOR, (x + look_x, y + look_y), iris_size * 0.5)
            # แววตา
            pygame.draw.circle(screen, WHITE, (x + look_x + iris_size*0.3, y + look_y - iris_size*0.3), iris_size * 0.25)

        if self.eye_open_ratio < 1.0:
            rect_h = (40*self.scale) + (eye_h/2 * (1-self.eye_open_ratio))
            pygame.draw.rect(screen, SKIN_COLOR, (x - eye_w/2, y - eye_h/2 - 40*self.scale, eye_w, rect_h))
        
        self.draw_eyebrow(screen, x, y - 45*self.scale, is_left)

    def draw(self, screen):
        cx = self.center_x + self.current_face_x
        cy = self.center_y + self.current_face_y
        s = self.scale

        # --- ตา ---
        eye_sep = 80 * s 
        eye_y_pos = cy - 20 * s
        self.draw_eye(screen, cx - eye_sep, eye_y_pos, True)
        self.draw_eye(screen, cx + eye_sep, eye_y_pos, False)

        # --- จมูก ---
        pygame.draw.circle(screen, SHADOW_COLOR, (cx, cy + 30*s), 6*s)

        # --- แก้มแดง ---
        if self.emotion == "SHY" or self.emotion == "HAPPY":
            pygame.draw.circle(screen, (255, 180, 180), (cx - 105*s, cy + 40*s), 20*s)
            pygame.draw.circle(screen, (255, 180, 180), (cx + 105*s, cy + 40*s), 20*s)

        # --- ปาก ---
        mouth_y = cy + 60*s
        mw = 50 * s
        mh = self.current_mouth_h * s
        
        if is_ai_speaking:
            rect = (cx - mw/2, mouth_y - mh/2, mw, mh)
            pygame.draw.ellipse(screen, (100, 50, 50), rect)
            if mh > 15:
                pygame.draw.circle(screen, LIP_COLOR, (cx, mouth_y + mh/2 - 8), 12)
        else:
            if self.emotion == "HAPPY":
                pygame.draw.arc(screen, (160, 90, 90), (cx - 25*s, mouth_y - 15*s, 50*s, 30*s), math.pi, 2*math.pi, 4)
            elif self.emotion == "SAD":
                pygame.draw.arc(screen, (160, 90, 90), (cx - 25*s, mouth_y + 5*s, 50*s, 30*s), 0, math.pi, 4)
            elif self.emotion == "SURPRISED":
                pygame.draw.circle(screen, (100, 50, 50), (cx, mouth_y + 5), 8*s, 2)
            elif self.emotion == "ANGRY" or self.emotion == "SUS":
                pygame.draw.line(screen, (160, 90, 90), (cx - 15*s, mouth_y + 5), (cx + 15*s, mouth_y + 5), 3)
            elif self.emotion == "POUT":
                pygame.draw.arc(screen, (160, 90, 90), (cx - 15*s, mouth_y - 5*s, 30*s, 20*s), 0, math.pi, 4)
            elif self.emotion == "ROLL":
                pygame.draw.line(screen, (160, 90, 90), (cx - 15*s, mouth_y), (cx + 15*s, mouth_y), 3)
            else:
                pygame.draw.arc(screen, (160, 90, 90), (cx - 15*s, mouth_y - 5*s, 30*s, 15*s), math.pi, 2*math.pi, 3)

# ---------------------------------------------------------
# Backend Threads
# ---------------------------------------------------------
def depth_to_colormap(depth_frame, max_depth=10000):
    """Convert Depth Frame to Colormap for display"""
    depth_normalized = np.clip(depth_frame, 0, max_depth)
    depth_normalized = (depth_normalized / max_depth * 255).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    return depth_colormap


def get_distance_at_point(depth_frame, x, y, region_size=5):
    """Calculate average distance at a point (unit: millimeters)"""
    h, w = depth_frame.shape

    x1 = max(0, x - region_size)
    x2 = min(w, x + region_size)
    y1 = max(0, y - region_size)
    y2 = min(h, y + region_size)

    region = depth_frame[y1:y2, x1:x2]
    valid_depths = region[region > 0]

    if len(valid_depths) > 0:
        return int(np.median(valid_depths))
    return 0


def camera_capture_thread():
    global latest_frame, latest_depth_frame, center_distance_mm, program_running

    # --- Oak-D Lite Pipeline Setup (DepthAI 2.x API) ---
    pipeline = dai.Pipeline()

    # ===== Setup RGB Camera =====
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(640, 480)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

    # ===== Setup Mono Cameras for Stereo Depth =====
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)

    mono_right = pipeline.create(dai.node.MonoCamera)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    # ===== Setup Stereo Depth =====
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(640, 400)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(False)

    # Link mono cameras to stereo
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    # ===== Create XLinkOut for RGB =====
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    # ===== Create XLinkOut for Depth =====
    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    print("📷 กำลังเชื่อมต่อกล้อง Oak-D Lite...")
    try:
        with dai.Device(pipeline) as device:
            print(f"📷 เชื่อมต่อ Oak-D Lite สำเร็จ!")
            try:
                print(f"   Device name: {device.getDeviceName()}")
                print(f"   USB speed: {device.getUsbSpeed()}")
            except: pass
            
            q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
            
            frame_count = 0
            print("📷 กำลังรอเฟรมแรกจากกล้อง...")
            
            while program_running:
                in_rgb = q_rgb.tryGet()
                in_depth = q_depth.tryGet()

                if in_rgb is not None:
                    frame = in_rgb.getCvFrame()
                    # Resize ให้เป็น 640x480
                    if frame.shape[1] != 640 or frame.shape[0] != 480:
                        frame = cv2.resize(frame, (640, 480))
                    # Update global latest_frame (ไม่ flip เพื่อให้ตรงกับ Depth)
                    latest_frame = frame
                    
                    frame_count += 1
                    if frame_count == 1:
                        print(f"✅ ได้รับเฟรมแรก! ขนาด: {frame.shape}")
                    elif frame_count % 300 == 0:
                        print(f"📷 Frame count: {frame_count}")

                if in_depth is not None:
                    depth_frame = in_depth.getFrame()
                    latest_depth_frame = depth_frame
                    
                    # คำนวณระยะทางจุดกลาง
                    h, w = depth_frame.shape
                    center_x, center_y = w // 2, h // 2
                    center_distance_mm = get_distance_at_point(depth_frame, center_x, center_y)

                if in_rgb is None and in_depth is None:
                    time.sleep(0.001)
                    
    except Exception as e:
        print(f"⚠️ Error with Oak-D Camera: {e}")
        import traceback
        traceback.print_exc()
        program_running = False

def play_audio(filename):
    global is_ai_speaking
    try:
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        is_ai_speaking = True
        while pygame.mixer.music.get_busy() and not stop_audio_event.is_set():
            time.sleep(0.05)
        if stop_audio_event.is_set():
            pygame.mixer.music.stop()
    except: pass
    finally:
        is_ai_speaking = False
        stop_audio_event.clear()
        try: os.remove(filename)
        except: pass

def speak_async(text):
    text = text.replace("*", "").strip()
    if not text: return
    print(f"🤖 AI: {text}")
    
    if is_ai_speaking:
        stop_audio_event.set()
        time.sleep(0.2) 

    try:
        tts = gTTS(text=text, lang='th')
        filename = f"voice_{int(time.time())}.mp3"
        tts.save(filename)
        threading.Thread(target=play_audio, args=(filename,)).start()
    except Exception as e:
        print(f"TTS Error: {e}")

def add_chat_message(role, text):
    """เพิ่มข้อความแชทลง global list (thread-safe)"""
    global new_chat_flag
    timestamp = time.strftime("%H:%M:%S")
    with chat_messages_lock:
        chat_messages.append({'role': role, 'text': text, 'time': timestamp})
        # เก็บแค่ 200 ข้อความล่าสุด
        if len(chat_messages) > 200:
            chat_messages.pop(0)
        new_chat_flag = True


def bot_logic_thread():
    global current_emotion, avatar_instance
    r = sr.Recognizer()
    r.energy_threshold = 3000
    r.dynamic_energy_threshold = True

    speak_async("สวัสดีค่ะ มีอะไรให้ช่วยไหมคะ")
    add_chat_message('ai', '[HAPPY] สวัสดีค่ะ มีอะไรให้ช่วยไหมคะ')
    
    while program_running:
        try:
            with sr.Microphone() as source:
                try:
                    audio = r.listen(source, timeout=3, phrase_time_limit=8)
                    try:
                        text = r.recognize_google(audio, language='th-TH')
                        if is_ai_speaking and text:
                            stop_audio_event.set()
                        
                        print(f"👤 คุณ: {text}")
                        add_chat_message('user', text)
                        
                        if "ปิดโปรแกรม" in text:
                            speak_async("บ๊ายบาย")
                            add_chat_message('ai', 'บ๊ายบาย 👋')
                            time.sleep(3)
                            os._exit(0)

                        content = [text]
                        if latest_frame is not None:
                            # Resize for API efficiency
                            small_frame = cv2.resize(latest_frame, (320, 240)) 
                            rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                            pil_img = Image.fromarray(rgb)
                            content.append(pil_img)

                        response = chat.send_message(content)
                        raw_text = response.text.strip()
                        
                        emotion_match = re.search(r'\[(HAPPY|SAD|ANGRY|SURPRISED|SHY|NEUTRAL|POUT|ROLL|SLEEP|SUS|BLINK)\]', raw_text)
                        
                        clean_text = raw_text
                        if emotion_match:
                            emo_tag = emotion_match.group(1)
                            if avatar_instance: 
                                avatar_instance.set_emotion(emo_tag)
                            print(f"✨ Emotion: {emo_tag}")
                            clean_text = re.sub(r'\[.*?\]', '', raw_text).strip()
                            # ส่งอารมณ์ไปยัง robot client
                            send_emotion_to_robot(emo_tag)
                        
                        add_chat_message('ai', raw_text)
                        speak_async(clean_text)

                    except sr.UnknownValueError: pass
                    except sr.RequestError: pass
                except sr.WaitTimeoutError: pass
        except Exception as e:
            print(f"Logic Error: {e}")
            time.sleep(1)

avatar_instance = None

# ---------------------------------------------------------
# ตรวจหาจอภาพ (Monitor Detection) - Windows API
# ---------------------------------------------------------
class MONITORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.wintypes.DWORD),
        ("rcMonitor", ctypes.wintypes.RECT),
        ("rcWork", ctypes.wintypes.RECT),
        ("dwFlags", ctypes.wintypes.DWORD),
    ]

def get_monitors():
    """ดึงข้อมูลจอภาพทั้งหมดที่เชื่อมต่อกับ Windows PC"""
    monitors = []
    
    def callback(hMonitor, hdcMonitor, lprcMonitor, dwData):
        info = MONITORINFO()
        info.cbSize = ctypes.sizeof(MONITORINFO)
        ctypes.windll.user32.GetMonitorInfoW(hMonitor, ctypes.byref(info))
        monitors.append({
            'x': info.rcMonitor.left,
            'y': info.rcMonitor.top,
            'width': info.rcMonitor.right - info.rcMonitor.left,
            'height': info.rcMonitor.bottom - info.rcMonitor.top,
            'is_primary': bool(info.dwFlags & 1)
        })
        return True
    
    MONITORENUMPROC = ctypes.WINFUNCTYPE(
        ctypes.c_bool,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.wintypes.RECT),
        ctypes.c_double
    )
    ctypes.windll.user32.EnumDisplayMonitors(
        None, None, MONITORENUMPROC(callback), 0
    )
    return monitors


# =============================================================
# Dashboard GUI (Tkinter) - แสดงบนจอหลัก
# =============================================================
class Dashboard:
    # สี Dark Theme
    BG_DARK = '#0d1117'
    BG_PANEL = '#161b22'
    BG_HEADER = '#1a1f2e'
    BG_CHAT_USER = '#1f3a5f'
    BG_CHAT_AI = '#1a2332'
    BG_ALARM = '#d32f2f'
    FG_WHITE = '#e6edf3'
    FG_GRAY = '#8b949e'
    FG_ACCENT = '#58a6ff'
    FG_GREEN = '#3fb950'
    FG_RED = '#f85149'
    FG_YELLOW = '#d29922'
    
    def __init__(self, primary_monitor, model_choices, current_model_info):
        self.root = tk.Tk()
        self.root.title("🤖 Emotional Robot AI Dashboard")
        self.root.configure(bg=self.BG_DARK)
        
        # ตำแหน่งบนจอหลัก (Primary Monitor)
        x, y = primary_monitor['x'], primary_monitor['y']
        w, h = primary_monitor['width'], primary_monitor['height']
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        self.root.state('zoomed')  # Maximize บน Windows
        
        self.model_choices = model_choices
        self.current_model_info = current_model_info
        self.alarm_visible = False
        self._chat_msg_count = 0
        
        # เก็บ reference ของ PhotoImage เพื่อไม่ให้ถูก garbage collected
        self._rgb_photo = None
        self._depth_photo = None
        
        # สร้าง UI
        self._create_styles()
        self._create_header()
        self._create_body()
        self._create_alarm_bar()
    
    def _create_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Dark.TFrame', background=self.BG_DARK)
        style.configure('Panel.TFrame', background=self.BG_PANEL)
        style.configure('Header.TFrame', background=self.BG_HEADER)
        style.configure('Dark.TLabel', background=self.BG_DARK, foreground=self.FG_WHITE,
                        font=('Segoe UI', 10))
        style.configure('Header.TLabel', background=self.BG_HEADER, foreground=self.FG_WHITE,
                        font=('Segoe UI', 14, 'bold'))
        style.configure('Status.TLabel', background=self.BG_HEADER, foreground=self.FG_GREEN,
                        font=('Segoe UI', 9))
        style.configure('CamTitle.TLabel', background=self.BG_PANEL, foreground=self.FG_ACCENT,
                        font=('Segoe UI', 11, 'bold'))
        style.configure('Dist.TLabel', background=self.BG_PANEL, foreground=self.FG_YELLOW,
                        font=('Segoe UI', 10))
        style.configure('DistDanger.TLabel', background=self.BG_PANEL, foreground=self.FG_RED,
                        font=('Segoe UI', 10, 'bold'))
        style.configure('ChatTitle.TLabel', background=self.BG_PANEL, foreground=self.FG_ACCENT,
                        font=('Segoe UI', 12, 'bold'))
        style.configure('Model.TCombobox', font=('Segoe UI', 10))
    
    def _create_header(self):
        header = ttk.Frame(self.root, style='Header.TFrame', height=50)
        header.pack(fill='x', padx=0, pady=0)
        header.pack_propagate(False)
        
        # Title
        ttk.Label(header, text="🤖 Emotional Robot AI Dashboard",
                  style='Header.TLabel').pack(side='left', padx=15, pady=10)
        
        # Model selector
        model_frame = ttk.Frame(header, style='Header.TFrame')
        model_frame.pack(side='right', padx=15, pady=8)
        
        ttk.Label(model_frame, text="โมเดล:", style='Status.TLabel').pack(side='left', padx=(0, 5))
        
        model_names = [info['name'] for info in self.model_choices.values()]
        self.model_var = tk.StringVar(value=self.current_model_info['name'])
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var,
                                        values=model_names, state='readonly',
                                        width=22, font=('Segoe UI', 10))
        self.model_combo.pack(side='left', padx=(0, 8))
        
        self.model_btn = tk.Button(model_frame, text="🔄 เปลี่ยน", command=self._on_model_change,
                                   bg='#238636', fg='white', font=('Segoe UI', 9, 'bold'),
                                   relief='flat', padx=10, pady=2, cursor='hand2')
        self.model_btn.pack(side='left')
        
        # Status
        self.status_label = ttk.Label(header, text="● Online", style='Status.TLabel')
        self.status_label.pack(side='right', padx=15)
    
    def _create_body(self):
        body = ttk.Frame(self.root, style='Dark.TFrame')
        body.pack(fill='both', expand=True, padx=8, pady=(5, 0))
        
        # ===== ฝั่งซ้าย: กล้อง =====
        cam_frame = ttk.Frame(body, style='Panel.TFrame')
        cam_frame.pack(side='left', fill='both', expand=True, padx=(0, 4))
        
        # -- RGB Camera --
        rgb_header = ttk.Frame(cam_frame, style='Panel.TFrame')
        rgb_header.pack(fill='x', padx=10, pady=(8, 2))
        ttk.Label(rgb_header, text="📷 RGB Camera (Oak-D Lite)",
                  style='CamTitle.TLabel').pack(side='left')
        self.rgb_dist_label = ttk.Label(rgb_header, text="Distance: --", style='Dist.TLabel')
        self.rgb_dist_label.pack(side='right')
        
        self.rgb_canvas = tk.Canvas(cam_frame, bg='#000000', highlightthickness=0)
        self.rgb_canvas.pack(fill='both', expand=True, padx=10, pady=(2, 5))
        
        # -- Depth Camera --
        depth_header = ttk.Frame(cam_frame, style='Panel.TFrame')
        depth_header.pack(fill='x', padx=10, pady=(5, 2))
        ttk.Label(depth_header, text="🌊 Depth Camera (Stereo)",
                  style='CamTitle.TLabel').pack(side='left')
        self.depth_dist_label = ttk.Label(depth_header, text="", style='Dist.TLabel')
        self.depth_dist_label.pack(side='right')
        
        self.depth_canvas = tk.Canvas(cam_frame, bg='#000000', highlightthickness=0)
        self.depth_canvas.pack(fill='both', expand=True, padx=10, pady=(2, 8))
        
        # ===== ฝั่งขวา: Chat Log =====
        chat_outer = ttk.Frame(body, style='Panel.TFrame', width=420)
        chat_outer.pack(side='right', fill='both', padx=(4, 0))
        chat_outer.pack_propagate(False)
        
        # Chat header
        chat_header = ttk.Frame(chat_outer, style='Panel.TFrame')
        chat_header.pack(fill='x', padx=10, pady=(8, 5))
        ttk.Label(chat_header, text="💬 Chat Log",
                  style='ChatTitle.TLabel').pack(side='left')
        self.emotion_label = ttk.Label(chat_header, text="😊 NEUTRAL",
                                       style='Status.TLabel')
        self.emotion_label.pack(side='right')
        
        # Chat text area
        self.chat_text = scrolledtext.ScrolledText(
            chat_outer, wrap='word', state='disabled',
            bg=self.BG_DARK, fg=self.FG_WHITE,
            font=('Segoe UI', 10), relief='flat',
            insertbackground=self.FG_WHITE,
            selectbackground=self.FG_ACCENT,
            padx=10, pady=5
        )
        self.chat_text.pack(fill='both', expand=True, padx=10, pady=(0, 8))
        
        # กำหนดสี tag สำหรับ user/ai
        self.chat_text.tag_configure('user_name', foreground='#58a6ff', font=('Segoe UI', 10, 'bold'))
        self.chat_text.tag_configure('ai_name', foreground='#3fb950', font=('Segoe UI', 10, 'bold'))
        self.chat_text.tag_configure('system_name', foreground='#d29922', font=('Segoe UI', 10, 'bold'))
        self.chat_text.tag_configure('timestamp', foreground='#484f58', font=('Segoe UI', 8))
        self.chat_text.tag_configure('emotion_tag', foreground='#f0883e', font=('Segoe UI', 10, 'bold'))
    
    def _create_alarm_bar(self):
        self.alarm_frame = tk.Frame(self.root, bg=self.BG_ALARM, height=50)
        self.alarm_label = tk.Label(self.alarm_frame,
                                    text="⚠️ PROXIMITY ALARM: อยู่ใกล้เกินไป! กรุณาถอยออก!",
                                    bg=self.BG_ALARM, fg='white',
                                    font=('Segoe UI', 14, 'bold'))
        self.alarm_label.pack(expand=True)
        # alarm bar จะถูก pack/pack_forget ตามสถานะ
    
    def _on_model_change(self):
        """เปลี่ยนโมเดลจาก dropdown"""
        global model, chat, selected_model_info, actual_model_name
        new_model_name = self.model_var.get()
        
        # หา model_info จากชื่อ
        new_info = None
        for info in self.model_choices.values():
            if info['name'] == new_model_name:
                new_info = info
                break
        
        if new_info is None or new_info['name'] == selected_model_info['name']:
            return
        
        try:
            self.model_btn.configure(text="⏳ กำลังเปลี่ยน...", state='disabled')
            self.root.update()
            
            new_model, new_actual_name = get_vision_model(new_info)
            
            # อัปเดต globals
            model = new_model
            selected_model_info = new_info
            actual_model_name = new_actual_name
            
            # สร้าง chat ใหม่
            chat = model.start_chat(history=[
                {"role": "user", "parts": [system_instruction]},
                {"role": "model", "parts": ["[HAPPY] พร้อมแสดงสีหน้าแล้วค่ะ! สั่งมาได้เลย"]}
            ])
            
            self.current_model_info = new_info
            add_chat_message('system', f'🔄 เปลี่ยนโมเดลเป็น: {new_info["name"]}')
            print(f"✅ เปลี่ยนโมเดลเป็น: {new_info['name']} ({new_actual_name})")
            
        except Exception as e:
            add_chat_message('system', f'❌ เปลี่ยนโมเดลไม่สำเร็จ: {e}')
            print(f"❌ Model change error: {e}")
        finally:
            self.model_btn.configure(text="🔄 เปลี่ยน", state='normal')
    
    def update_rgb_frame(self, frame, is_too_close=False):
        """อัปเดตภาพ RGB บน Canvas"""
        if frame is None:
            return
        try:
            display = frame.copy()
            
            # วาด overlay บน frame
            border_color = (0, 0, 255) if is_too_close else (0, 255, 0)
            cv2.rectangle(display, (10, 10), (display.shape[1]-10, display.shape[0]-10), border_color, 2)
            
            # Crosshair
            h_f, w_f = display.shape[:2]
            cx, cy = w_f // 2, h_f // 2
            cv2.drawMarker(display, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            
            # Too close overlay
            if is_too_close:
                overlay = display.copy()
                cv2.rectangle(overlay, (0, h_f - 50), (w_f, h_f), (0, 0, 200), -1)
                cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
                cv2.putText(display, "!! TOO CLOSE !!", (20, h_f - 18),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # ปรับขนาดให้พอดี canvas
            canvas_w = self.rgb_canvas.winfo_width()
            canvas_h = self.rgb_canvas.winfo_height()
            if canvas_w > 1 and canvas_h > 1:
                display = cv2.resize(display, (canvas_w, canvas_h))
            
            # แปลง BGR -> RGB -> PIL -> PhotoImage
            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            self._rgb_photo = ImageTk.PhotoImage(pil_img)
            self.rgb_canvas.create_image(0, 0, anchor='nw', image=self._rgb_photo)
        except Exception:
            pass
    
    def update_depth_frame(self, depth_frame, is_too_close=False):
        """อัปเดตภาพ Depth บน Canvas"""
        if depth_frame is None:
            return
        try:
            colormap = depth_to_colormap(depth_frame)
            h_d, w_d = depth_frame.shape
            cx, cy = w_d // 2, h_d // 2
            
            # Crosshair
            cv2.circle(colormap, (cx, cy), 5, (255, 255, 255), -1)
            cv2.drawMarker(colormap, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
            
            # Too close overlay
            if is_too_close:
                cv2.putText(colormap, "!! TOO CLOSE !!", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # ปรับขนาด
            canvas_w = self.depth_canvas.winfo_width()
            canvas_h = self.depth_canvas.winfo_height()
            if canvas_w > 1 and canvas_h > 1:
                colormap = cv2.resize(colormap, (canvas_w, canvas_h))
            
            rgb = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            self._depth_photo = ImageTk.PhotoImage(pil_img)
            self.depth_canvas.create_image(0, 0, anchor='nw', image=self._depth_photo)
        except Exception:
            pass
    
    def update_distance(self, distance_mm, is_too_close=False):
        """อัปเดตข้อมูลระยะทาง"""
        if distance_mm > 0:
            dist_text = f"Distance: {distance_mm} mm ({distance_mm/1000:.2f} m)"
            style = 'DistDanger.TLabel' if is_too_close else 'Dist.TLabel'
            self.rgb_dist_label.configure(text=dist_text, style=style)
            self.depth_dist_label.configure(text=dist_text, style=style)
        else:
            self.rgb_dist_label.configure(text="Distance: --", style='Dist.TLabel')
            self.depth_dist_label.configure(text="", style='Dist.TLabel')
    
    def update_chat(self):
        """อัปเดต chat log จาก global chat_messages"""
        global new_chat_flag
        if not new_chat_flag:
            return
        
        with chat_messages_lock:
            msgs = list(chat_messages)
            new_chat_flag = False
        
        if len(msgs) == self._chat_msg_count:
            return
        
        # เพิ่มเฉพาะข้อความใหม่
        new_msgs = msgs[self._chat_msg_count:]
        self._chat_msg_count = len(msgs)
        
        self.chat_text.configure(state='normal')
        for msg in new_msgs:
            timestamp = msg['time']
            role = msg['role']
            text = msg['text']
            
            self.chat_text.insert('end', f"[{timestamp}] ", 'timestamp')
            
            if role == 'user':
                self.chat_text.insert('end', "👤 คุณ: ", 'user_name')
                self.chat_text.insert('end', f"{text}\n")
            elif role == 'ai':
                self.chat_text.insert('end', "🤖 AI: ", 'ai_name')
                # แยก emotion tag ออกมาแสดงสี
                emotion_match = re.search(r'\[(\w+)\]', text)
                if emotion_match:
                    tag = emotion_match.group(0)
                    rest = text.replace(tag, '').strip()
                    self.chat_text.insert('end', f"{tag} ", 'emotion_tag')
                    self.chat_text.insert('end', f"{rest}\n")
                    # อัปเดต emotion label
                    emo = emotion_match.group(1)
                    emoji_map = {'HAPPY': '😊', 'SAD': '😢', 'ANGRY': '😠',
                                'SURPRISED': '😲', 'SHY': '😳', 'NEUTRAL': '😐',
                                'POUT': '😤', 'ROLL': '🙄', 'SLEEP': '😴',
                                'SUS': '🤨', 'BLINK': '😉'}
                    emoji = emoji_map.get(emo, '🤖')
                    self.emotion_label.configure(text=f"{emoji} {emo}")
                else:
                    self.chat_text.insert('end', f"{text}\n")
            elif role == 'system':
                self.chat_text.insert('end', "⚙️ ", 'system_name')
                self.chat_text.insert('end', f"{text}\n")
        
        self.chat_text.configure(state='disabled')
        self.chat_text.see('end')  # Auto-scroll ลงล่าง
    
    def show_alarm(self, distance_mm):
        """แสดง alarm bar"""
        if not self.alarm_visible:
            distance_cm = distance_mm / 10
            self.alarm_label.configure(
                text=f"⚠️ PROXIMITY ALARM: อยู่ใกล้เกินไป! ({distance_cm:.0f} cm) กรุณาถอยออก!"
            )
            self.alarm_frame.pack(fill='x', padx=0, pady=0, side='bottom')
            self.alarm_visible = True
    
    def hide_alarm(self):
        """ซ่อน alarm bar"""
        if self.alarm_visible:
            self.alarm_frame.pack_forget()
            self.alarm_visible = False
    
    def update(self):
        """เรียกจาก main loop แทน root.mainloop()"""
        try:
            self.root.update()
        except tk.TclError:
            pass
    
    def destroy(self):
        try:
            self.root.destroy()
        except:
            pass


# =============================================================
# Main Function
# =============================================================
def main():
    global program_running, avatar_instance, last_proximity_warning_time
    
    # --- ตั้ง DPI Awareness ---
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass
    
    # --- ตรวจหาจอภาพ ---
    monitors = get_monitors()
    print(f"\n🖥️  ตรวจพบจอภาพ {len(monitors)} จอ:")
    for i, m in enumerate(monitors):
        primary_tag = " (Primary)" if m['is_primary'] else ""
        print(f"   จอที่ {i+1}: {m['width']}x{m['height']} ตำแหน่ง ({m['x']}, {m['y']}){primary_tag}")
    
    # หา Primary Monitor (สำหรับ Dashboard)
    primary_monitor = None
    second_monitor = None
    for m in monitors:
        if m['is_primary']:
            primary_monitor = m
        else:
            if second_monitor is None:
                second_monitor = m
    
    if primary_monitor is None:
        primary_monitor = monitors[0] if monitors else {'x': 0, 'y': 0, 'width': 1920, 'height': 1080}
    
    # จอสำหรับ Avatar (ใช้จอที่ 2 ถ้ามี ไม่งั้นใช้จอหลัก)
    if second_monitor:
        avatar_monitor = second_monitor
        print(f"\n🖥️  Avatar → จอที่ 2: {avatar_monitor['width']}x{avatar_monitor['height']}")
        print(f"🖥️  Dashboard → จอหลัก: {primary_monitor['width']}x{primary_monitor['height']}")
    else:
        avatar_monitor = primary_monitor
        print(f"\n⚠️  มีจอเดียว ทั้ง Avatar และ Dashboard จะแสดงที่จอเดียวกัน")
    
    # --- สร้าง Dashboard (Tkinter) บนจอหลัก ---
    dashboard = Dashboard(primary_monitor, MODEL_CHOICES, selected_model_info)
    add_chat_message('system', f'🚀 เริ่มระบบด้วยโมเดล: {selected_model_info["name"]}')
    
    # --- ตั้งตำแหน่ง Pygame ไปจอ Avatar ---
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{avatar_monitor['x']},{avatar_monitor['y']}"
    
    pygame.init()
    pygame.mixer.init()
    
    AVT_W = avatar_monitor['width']
    AVT_H = avatar_monitor['height']
    is_fullscreen = True
    
    screen = pygame.display.set_mode((AVT_W, AVT_H), pygame.NOFRAME)
    pygame.display.set_caption(f"Gemini Avatar (Oak-D) - {selected_model_info['name']}")
    clock = pygame.time.Clock()
    
    avatar_instance = Avatar(AVT_W, AVT_H)
    
    # Start threads
    threading.Thread(target=camera_capture_thread, daemon=True).start()
    threading.Thread(target=bot_logic_thread, daemon=True).start()
    threading.Thread(target=tcp_server_thread, daemon=True).start()
    
    print(f"\n--- เริ่มต้นระบบ Emotional Robot AI ---")
    print(f"🤖 โมเดล: {selected_model_info['name']} ({actual_model_name})")
    print(f"🖥️  Avatar: {AVT_W}x{AVT_H} ที่ ({avatar_monitor['x']}, {avatar_monitor['y']})")
    print(f"🖥️  Dashboard: จอหลัก ({primary_monitor['width']}x{primary_monitor['height']})")
    print("กด 'F' สลับ fullscreen/window Avatar, กด 'ESC' เพื่อออก")

    while program_running:
        # === Pygame Events (Avatar) ===
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                program_running = False
            
            if event.type == pygame.VIDEORESIZE:
                if not is_fullscreen:
                    AVT_W, AVT_H = event.w, event.h
                    screen = pygame.display.set_mode((AVT_W, AVT_H), pygame.RESIZABLE)
                    avatar_instance.update_dimensions(AVT_W, AVT_H)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    program_running = False
                
                if event.key == pygame.K_f:
                    is_fullscreen = not is_fullscreen
                    if is_fullscreen:
                        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{avatar_monitor['x']},{avatar_monitor['y']}"
                        screen = pygame.display.set_mode(
                            (avatar_monitor['width'], avatar_monitor['height']), pygame.NOFRAME
                        )
                        avatar_instance.update_dimensions(avatar_monitor['width'], avatar_monitor['height'])
                    else:
                        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{avatar_monitor['x'] + 50},{avatar_monitor['y'] + 50}"
                        screen = pygame.display.set_mode((480, 320), pygame.RESIZABLE)
                        avatar_instance.update_dimensions(480, 320)

        # === วาด Avatar (Pygame) ===
        screen.fill(SKIN_COLOR)
        avatar_instance.update()
        avatar_instance.draw(screen)
        pygame.display.flip()

        # === ตรวจสอบ Proximity ===
        is_too_close = False
        if 0 < center_distance_mm < PROXIMITY_THRESHOLD_MM:
            is_too_close = True
            current_time = time.time()
            if current_time - last_proximity_warning_time > PROXIMITY_COOLDOWN_SEC:
                last_proximity_warning_time = current_time
                warning_msg = f"อุ๊ย! ใกล้ไปแล้วค่ะ กรุณาออกห่างเพื่อความปลอดภัย"
                print(f"⚠️ PROXIMITY WARNING: {center_distance_mm} mm")
                if avatar_instance:
                    avatar_instance.set_emotion("SURPRISED")
                send_emotion_to_robot("PROXIMITY")
                add_chat_message('system', f'⚠️ เตือน! ระยะใกล้เกินไป: {center_distance_mm} mm')
                speak_async(warning_msg)
        
        # === อัปเดต Dashboard (Tkinter) ===
        # อัปเดตภาพกล้อง
        dashboard.update_rgb_frame(latest_frame, is_too_close)
        dashboard.update_depth_frame(latest_depth_frame, is_too_close)
        
        # อัปเดตระยะทาง
        dashboard.update_distance(center_distance_mm, is_too_close)
        
        # อัปเดต chat log
        dashboard.update_chat()
        
        # แสดง/ซ่อน alarm
        if is_too_close:
            dashboard.show_alarm(center_distance_mm)
        else:
            dashboard.hide_alarm()
        
        # Tkinter update
        dashboard.update()

        clock.tick(30)

    dashboard.destroy()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()

