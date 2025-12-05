import cv2
import torch
import numpy as np
import pygame
import time
import threading
import smtplib
import os
from email.message import EmailMessage
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Initialize pygame for alarm sound
pygame.init()
pygame.mixer.music.load("Alarm/alarm.wav")

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Email Configuration
SENDER_EMAIL = "your_email@gmail.com"
EMAIL_PASSWORD = "your_app_password"
RECEIVER_EMAIL = "receiver_email@gmail.com"

# Global variables
cap = None
video_path = None
is_running = False
pts = []
count = 0
max_photos = 3
last_detection_time = 0
alarm_playing = False

# Function to send an email alert
def send_email_alert(image_path):
    try:
        msg = EmailMessage()
        msg["Subject"] = "🚨 Intrusion Detected! 🚨"
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL
        msg.set_content("Suspicious activity detected! See the attached image.")

        with open(image_path, "rb") as f:
            img_data = f.read()
            msg.add_attachment(img_data, maintype="image", subtype="jpeg", filename="intrusion.jpg")

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, EMAIL_PASSWORD)
            server.send_message(msg)

        print("✅ Email alert sent successfully!")

    except Exception as e:
        print(f"❌ Failed to send email: {e}")

# Function to select video file
def select_video():
    global video_path
    video_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
    if video_path:
        lbl_video_path.config(text=f"Selected: {video_path}")

# Mouse callback function to select ROI
def draw_polygon(event, x, y, flags, param):
    global pts
    if event == cv2.EVENT_LBUTTONDOWN:  # Left-click to select points
        pts.append((x, y))
        print(f"Point selected: {x}, {y}")
    elif event == cv2.EVENT_RBUTTONDOWN:  # Right-click to reset selection
        pts = []
        print("ROI selection reset.")

# Function to let user select ROI
def select_roi(frame):
    global pts
    pts = []
    cv2.namedWindow("ROI Selection")
    cv2.setMouseCallback("ROI Selection", draw_polygon)

    while True:
        temp_frame = frame.copy()
        for point in pts:
            cv2.circle(temp_frame, point, 5, (0, 0, 255), -1)

        cv2.imshow("ROI Selection", temp_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 13:  # Press ENTER to confirm selection
            if len(pts) < 4:
                print("Select at least 4 points.")
            else:
                cv2.destroyWindow("ROI Selection")
                return pts

        if key == 27:  # Press ESC to cancel selection
            pts = []
            cv2.destroyWindow("ROI Selection")
            return None

# Function to start processing video
def start_detection():
    global cap, is_running, video_path
    if video_path is None:
        lbl_status.config(text="No video selected!", fg="red")
        return

    cap = cv2.VideoCapture(video_path)
    is_running = True

    # Let user select ROI
    ret, frame = cap.read()
    if not ret:
        print("Failed to load video.")
        return

    roi_pts = select_roi(frame)  # Select ROI before processing

    if roi_pts is None or len(roi_pts) < 4:
        print("No ROI selected. Exiting...")
        cap.release()
        cv2.destroyAllWindows()
        return

    thread = threading.Thread(target=process_video, args=(roi_pts,))
    thread.start()

# Function to stop detection
def stop_detection():
    global is_running
    is_running = False

# Function to check if a point is inside a polygon
def inside_polygon(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, np.int32), (point[0], point[1]), False) >= 0

# Function to process video
def process_video(roi_pts):
    global cap, is_running, count, last_detection_time, alarm_playing

    while cap.isOpened() and is_running:
        ret, frame = cap.read()
        if not ret:
            break

        frame_detected = frame.copy()
        height, width = frame.shape[:2]
        frame = cv2.resize(frame, (640, int(640 * (height / width))))

        results = model(frame)
        person_detected = False

        for index, row in results.pandas().xyxy[0].iterrows():
            if row['name'] != 'person':
                continue

            x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
            cv2.putText(frame, "Person", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

            overlay = frame.copy()
            cv2.fillPoly(overlay, [np.array(roi_pts)], (0, 255, 0))
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

            if inside_polygon((center_x, center_y), roi_pts):
                person_detected = True

                if count < max_photos and (time.time() - last_detection_time) > 2:
                    img_path = f"Detected_Photos/detected_{count}.jpg"
                    cv2.imwrite(img_path, frame_detected)
                    count += 1
                    last_detection_time = time.time()
                    send_email_alert(img_path)

                if not alarm_playing:
                    pygame.mixer.music.play()
                    alarm_playing = True

                cv2.putText(frame, "INTRUSION DETECTED!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        if not person_detected and alarm_playing:
            pygame.mixer.music.stop()
            alarm_playing = False

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = ImageTk.PhotoImage(img)

        lbl_video.config(image=img)
        lbl_video.image = img

    cap.release()
    cv2.destroyAllWindows()

# GUI Setup
root = tk.Tk()
root.title("Burglary Detection System")
root.geometry("800x600")
root.configure(bg='#2c3e50')

# Custom style configuration
style = {
    'font': ('Helvetica', 12),
    'bg': '#2c3e50',
    'fg': 'white',
    'activebg': '#34495e',
    'warning': '#e74c3c',
    'success': '#2ecc71',
    'button_bg': '#3498db',
    'padding': 10
}

# Frame for controls
control_frame = tk.Frame(root, bg=style['bg'], padx=20, pady=20)
control_frame.pack(pady=20)

btn_select_video = tk.Button(
    control_frame,
    text="📤 Upload Video",
    command=select_video,
    bg=style['button_bg'],
    fg=style['fg'],
    activebackground=style['activebg'],
    font=style['font'],
    borderwidth=3,
    relief='ridge'
)
btn_select_video.grid(row=0, column=0, padx=style['padding'], pady=style['padding'])

btn_start = tk.Button(
    control_frame,
    text="▶ Start Detection",
    command=start_detection,
    bg='#27ae60',
    fg=style['fg'],
    activebackground='#219a52',
    font=style['font'],
    borderwidth=3,
    relief='ridge'
)
btn_start.grid(row=0, column=1, padx=style['padding'], pady=style['padding'])

btn_stop = tk.Button(
    control_frame,
    text="⏹ Stop Detection",
    command=stop_detection,
    bg=style['warning'],
    fg=style['fg'],
    activebackground='#c0392b',
    font=style['font'],
    borderwidth=3,
    relief='ridge'
)
btn_stop.grid(row=0, column=2, padx=style['padding'], pady=style['padding'])

lbl_video_path = tk.Label(
    root,
    text="No video selected",
    fg=style['success'],
    bg=style['bg'],
    font=('Helvetica', 10, 'italic')
)
lbl_video_path.pack(pady=10)

warning_label = tk.Label(
    root,
    text="⚠ Warning: Do not close window while processing!",
    fg=style['warning'],
    bg=style['bg'],
    font=('Helvetica', 12, 'bold')
)
warning_label = tk.Label(
    root,
    text="⚠ Warning: After selecting ROI press Enter and wait for video to play",
    fg=style['warning'],
    bg=style['bg'],
    font=('Helvetica', 12, 'bold')
)
warning_label.pack(pady=5)

lbl_video = tk.Label(root, bg=style['bg'])
lbl_video.pack(pady=20)

root.mainloop()
