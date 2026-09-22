import cv2
import threading
import time
from flask import Flask, jsonify
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
CORS(app) # อนุญาตให้เว็บดึงข้อมูลได้

# กำหนดสถานะลานจอดรถ (False = ว่าง, True = มีรถ)
parking_status = {
    "A1": False, "A2": False, "A3": False, "A4": False, "A5": False
}

# กำหนดพิกัดกรอบช่องจอดรถ (x_min, y_min, x_max, y_max)
# **หมายเหตุ: ตัวเลขนี้คุณต้องไปกะระยะให้ตรงกับวิดีโอ test_video.mp4 ของคุณเอง**
parking_zones = {
    "A1": (50, 100, 250, 300),
    "A2": (300, 100, 500, 300),
    "A3": (550, 100, 750, 300),
    "A4": (800, 100, 1000, 300),
    "A5": (1050, 100, 1250, 300)
}

def check_intersection(car_box, zone_box):
    # ฟังก์ชันเช็คว่ารถทับกับช่องจอดหรือไม่
    x_left = max(car_box[0], zone_box[0])
    y_top = max(car_box[1], zone_box[1])
    x_right = min(car_box[2], zone_box[2])
    y_bottom = min(car_box[3], zone_box[3])
    
    if x_right < x_left or y_bottom < y_top:
        return False
    return True

def run_yolo():
    model = YOLO('yolov8n.pt') 
    
    # ใช้วิดีโอแทนกล้อง
    video_path = 'test_video.mp4'
    cap = cv2.VideoCapture(video_path) 
    
    while True:
        success, frame = cap.read()
        if not success:
            # ถ้าวิดีโอเล่นจบ ให้กลับไปเริ่มใหม่
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        results = model(frame, classes=[2, 3, 5, 7]) # 2=car, 3=motorcycle, 5=bus, 7=truck
        
        # รีเซ็ตสถานะเป็นว่างก่อน
        for key in parking_status:
            parking_status[key] = False
            
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                car_box = (x1, y1, x2, y2)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                for slot_id, zone_box in parking_zones.items():
                    if check_intersection(car_box, zone_box):
                        parking_status[slot_id] = True # เปลี่ยนสถานะเป็นมีรถ
                        
        # วาดกรอบช่องจอด
        for slot_id, zone_box in parking_zones.items():
            color = (0, 0, 255) if parking_status[slot_id] else (0, 255, 0)
            cv2.rectangle(frame, (zone_box[0], zone_box[1]), (zone_box[2], zone_box[3]), color, 2)
            cv2.putText(frame, slot_id, (zone_box[0], zone_box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Detection System (Video Mode)", frame)
        
        # หน่วงเวลาเล็กน้อยเพื่อให้วิดีโอเล่นในความเร็วปกติ
        time.sleep(0.03)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

@app.route('/api/status')
def get_status():
    return jsonify(parking_status)

if __name__ == '__main__':
    # รัน YOLO แยกจาก API
    t = threading.Thread(target=run_yolo)
    t.daemon = True
    t.start()
    
    print("🚀 เซิร์ฟเวอร์ API พร้อมแล้วที่ http://localhost:5000/api/status")
    app.run(host='0.0.0.0', port=5000)
