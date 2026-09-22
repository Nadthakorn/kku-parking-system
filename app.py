import cv2
import threading
import time
from flask import Flask, jsonify
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
# อนุญาตให้เว็บดึงข้อมูลข้ามโดเมนได้
CORS(app) 

# ==========================================
# ⚙️ ตั้งค่าแหล่งภาพ (ตั้งค่าก่อนพรีเซนต์ตรงนี้)
# ==========================================
USE_CAMERA = False   # 🔴 วันซ้อมใช้ False (เล่นวิดีโอ), วันพรีเซนต์จริงเปลี่ยนเป็น True (ใช้กล้อง)
CAMERA_INDEX = 1     # 0 = กล้องโน้ตบุ๊ก, 1 หรือ 2 = กล้อง USB ที่นำมาต่อเพิ่ม
VIDEO_PATH = 'test_video.mp4'

# โครงสร้างข้อมูลให้ตรงกับที่หน้าเว็บ index_2.html รองรับ (มีอาคาร A ถึง G)
# True = ว่าง (Available - สีเขียว), False = มีรถจอด (Occupied - สีแดง)
buildings_data = [
    {"id": "A", "slots": [True, True, True, True, True, True, True, True, True, True]},
    {"id": "B", "slots": [True, True, True, True, True, True, True, True, True, True]},
    {"id": "C", "slots": [True, True, True, True, True, True, True, True, True, True]},
    {"id": "D", "slots": [True, True, True, True, True, True, True, True, True, True]},
    {"id": "E", "slots": [True, True, True, True, True, True, True, True, True, True]},
    {"id": "F", "slots": [True, True, True, True, True, True, True, True, True, True]},
    {"id": "G", "slots": [True, True, True, True, True, True, True, True, True, True]}
]

# กำหนดพิกัดกรอบช่องจอดสำหรับตึก A (5 ช่องแรก A1-A5)
parking_zones = {
    0: (50, 100, 250, 300),   # A-01 (index 0)
    1: (300, 100, 500, 300),  # A-02 (index 1)
    2: (550, 100, 750, 300),  # A-03 (index 2)
    3: (800, 100, 1000, 300), # A-04 (index 3)
    4: (1050, 100, 1250, 300) # A-05 (index 4)
}

def check_intersection(car_box, zone_box):
    # เช็คว่ากรอบของรถ ทับซ้อนกับกรอบของช่องจอดหรือไม่
    x_left = max(car_box[0], zone_box[0])
    y_top = max(car_box[1], zone_box[1])
    x_right = min(car_box[2], zone_box[2])
    y_bottom = min(car_box[3], zone_box[3])
    
    if x_right < x_left or y_bottom < y_top:
        return False
    return True

def run_yolo():
    model = YOLO('yolov8n.pt')  
    
    # เลือกว่าจะใช้กล้องจริงหรือวิดีโอ
    if USE_CAMERA:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        print(f"🎥 กำลังเปิดกล้อง (Index {CAMERA_INDEX})...")
    else:
        cap = cv2.VideoCapture(VIDEO_PATH)
        print(f"🎬 กำลังเล่นวิดีโอจำลอง...")
    
    while True:
        success, frame = cap.read()
        if not success:
            if not USE_CAMERA:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # วนลูปวิดีโอ
                continue
            else:
                print("❌ สัญญาณกล้องหลุด! กรุณาตรวจสอบสาย USB")
                break
                
        results = model(frame, classes=[2, 3, 5, 7]) # ตรวจเฉพาะยานพาหนะ
        
        # ดึงข้อมูลของอาคาร A มาอัปเดต
        building_a = next(b for b in buildings_data if b["id"] == "A")
        
        # เก็บสถานะชั่วคราวสำหรับ 5 ช่องแรก (False = ว่าง, True = มีรถ)
        slot_occupied = [False, False, False, False, False]
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                car_box = (x1, y1, x2, y2)
                
                # วาดกรอบสีฟ้าครอบรถที่เจอ
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # เช็ครถเข้าซองจอดตึก A
                for idx, zone_box in parking_zones.items():
                    if check_intersection(car_box, zone_box):
                        slot_occupied[idx] = True # มีรถจอด
                        
        # แปลงสถานะให้ตรงกับเว็บ (เว็บใช้ True = ว่าง, False = มีรถจอด)
        for i in range(5):
            building_a["slots"][i] = not slot_occupied[i]

        # วาดกรอบช่องจอด (แดง = มีรถ, เขียว = ว่าง)
        for idx, zone_box in parking_zones.items():
            is_occupied = slot_occupied[idx]
            color = (0, 0, 255) if is_occupied else (0, 255, 0)
            cv2.rectangle(frame, (zone_box[0], zone_box[1]), (zone_box[2], zone_box[3]), color, 2)
            cv2.putText(frame, f"A-{idx+1:02d}", (zone_box[0], zone_box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("KKU Parking Camera", frame)
        
        time.sleep(0.03) 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/api/status', methods=['GET'])
def get_status():
    # ส่งข้อมูลออกไปในรูปแบบที่ JavaScript ของเว็บ index_2.html รอรับอยู่
    return jsonify({"buildings": buildings_data})

if __name__ == '__main__':
    t = threading.Thread(target=run_yolo)
    t.daemon = True
    t.start()
    
    print("🚀 ระบบ API รันแล้วที่ http://localhost:3000/api/status")
    # เปลี่ยนมาใช้พอร์ต 3000 ให้ตรงกับคำสั่ง ngrok http 3000
    app.run(host='0.0.0.0', port=3000)
