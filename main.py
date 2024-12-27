import cv2
import math
import time
import threading
import queue
from ultralytics import YOLO
import torch
import logging
import requests  # Nodemcu'ya HTTP isteği atmak için

#########################################################
# ROIManager: ROI (ilgi alanı) bölgesini yönetir
#########################################################
#nodemcu_ip = "192.168.1.114"  # Nodemcu'nun IP adresini buraya yazın
#nodemcu_url = f"http://{nodemcu_ip}/update"
class ROIManager:
    def __init__(self, frame_width, frame_height):
        self.roi_x1 = 0
        self.roi_y1 = 0
        self.roi_x2 = frame_width // 2
        self.roi_y2 = frame_height

    def is_inside_roi(self, x, y):
        return self.roi_x1 <= x <= self.roi_x2 and self.roi_y1 <= y <= self.roi_y2

    def draw_roi(self, frame):
        cv2.rectangle(frame, (self.roi_x1, self.roi_y1), (self.roi_x2, self.roi_y2), (0, 255, 255), 2)
        cv2.putText(frame, "ROI", (self.roi_x1 + 10, self.roi_y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


#########################################################
# VehicleDataManager: Araç sayımlarını tutar
#########################################################
class VehicleDataManager:
    def __init__(self):
        self.data = {}
        self.lock = threading.Lock()

    def add_vehicle(self, camera_id, class_name):
        with self.lock:
            if camera_id not in self.data:
                self.data[camera_id] = {}
            if class_name not in self.data[camera_id]:
                self.data[camera_id][class_name] = 0
            self.data[camera_id][class_name] += 1

    def get_vehicle_data(self, camera_id):
        with self.lock:
            return self.data.get(camera_id, {})

    def reset_camera_data(self, camera_id):
        with self.lock:
            self.data[camera_id] = {}

    def update_with_remaining_objects(self, camera_id, detections, roi_manager):
        with self.lock:
            for detection in detections:
                x1, y1, x2, y2, class_id, _ = detection
                obj_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                if roi_manager.is_inside_roi(obj_center[0], obj_center[1]):
                    if camera_id not in self.data:
                        self.data[camera_id] = {}
                    if str(class_id) not in self.data[camera_id]:
                        self.data[camera_id][str(class_id)] = 0
                    self.data[camera_id][str(class_id)] += 1


#########################################################
# TrafficScoreCalculator: Araç türlerine göre skor hesaplar
#########################################################
class TrafficScoreCalculator:
    def calculate_scores(self, vehicle_data_manager):
        weights = {"0": 2.0, "1": 1.0, "2": 2.5, "3": 0.5}
        scores = {}
        with vehicle_data_manager.lock:
            for camera_id, classes in vehicle_data_manager.data.items():
                score = 0
                for cname, count in classes.items():
                    w = weights.get(cname, 0)
                    score += count * w
                scores[camera_id] = score
        return scores


#########################################################
# FrameGrabber: Kameradan frame alır, kuyruğa koyar
#########################################################
class FrameGrabber(threading.Thread):
    def __init__(self, cap, frame_queue):
        super().__init__()
        self.cap = cap
        self.frame_queue = frame_queue
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)

    def stop(self):
        self.running = False


#########################################################
# YOLOProcessor: ROI üzerindeki frame'i YOLO ile işler
#########################################################
class YOLOProcessor(threading.Thread):
    def __init__(self, model, frame_queue, result_queue, roi_manager, confidence_threshold=0.5):
        super().__init__()
        self.model = model
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.roi_manager = roi_manager
        self.confidence_threshold = confidence_threshold
        self.running = True

    def run(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.01)
            except queue.Empty:
                continue

            roi_frame = frame[self.roi_manager.roi_y1:self.roi_manager.roi_y2,
                              self.roi_manager.roi_x1:self.roi_manager.roi_x2]

            with torch.no_grad():
                results = self.model(roi_frame)

            detections = []
            for result in results[0].boxes.data:
                x1, y1, x2, y2 = map(int, result[:4])
                conf = float(result[4])
                class_id = int(result[5])

                if conf > self.confidence_threshold:
                    x1 += self.roi_manager.roi_x1
                    y1 += self.roi_manager.roi_y1
                    x2 += self.roi_manager.roi_x1
                    y2 += self.roi_manager.roi_y1
                    detections.append((x1, y1, x2, y2, class_id, conf))

            if self.result_queue.full():
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass
            self.result_queue.put((frame, detections))

    def stop(self):
        self.running = False


#########################################################
# Camera: Her kamera için frame işleme döngüsü
#########################################################
class Camera:
    def __init__(self, camera_id, model, window_name, vehicle_data_manager, score_calculator,
                 width=640, height=480, max_distance=50):

        self.camera_id = camera_id
        self.model = model
        self.window_name = window_name
        self.vehicle_data_manager = vehicle_data_manager
        self.score_calculator = score_calculator
        self.width = width
        self.height = height
        self.max_distance = max_distance
        self.tespiti_durdur = False

        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"{self.window_name}: Kamera açılamadı!")
            exit()

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.roi_manager = ROIManager(self.width, self.height)

        self.tracked_objects = {}
        self.object_id_counter = 0

        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)

        self.frame_grabber = FrameGrabber(self.cap, self.frame_queue)
        self.yolo_processor = YOLOProcessor(self.model, self.frame_queue, self.result_queue, self.roi_manager)

    def calculate_distance(self, obj1, obj2):
        x1, y1 = obj1
        x2, y2 = obj2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def update_tracked_objects(self, obj_center, class_id):
        object_id = None
        for track_id, (tracked_center, last_seen, inside_roi, counted) in self.tracked_objects.items():
            if self.calculate_distance(tracked_center, obj_center) < self.max_distance:
                object_id = track_id
                break

        if object_id is None:
            self.object_id_counter += 1
            object_id = self.object_id_counter
            inside = self.roi_manager.is_inside_roi(obj_center[0], obj_center[1])
            self.tracked_objects[object_id] = (obj_center, time.time(), inside, False)
        else:
            old_data = self.tracked_objects[object_id]
            inside = self.roi_manager.is_inside_roi(obj_center[0], obj_center[1])
            self.tracked_objects[object_id] = (obj_center, time.time(), inside, old_data[3])

        current_data = self.tracked_objects[object_id]
        inside_roi = current_data[2]
        counted = current_data[3]

        if inside_roi and not counted:
            self.vehicle_data_manager.add_vehicle(self.camera_id, str(class_id))
            self.tracked_objects[object_id] = (current_data[0], current_data[1], current_data[2], True)

        return object_id

    def pause_detection(self):
        """Nesne tespitini durdur."""
        self.yolo_processor.stop()

    def resume_detection(self):
        """Nesne tespitini yeniden başlat."""
        self.yolo_processor = YOLOProcessor(self.model, self.frame_queue, self.result_queue, self.roi_manager)
        self.yolo_processor.start()

    def stop(self):
        self.frame_grabber.stop()
        self.yolo_processor.stop()
        self.frame_grabber.join()
        self.yolo_processor.join()

        if self.cap.isOpened():
            self.cap.release()

        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass

        print(f"Kamera {self.camera_id} durduruldu.")

    def process_frame(self):
        self.frame_grabber.start()
        self.yolo_processor.start()

        last_print_time = time.time()

        while True:
            try:
                frame, detections = self.result_queue.get(timeout=0.01)
            except queue.Empty:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # ROI çizimi ve skor yazısını ekle
            self.roi_manager.draw_roi(frame)

            current_time = time.time()
            if current_time - last_print_time >= 1.0:
                scores = self.score_calculator.calculate_scores(self.vehicle_data_manager)
                current_score = scores.get(self.camera_id, 0)
                last_print_time = current_time

            # Skoru her karede yaz
            scores = self.score_calculator.calculate_scores(self.vehicle_data_manager)
            current_score = scores.get(self.camera_id, 0)
            cv2.putText(frame, f"Skor: {current_score:.2f}", (self.width - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Tespit durdurulmuşsa yalnızca ROI ve skor göster
            if self.tespiti_durdur:
                cv2.imshow(self.window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Tespit ve nesne işleme
            for (x1, y1, x2, y2, class_id, conf) in detections:
                obj_center = ((x1 + x2)//2, (y1 + y2)//2)
                if self.roi_manager.is_inside_roi(obj_center[0], obj_center[1]):
                    object_id = self.update_tracked_objects(obj_center, class_id)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"ID: {object_id}, Class: {class_id}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, (255, 0, 0), 2)

            cv2.imshow(self.window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

#########################################################
# CameraManager: Birden fazla kamerayı yönetir
#########################################################
class CameraManager:
    def __init__(self, model, vehicle_data_manager, score_calculator, camera_ids=[0,1], width=640, height=480, max_distance=50):
        self.model = model
        self.vehicle_data_manager = vehicle_data_manager
        self.score_calculator = score_calculator
        self.cameras = []
        for cam_id in camera_ids:
            cam = Camera(cam_id, self.model, f"Kamera {cam_id}", self.vehicle_data_manager, self.score_calculator, width, height, max_distance=max_distance)
            self.cameras.append(cam)

    def start_cameras(self):
        self.threads = []
        for cam in self.cameras:
            t = threading.Thread(target=cam.process_frame)
            t.start()
            self.threads.append(t)

    def stop_all_cameras(self):
        for cam in self.cameras:
            cam.stop()

        for t in self.threads:
            t.join()

        cv2.destroyAllWindows()




#########################################################
# Dinamik Trafik Işığı Kontrolü (Nodemcu Entegre ve Tek Yön Yeşil, Diğerleri Kırmızı)
#########################################################

# Kamera ve trafik kontrol fonksiyonu
def run_intersection_control(camera_ids, vehicle_data_manager, score_calculator):
    
    remaining_vehicles = {cid: 0 for cid in camera_ids}  # Her kamera için kalan araç sayısını takip eder

    last_direction = None  # Son seçilen yönü takip etmek için

    # İlk yoğunluğu hesapla ve en yoğun yönü seç
    scores = score_calculator.calculate_scores(vehicle_data_manager)
    direction_scores = {cid: scores.get(cid, 0) for cid in camera_ids}
    sorted_dirs = sorted(direction_scores.items(), key=lambda x: x[1], reverse=True)

    yellow_time = 3  # Sarı ışık süresi (sabit olarak 3 saniye)

    while True:
        scores = score_calculator.calculate_scores(vehicle_data_manager)

        # Kameralardan gelen araç sayısını güncelle
        for cid in camera_ids:
            vehicle_data = vehicle_data_manager.get_vehicle_data(cid)
            detected_vehicles = vehicle_data.get("vehicle_count", 0)
            # Eğer geçiş sonrası araçlar kalmamışsa kalan araç sayısını sıfırla
            if detected_vehicles == 0:
                remaining_vehicles[cid] = 0
            else:
                remaining_vehicles[cid] += detected_vehicles

        direction_scores = {}
        for cid in camera_ids:
            # Kalan araçları mevcut skorlarla birleştir
            density = scores.get(cid, 0) + remaining_vehicles[cid]
            total_score = density
            direction_scores[cid] = total_score

        # Skorları sıralama ve seçilecek yönü belirleme
        sorted_dirs = sorted(direction_scores.items(), key=lambda x: x[1], reverse=True)
        chosen_direction, chosen_score = sorted_dirs[0]

        # Eğer aynı yön üst üste seçilmişse bir sonraki yön zorlanır
        if chosen_direction == last_direction and len(sorted_dirs) > 1:
            chosen_direction, chosen_score = sorted_dirs[1]

        # Skor sıfır olsa bile minimum 5 saniye yeşil ışık süresi
        green_time = int(max(min(max(chosen_score, 5), 30), 5))
        # Yeşil ışık süresi boyunca tespiti durdur
        for camera in manager.cameras:
            if camera.camera_id == chosen_direction:
                camera.tespiti_durdur = True

        # Yeşil ışık süresi
        print(f"webcam-{chosen_direction} yeşil ışık {green_time}sn")
        vehicles_before = vehicle_data_manager.get_vehicle_data(chosen_direction).get("vehicle_count", 0)

        # ESP8266'ya veri gönder: sadece yeşil ışık süresi ve yön bilgisi
        send_to_nodemcu(chosen_direction, green_time)

        for s in range(green_time, 0, -1):
            print(f"webcam-{chosen_direction} yeşil ışık {s}sn")
            time.sleep(1)

        # Sarı ışık simülasyonu (Python kontrol eder)
        print(f"webcam-{chosen_direction} sarı ışık {yellow_time}sn")
        for s in range(yellow_time, 0, -1):
            print(f"webcam-{chosen_direction} sarı ışık {s}sn")
            time.sleep(1)

        # Kırmızı ışık sırasında skor sıfırla ve tespiti yeniden başlat
        print(f"webcam-{chosen_direction} kırmızı ışık")
        vehicle_data_manager.reset_camera_data(chosen_direction)

        for camera in manager.cameras:
            if camera.camera_id == chosen_direction:
                camera.tespiti_durdur = False


        # Geçen araçları hesapla
        vehicles_after = vehicle_data_manager.get_vehicle_data(chosen_direction).get("vehicle_count", 0)
        vehicles_cleared = max(vehicles_before - vehicles_after, 0)
        remaining_vehicles[chosen_direction] = max(0, remaining_vehicles[chosen_direction] - vehicles_cleared)

        # Yeni yeşil ışık yönünü belirle
        last_direction = chosen_direction  # Son seçilen yönü kaydet
        next_direction = None
        for dir_index, (cid, _) in enumerate(sorted_dirs):
            if cid != chosen_direction:
                next_direction = cid
                break
        if next_direction is None:  # Tüm yönler sıfır skora sahipse sırayı döndür
            next_direction = chosen_direction

def send_to_nodemcu(direction, green_time):
    """
    Nodemcu'ya sadece yeşil ışık süresi ve yön bilgisini gönderir.
    :param direction: Aktif yön bilgisi
    :param green_time: Yeşil ışık süresi
    """
    nodemcu_url = "http://192.168.1.114/update"  # Nodemcu'nun IP adresini buraya yazın

    payload = f"{direction},{green_time}"

    headers = {
        "Content-Type": "text/plain"
    }

    try:
        requests.post(nodemcu_url, data=payload, headers=headers, timeout=5)
        print(f"Nodemcu'ya veri gönderildi: Yön = {direction}, Süre = {green_time}sn")
    except requests.exceptions.ConnectTimeout:
        print("Nodemcu'ya bağlantı zaman aşımı.")
    except requests.exceptions.ConnectionError:
        print("Nodemcu'ya bağlantı kurulamadı. IP adresini ve ağı kontrol edin.")
    except requests.exceptions.RequestException as e:
        print(f"Nodemcu'ya bağlantı hatası: {e}")

#########################################################
# Ana Kod
#########################################################
if __name__ == "__main__":
    logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

    model_path = "C:/Users/Faruk/Desktop/ProjeV1/projeV1/best.pt"
    model = YOLO(model_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    cuda_available = torch.cuda.is_available()
    print("CUDA mevcut mu?", cuda_available, flush=True)
    if cuda_available:
        print("CUDA Versiyonu:", torch.version.cuda, flush=True)
        print("Kullanılan GPU:", torch.cuda.get_device_name(torch.cuda.current_device()), flush=True)
    else:
        print("CUDA desteklenmiyor, CPU üzerinde çalışılıyor.", flush=True)

    vehicle_data_manager = VehicleDataManager()
    score_calculator = TrafficScoreCalculator()

    camera_ids = [0,1]  # Burada örnek olarak 2 kamera kullandık.
    manager = CameraManager(model, vehicle_data_manager, score_calculator, camera_ids=camera_ids)
    manager.start_cameras()

    try:
        run_intersection_control(camera_ids, vehicle_data_manager, score_calculator)
    except KeyboardInterrupt:
        manager.stop_all_cameras()
        print("Program sonlandırıldı.", flush=True)
