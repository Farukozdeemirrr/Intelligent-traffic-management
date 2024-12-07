import cv2
import math
import time
from roi_manager import ROIManager

class Camera:
    def __init__(self, camera_id, model, window_name, vehicle_data_manager, score_calculator, width=640, height=480, confidence_threshold=0.5, max_distance=50):
        self.camera_id = camera_id
        self.model = model
        self.window_name = window_name
        self.vehicle_data_manager = vehicle_data_manager
        self.score_calculator = score_calculator
        self.width = width
        self.height = height
        self.confidence_threshold = confidence_threshold
        self.cap = cv2.VideoCapture(self.camera_id)
        self.max_distance = max_distance

        # Benzersiz nesne izleme için ID eşleme
        self.tracked_objects = {}  # {track_id: (center_x, center_y, last_seen_time, inside_roi, counted)}
        self.object_id_counter = 0

        if not self.cap.isOpened():
            print(f"{self.window_name}: Kamera açılamadı!")
            exit()

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # ROIManager nesnesini oluştur
        self.roi_manager = ROIManager(self.width, self.height)

    def calculate_distance(self, obj1, obj2):
        """İki nokta arasındaki mesafeyi hesaplar."""
        x1, y1 = obj1
        x2, y2 = obj2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def update_tracked_objects(self, obj_center, class_id):
        """Mevcut izleme listesini günceller."""
        object_id = None

        # Mevcut nesnelerle eşleştir
        for track_id, (tracked_center, last_seen, inside_roi, counted) in self.tracked_objects.items():
            if self.calculate_distance(tracked_center, obj_center) < self.max_distance:
                object_id = track_id
                break

        if object_id is None:  # Yeni nesne
            self.object_id_counter += 1
            object_id = self.object_id_counter
            self.tracked_objects[object_id] = (obj_center, time.time(), True, False)

        # Sadece ROI içindeki ve henüz sayılmamış araçları say
        if not self.tracked_objects[object_id][3]:  # Counted = False
            self.vehicle_data_manager.add_vehicle(self.camera_id, str(class_id))
            self.tracked_objects[object_id] = (obj_center, time.time(), True, True)  # Counted = True

        else:
            self.tracked_objects[object_id] = (obj_center, time.time(), True, True)

        return object_id

    def process_frame(self):
        """Kamera çerçevesini işleyip araçları sayar."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print(f"{self.window_name}: Frame alınamadı!")
                break

            # YOLO modelini çalıştır
            results = self.model(frame)

            # Yeni izleme verilerini tutmak için geçici sözlük
            current_tracked_objects = {}

            for result in results[0].boxes.data:
                x1, y1, x2, y2 = map(int, result[:4])
                class_id = int(result[5])
                confidence = result[4]

                if confidence > self.confidence_threshold:
                    obj_center = ((x1 + x2) // 2, (y1 + y2) // 2)  # Nesnenin merkezi

                    # ROI içindeki yeni nesneleri kontrol et
                    if self.roi_manager.is_inside_roi(obj_center[0], obj_center[1]):
                        object_id = self.update_tracked_objects(obj_center, class_id)

                        # Görselleştirme
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"ID: {object_id}, Class: {class_id}"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # İzleme listesini güncelle
            self.tracked_objects = {
                track_id: data for track_id, data in self.tracked_objects.items()
                if time.time() - data[1] < 5  # 2 saniye boyunca aktif olan nesneleri tut
            }

            # ROI'yi çiz
            self.roi_manager.draw_roi(frame)

            # Araç sayımını göster
            y_offset = 30
            for class_name, count in self.vehicle_data_manager.get_vehicle_data(self.camera_id).items():
                cv2.putText(frame, f"{class_name}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                y_offset += 30

            # Skoru göster
            scores = self.score_calculator.calculate_scores(self.vehicle_data_manager)
            current_score = scores.get(self.camera_id, 0)
            cv2.putText(frame, f"Skor: {current_score:.2f}", (self.width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Çerçeveyi göster
            cv2.imshow(self.window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
