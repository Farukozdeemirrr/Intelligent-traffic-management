from camera import Camera
import threading
import cv2
class CameraManager:
    def __init__(self, model, vehicle_data_manager, score_calculator):
        """
        Kamera yönetimini sağlayan sınıf.

        Args:
            model: YOLO model nesnesi.
            vehicle_data_manager: VehicleDataManager nesnesi.
            score_calculator: TrafficScoreCalculator nesnesi.
        """
        self.model = model
        self.vehicle_data_manager = vehicle_data_manager
        self.score_calculator = score_calculator
        self.cameras = []  # Kameraları tutmak için bir liste

        # Mevcut kameraları bul ve ekle
        self.get_available_cameras()

    def get_available_cameras(self):
        """
        Mevcut kameraları bulur ve listeye ekler.
        """
        for i in range(10):  # İlk 10 kamera indeksini kontrol et
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Kamera açılabiliyorsa listeye ekle
                self.cameras.append(Camera(i, self.model, f"Kamera {i}", self.vehicle_data_manager, self.score_calculator))
                cap.release()

    def start_threads(self):
        """
        Kameralar için iş parçacıklarını başlatır.
        """
        threads = []
        for camera in self.cameras:
            thread = threading.Thread(target=camera.process_frame)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()
    def get_frame(self, camera_id):
        """Belirtilen kameradan çerçeve alır."""
        if camera_id in self.cameras:
            ret, frame = self.cameras[camera_id].read()
            if ret:
                return frame
        return None