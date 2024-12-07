from ultralytics import YOLO
from camera_manager import CameraManager
from vehicle_data_manager import VehicleDataManager
from traffic_score_calculator import TrafficScoreCalculator
from traffic_light_manager import TrafficLightManager
import cv2

if __name__ == "__main__":
    model = YOLO("C:/Users/Faruk/Desktop/proje/best.pt")

    # Araç verisi ve skor hesaplama yöneticileri
    vehicle_data_manager = VehicleDataManager()
    score_calculator = TrafficScoreCalculator()

    # Kamera yöneticisini oluştur
    camera_manager = CameraManager(model, vehicle_data_manager, score_calculator)

    # Kameraları başlat
    print("Kameralar başlatılıyor...")
    camera_manager.start_threads()

    # Trafik ışık yöneticisini oluştur
    traffic_light_manager = TrafficLightManager(score_calculator, vehicle_data_manager, camera_manager)

    # İlk skorları hesapla ve güncelle
    print("Trafik ışığı sıralaması başlatılıyor...")
    traffic_light_manager.update_scores()

    # Trafik ışıklarını işlemeye başla
    traffic_light_manager.process_traffic_lights()

    print("Program tamamlandı.")
