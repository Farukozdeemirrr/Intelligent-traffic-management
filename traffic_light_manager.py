import time
import cv2

class TrafficLightManager:
    def __init__(self, score_calculator, vehicle_data_manager, camera_manager):
        self.score_calculator = score_calculator
        self.vehicle_data_manager = vehicle_data_manager
        self.camera_manager = camera_manager
        self.sorted_scores = []  # Skorların sıralı tutulacağı liste
        self.green_light_queue = []  # Yeşil ışık için sıraya alınacak liste

    def update_scores(self):
        """Skorları günceller ve sıralar."""
        scores = self.score_calculator.calculate_scores(self.vehicle_data_manager)
        # Skorları büyükten küçüğe sırala
        self.sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def process_traffic_lights(self):
        """Trafik ışıklarını sıraya göre kontrol eder."""
        while True:
            # Eğer sıralanmış skorlar boşsa yeniden doldur
            if not self.sorted_scores:
                self.update_scores()

            if not self.sorted_scores:
                print("Tüm kameralar işleme alındı. Bekleniyor...")
                time.sleep(2)  # Tüm kameralar işlenmişse bir süre bekle
                continue

            # En yüksek skoru ve ilgili kamerayı al
            camera_id, score = self.sorted_scores.pop(0)  # İlk elemanı çıkar
            self.green_light_queue.append((camera_id, score))  # Yeşil ışık için sıraya al

            # Yeşil ışık süresini skora göre hesapla
            green_light_duration = int(score * 10)  # Örnek olarak skoru 10 ile çarpıyoruz

            print(f"Kamera {camera_id}: Yeşil ışık YANIYOR ({green_light_duration} saniye)")
            start_time = time.time()
            while time.time() - start_time < green_light_duration:
                frame = self.camera_manager.get_frame(camera_id)
                if frame is not None:
                    cv2.putText(frame, "Yeşil Işık", (frame.shape[1] - 150, frame.shape[0] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow(f"Kamera {camera_id}", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return  # Çıkış yap

            # Sarı ışık
            print(f"Kamera {camera_id}: Sarı ışık YANIYOR (5 saniye)")
            start_time = time.time()
            while time.time() - start_time < 5:
                frame = self.camera_manager.get_frame(camera_id)
                if frame is not None:
                    cv2.putText(frame, "Sarı Işık", (frame.shape[1] - 150, frame.shape[0] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow(f"Kamera {camera_id}", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return  # Çıkış yap

            # Kırmızı ışık
            print(f"Kamera {camera_id}: Kırmızı ışık YANIYOR")
            frame = self.camera_manager.get_frame(camera_id)
            if frame is not None:
                cv2.putText(frame, "Kırmızı Işık", (frame.shape[1] - 150, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow(f"Kamera {camera_id}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return  # Çıkış yap

            # Yeşil ışık kuyruğu dolduysa yeniden sıralama yap
            if len(self.green_light_queue) == len(self.camera_manager.get_active_camera_ids()):
                print("Tüm kameralar için trafik ışığı tamamlandı. Sıralama yeniden yapılıyor...")
                self.green_light_queue.clear()  # Kuyruğu temizle
                self.update_scores()
