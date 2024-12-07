class TrafficScoreCalculator:
    def __init__(self, weights=None):
        # Varsayılan ağırlık değerlerini ayarla
        self.weights = weights or {"0": 0.15, "1": 0.1, "2": 0.2, "3": 0.05}

    def calculate_scores(self, vehicle_data_manager):
        """
        Tüm kameralar için araç türlerini ağırlıklarla çarparak toplam trafik skorlarını hesaplar.

        Args:
            vehicle_data_manager (VehicleDataManager): Araç verilerini yöneten sınıf.

        Returns:
            dict: Her kamera için hesaplanan skorlar. Örneğin: {camera_id: score}
        """
        all_vehicle_data = vehicle_data_manager.get_all_vehicle_data()
        scores = {}

        for camera_id, vehicle_data in all_vehicle_data.items():
            # Her kameranın skorunu hesapla
            score = sum(vehicle_data.get(str(class_id), 0) * self.weights[str(class_id)] for class_id in self.weights)
            scores[camera_id] = score

        return scores
