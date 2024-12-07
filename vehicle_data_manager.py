import time
class VehicleDataManager:
    def __init__(self):
        # Her kamera için araç türü verilerini tutar
        self.vehicle_data = {}  # {camera_id: {class_name: count}}

    def add_vehicle(self, camera_id, class_name):
        """Belirtilen kameradan bir araç türü ekler."""
        if camera_id not in self.vehicle_data:
            self.vehicle_data[camera_id] = {}
        if class_name in self.vehicle_data[camera_id]:
            self.vehicle_data[camera_id][class_name] += 1
        else:
            self.vehicle_data[camera_id][class_name] = 1

    def get_vehicle_data(self, camera_id):
        """Belirtilen kameranın araç verilerini döndürür."""
        return self.vehicle_data.get(camera_id, {})

    def get_all_vehicle_data(self):
        """Tüm kameraların araç verilerini liste halinde döndürür."""
        return self.vehicle_data

def get_all_vehicle_counts_by_class(vehicle_data_manager):
    """
    Tüm kameralar için her türdeki araç sayısını döndürür.

    Args:
        vehicle_data_manager (VehicleDataManager): Araç verilerini yöneten sınıf.

    Returns:
        dict: Her kamera için 0, 1, 2, 3 türündeki araç sayıları {camera_id: {class_id: count}}.
    """
    all_vehicle_data = vehicle_data_manager.get_all_vehicle_data()
    result = {}

    for camera_id, vehicle_data in all_vehicle_data.items():
        # Her tür için 0 olarak başlat ve varsa güncelle
        result[camera_id] = {
            "0": vehicle_data.get("0", 0),
            "1": vehicle_data.get("1", 0),
            "2": vehicle_data.get("2", 0),
            "3": vehicle_data.get("3", 0),
        }

    return result


def display_all_vehicle_counts(vehicle_data_manager):
    """
    Tüm kameraların 0, 1, 2, 3 türlerindeki araç sayımlarını yazdırır.

    Args:
        vehicle_data_manager (VehicleDataManager): Araç verilerini yöneten sınıf.
    """
    vehicle_counts = get_all_vehicle_counts_by_class(vehicle_data_manager)

    print("\nHer Kameradaki Araç Türü Sayıları:")
    for camera_id, counts in vehicle_counts.items():
        print(f"Kamera {camera_id}:")
        for class_id, count in counts.items():
            print(f"  Tür {class_id}: {count}")

