import cv2


class ROIManager:
    def __init__(self, frame_width, frame_height):
        """
        ROIManager sınıfı, bir ROI'nin (Region of Interest) tanımlanmasını ve kontrol edilmesini sağlar.
        """
        self.roi_x1 = 0
        self.roi_y1 = 0
        self.roi_x2 = frame_width // 2  # Varsayılan olarak ekranın sol yarısını alır
        self.roi_y2 = frame_height

    def is_inside_roi(self, x, y):
        """Bir noktanın ROI içinde olup olmadığını kontrol eder."""
        return self.roi_x1 <= x <= self.roi_x2 and self.roi_y1 <= y <= self.roi_y2

    def draw_roi(self, frame):
        """Çerçevede ROI'yi çizer."""
        cv2.rectangle(frame, (self.roi_x1, self.roi_y1), (self.roi_x2, self.roi_y2), (0, 255, 255), 2)
        cv2.putText(frame, "ROI", (self.roi_x1 + 10, self.roi_y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
