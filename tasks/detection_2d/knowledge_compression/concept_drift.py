"""
concept_drift.py
Giám sát Trôi dạt Khái niệm (Concept Drift) — Eq. 33-34 (Research Proposal).

Dùng cửa sổ trung bình trượt (Moving Average Window) trên W vòng lặp gần nhất
để theo dõi sự biến động của hàm mất mát. Kích hoạt tái phân cụm (Re-cluster)
nếu độ lệch vượt ngưỡng tĩnh epsilon_drift.
"""

from collections import deque
from typing import Dict, List


class ConceptDriftMonitor:
    """
    Theo dõi sự thay đổi của hàm mất mát cục bộ tại các AUV cảm biến.
    """
    def __init__(self, window_size: int = 5, epsilon_drift: float = 0.05):
        """
        Args:
            window_size (int):   W - kích thước cửa sổ trượt (số vòng lặp).
            epsilon_drift (float): ε_drift - ngưỡng dung sai hình học.
        """
        self.W = window_size
        self.epsilon = epsilon_drift
        
        # Lưu trữ lịch sử hàm loss cho từng auv (deque độ dài max = 2*W)
        self.history: Dict[int, deque] = {}

    def update(self, auv_id: int, current_loss: float):
        """Cập nhật lịch sử loss của auv."""
        if auv_id not in self.history:
            self.history[auv_id] = deque(maxlen=2 * self.W)
        self.history[auv_id].append(current_loss)

    def check_drift(self, auv_id: int) -> bool:
        """
        Kiểm tra xem auv_id có vi phạm ngưỡng drift không (Eq. 34).
        
        Trả về True nếu | L_bar(t) - L_bar(t-W) | > ε_drift
        """
        hist = self.history.get(auv_id)
        if hist is None or len(hist) < 2 * self.W:
            return False
            
        # Lấy W giá trị gần nhất [t-W+1 : t]
        recent_W = list(hist)[-self.W:]
        
        # Lấy W giá trị chu kỳ trước [t-2W+1 : t-W]
        past_W = list(hist)[-2 * self.W : -self.W]
        
        # Tính trung bình (Eq. 33)
        L_bar_current = sum(recent_W) / self.W
        L_bar_past = sum(past_W) / self.W
        
        # So sánh với ngưỡng (Eq. 34)
        return abs(L_bar_current - L_bar_past) > self.epsilon

    def check_global_drift(self) -> bool:
        """
        Kiểm tra trạng thái trôi dạt trên toàn mạng.
        (Ví dụ: Nếu bất kỳ auv nào báo cáo drift, hệ thống kích hoạt re-cluster)
        """
        for auv_id in self.history.keys():
            if self.check_drift(auv_id):
                return True
        return False

    def clear(self):
        """Reset lịch sử khi tái phân cụm."""
        self.history.clear()
