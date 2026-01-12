import numpy as np
from collections import deque # ★ 이 줄이 꼭 있어야 합니다!

class CoordinateSmoother:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history_x = []
        self.history_y = []

    def update(self, new_x, new_y):
        """새로운 좌표를 받아 평활화된(부드러운) 좌표를 반환합니다."""
        self.history_x.append(new_x)
        self.history_y.append(new_y)

        if len(self.history_x) > self.window_size:
            self.history_x.pop(0)
            self.history_y.pop(0)

        smooth_x = sum(self.history_x) / len(self.history_x)
        smooth_y = sum(self.history_y) / len(self.history_y)
        
        return smooth_x, smooth_y

def get_heat_center(pixels):
    """8x8 열화상 데이터에서 가장 뜨거운 지점의 좌표를 찾습니다."""
    idx = np.argmax(pixels)
    r, c = divmod(idx, 8)
    return r, c

class MultiScaleBuffer:
    def __init__(self, short_term_size=10, long_term_size=60):
        # 단기 버퍼: 충격 감지용
        self.short_term = deque(maxlen=short_term_size)
        # 장기 버퍼: 배회 감지용
        self.long_term = deque(maxlen=long_term_size)

    def update(self, impact, is_detected):
        self.short_term.append(impact)
        self.long_term.append(1 if is_detected else 0)

    def get_features(self):
        # 단기 특징: 최근 가장 큰 충격량
        short_term_impact = max(self.short_term) if self.short_term else 16384
        
        # 장기 특징: 전체 시간 중 객체가 머문 비율 (0~1 사이 값)
        loitering_score = sum(self.long_term) / len(self.long_term) if self.long_term else 0
        
        return short_term_impact, loitering_score