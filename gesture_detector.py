"""
Módulo de detecção e classificação de gestos para memes de gatos.
Usa MediaPipe HandLandmarker e geometria dos 21 landmarks da mão.
"""

import math
from typing import List, Tuple, Optional


def dist2d(p1, p2) -> float:
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def dist3d(p1, p2) -> float:
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)


def is_finger_extended(landmarks, tip_idx: int, pip_idx: int, mcp_idx: int, wrist_idx: int = 0) -> bool:
    wrist = landmarks[wrist_idx]
    mcp = landmarks[mcp_idx]
    pip = landmarks[pip_idx]
    tip = landmarks[tip_idx]

    d_tip_wrist = dist2d(tip, wrist)
    d_pip_wrist = dist2d(pip, wrist)
    d_tip_mcp = dist2d(tip, mcp)
    d_pip_mcp = dist2d(pip, mcp)

    return (d_tip_wrist > d_pip_wrist * 1.1) and (d_tip_mcp > 1.3 * d_pip_mcp)


def is_finger_folded(landmarks, tip_idx: int, pip_idx: int, mcp_idx: int, wrist_idx: int = 0) -> bool:
    wrist = landmarks[wrist_idx]
    mcp = landmarks[mcp_idx]
    pip = landmarks[pip_idx]
    tip = landmarks[tip_idx]

    d_tip_wrist = dist2d(tip, wrist)
    d_pip_wrist = dist2d(pip, wrist)
    d_tip_mcp = dist2d(tip, mcp)
    d_pip_mcp = dist2d(pip, mcp)

    return (d_tip_mcp < 1.35 * d_pip_mcp) or (d_tip_wrist < d_pip_wrist * 1.15)


def is_index_pointing_up(landmarks) -> bool:
    """Gesto do meme 'Erm Ackshuallay' (☝️)"""
    mcp = landmarks[5]
    pip = landmarks[6]
    tip = landmarks[8]

    dy = mcp.y - tip.y
    dx = abs(tip.x - mcp.x)

    pointing_up = (tip.y < pip.y) and (pip.y < mcp.y) and (dy > 0.08)
    mostly_vertical = dy > 1.2 * dx
    extended = is_finger_extended(landmarks, 8, 6, 5, 0)

    middle_folded = is_finger_folded(landmarks, 12, 10, 9, 0)
    ring_folded = is_finger_folded(landmarks, 16, 14, 13, 0)
    pinky_folded = is_finger_folded(landmarks, 20, 18, 17, 0)

    thumb_tip = landmarks[4]
    thumb_not_up = (thumb_tip.y > tip.y + 0.04)

    return pointing_up and mostly_vertical and extended and middle_folded and ring_folded and pinky_folded and thumb_not_up


def is_thumbs_up(landmarks) -> bool:
    """Gesto do meme do Thumbs Up Cat (👍)"""
    wrist = landmarks[0]
    thumb_cmc = landmarks[1]
    thumb_mcp = landmarks[2]
    thumb_ip = landmarks[3]
    thumb_tip = landmarks[4]

    # Polegar apontado para cima
    thumb_up = (thumb_tip.y < thumb_ip.y) and (thumb_ip.y < thumb_mcp.y)
    dy_thumb = thumb_mcp.y - thumb_tip.y
    mostly_vertical_thumb = dy_thumb > 0.05

    # Todos os 4 dedos restantes devem estar recolhidos
    idx_folded = is_finger_folded(landmarks, 8, 6, 5, 0)
    mid_folded = is_finger_folded(landmarks, 12, 10, 9, 0)
    rng_folded = is_finger_folded(landmarks, 16, 14, 13, 0)
    pky_folded = is_finger_folded(landmarks, 20, 18, 17, 0)

    idx_ext = is_finger_extended(landmarks, 8, 6, 5, 0)

    # Não deve estar tocando o indicador (para não confundir com finger heart)
    d_thumb_idx = dist2d(thumb_tip, landmarks[8])

    return (thumb_up and mostly_vertical_thumb and
            idx_folded and mid_folded and rng_folded and pky_folded and
            not idx_ext and d_thumb_idx > 0.07)


def is_rock_sign(landmarks) -> bool:
    """Gesto do meme Guitar / Rock Cat (🤘)"""
    # Indicador e Mínimo estendidos
    idx_ext = is_finger_extended(landmarks, 8, 6, 5, 0)
    pky_ext = is_finger_extended(landmarks, 20, 18, 17, 0)

    # Médio e Anelar dobrados
    mid_folded = is_finger_folded(landmarks, 12, 10, 9, 0)
    rng_folded = is_finger_folded(landmarks, 16, 14, 13, 0)

    idx_tip = landmarks[8]
    idx_pip = landmarks[6]
    pky_tip = landmarks[20]
    pky_pip = landmarks[18]

    # Indicador e mínimo apontando para cima
    upward = (idx_tip.y < idx_pip.y) and (pky_tip.y < pky_pip.y)

    return idx_ext and pky_ext and mid_folded and rng_folded and upward


def is_finger_gun(landmarks) -> bool:
    """Gesto da Pistola de Dedos (👉 / 🔫)"""
    idx_ext = is_finger_extended(landmarks, 8, 6, 5, 0)
    idx_tip = landmarks[8]
    idx_mcp = landmarks[5]

    # Anelar e mínimo dobrados
    rng_folded = is_finger_folded(landmarks, 16, 14, 13, 0)
    pky_folded = is_finger_folded(landmarks, 20, 18, 17, 0)

    # Polegar apontado para cima (o 'cão' da pistola)
    thumb_tip = landmarks[4]
    thumb_mcp = landmarks[2]
    thumb_up = (thumb_tip.y < thumb_mcp.y)

    # Indicador predominantemente horizontal
    dx = abs(idx_tip.x - idx_mcp.x)
    dy = abs(idx_tip.y - idx_mcp.y)
    horizontal = (dx > 0.07) and (dx > dy * 0.7)

    d_thumb_idx = dist2d(thumb_tip, idx_tip)

    return idx_ext and rng_folded and pky_folded and thumb_up and horizontal and (d_thumb_idx > 0.08)


def is_finger_heart(landmarks) -> bool:
    """Gesto do Mini Coração Coreano (🫰)"""
    thumb_tip = landmarks[4]
    thumb_mcp = landmarks[2]
    idx_tip = landmarks[8]
    mid_tip = landmarks[12]

    # Pontas do polegar e indicador muito próximas (pinçadas/cruzadas)
    d_thumb_idx = dist2d(thumb_tip, idx_tip)

    # Médio, anelar e mínimo dobrados
    mid_folded = is_finger_folded(landmarks, 12, 10, 9, 0)
    rng_folded = is_finger_folded(landmarks, 16, 14, 13, 0)
    pky_folded = is_finger_folded(landmarks, 20, 18, 17, 0)

    # As pontas do polegar e indicador devem estar visivelmente mais altas que os outros dedos dobrados
    # No OpenCV, y menor = mais alto na imagem
    higher = (thumb_tip.y < mid_tip.y - 0.03) and (idx_tip.y < mid_tip.y - 0.03)
    thumb_up = thumb_tip.y < thumb_mcp.y

    return (d_thumb_idx < 0.065) and mid_folded and rng_folded and pky_folded and higher and thumb_up


def is_fist_facing_camera(landmarks) -> bool:
    """Gesto do Murro do Gato (👊)"""
    idx_folded = is_finger_folded(landmarks, 8, 6, 5, 0)
    mid_folded = is_finger_folded(landmarks, 12, 10, 9, 0)
    rng_folded = is_finger_folded(landmarks, 16, 14, 13, 0)
    pky_folded = is_finger_folded(landmarks, 20, 18, 17, 0)

    idx_ext = is_finger_extended(landmarks, 8, 6, 5, 0)
    mid_ext = is_finger_extended(landmarks, 12, 10, 9, 0)
    rng_ext = is_finger_extended(landmarks, 16, 14, 13, 0)
    pky_ext = is_finger_extended(landmarks, 20, 18, 17, 0)

    if idx_ext or mid_ext or rng_ext or pky_ext:
        return False

    all_four_folded = idx_folded and mid_folded and rng_folded and pky_folded

    wrist = landmarks[0]
    thumb_tip = landmarks[4]
    pinky_mcp = landmarks[17]
    middle_mcp = landmarks[9]

    d_wrist_mid = dist2d(wrist, middle_mcp)
    d_thumb_pky = dist2d(thumb_tip, pinky_mcp)
    thumb_folded = (d_thumb_pky < d_wrist_mid * 1.5) or (dist2d(thumb_tip, wrist) < d_wrist_mid * 1.3)

    xs = [p.x for p in landmarks]
    ys = [p.y for p in landmarks]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    aspect = width / max(height, 1e-5)
    compact = 0.4 < aspect < 2.2

    return all_four_folded and thumb_folded and compact


def is_shy_pose(all_hand_landmarks: List[List]) -> bool:
    """Gesto do Gato Tímido Corado (👉👈) com duas mãos"""
    if len(all_hand_landmarks) < 2:
        return False

    h1 = all_hand_landmarks[0]
    h2 = all_hand_landmarks[1]

    if h1[5].x < h2[5].x:
        left_hand, right_hand = h1, h2
    else:
        left_hand, right_hand = h2, h1

    for hand in [left_hand, right_hand]:
        if not is_finger_extended(hand, 8, 6, 5, 0):
            return False
        if not is_finger_folded(hand, 12, 10, 9, 0):
            return False
        if not is_finger_folded(hand, 16, 14, 13, 0):
            return False
        if not is_finger_folded(hand, 20, 18, 17, 0):
            return False

    left_mcp = left_hand[5]
    left_tip = left_hand[8]
    dx_left = left_tip.x - left_mcp.x
    dy_left = abs(left_tip.y - left_mcp.y)

    right_mcp = right_hand[5]
    right_tip = right_hand[8]
    dx_right = right_tip.x - right_mcp.x
    dy_right = abs(right_tip.y - right_mcp.y)

    left_points_right = (dx_left > 0.02) and (dx_left > dy_left * 0.4)
    right_points_left = (dx_right < -0.02) and (abs(dx_right) > dy_right * 0.4)

    same_horizontal_level = abs(left_tip.y - right_tip.y) < 0.30

    d_tips = dist2d(left_tip, right_tip)
    d_mcps = dist2d(left_mcp, right_mcp)
    converging = (d_tips < d_mcps * 1.2) or (d_tips < 0.30)

    return left_points_right and right_points_left and same_horizontal_level and converging


def classify_gesture(all_hand_landmarks: List[List]) -> str:
    """
    Retorna uma das seguintes tags:
    - 'shy': duas mãos com indicadores apontando um para o outro (👉👈)
    - 'rock_guitar': corninhos do rock (🤘)
    - 'thumbs_up': polegar para cima (👍)
    - 'finger_heart': mini coração com dedos (🫰)
    - 'erhm_actually': indicador para cima (☝️)
    - 'finger_gun': pistola de dedos (👉 / 🔫)
    - 'punch': punho fechado virado para a câmara (👊)
    - 'neutral': mão neutra/outra pose ou nenhuma mão visível
    """
    if not all_hand_landmarks or len(all_hand_landmarks) == 0:
        return 'neutral'

    # 1. Checa pose com duas mãos (shy)
    if len(all_hand_landmarks) >= 2 and is_shy_pose(all_hand_landmarks):
        return 'shy'

    # 2. Gestos de mão única (em ordem de especificidade)
    for landmarks in all_hand_landmarks:
        if is_finger_heart(landmarks):
            return 'finger_heart'

    for landmarks in all_hand_landmarks:
        if is_rock_sign(landmarks):
            return 'rock_guitar'

    for landmarks in all_hand_landmarks:
        if is_thumbs_up(landmarks):
            return 'thumbs_up'

    for landmarks in all_hand_landmarks:
        if is_index_pointing_up(landmarks):
            return 'erhm_actually'

    for landmarks in all_hand_landmarks:
        if is_finger_gun(landmarks):
            return 'finger_gun'

    for landmarks in all_hand_landmarks:
        if is_fist_facing_camera(landmarks):
            return 'punch'

    return 'neutral'


class GestureSmoother:
    """
    Filtro de suavização temporal para evitar cintilação rápida entre quadros.
    """
    def __init__(self, history_size: int = 5, min_consecutive: int = 3):
        self.history_size = history_size
        self.min_consecutive = min_consecutive
        self.history: List[str] = []
        self.current_gesture: str = 'neutral'

    def update(self, detected_gesture: str) -> str:
        self.history.append(detected_gesture)
        if len(self.history) > self.history_size:
            self.history.pop(0)

        recent = self.history[-self.min_consecutive:]
        if len(recent) == self.min_consecutive and all(g == recent[0] for g in recent):
            self.current_gesture = recent[0]
        elif detected_gesture == 'neutral' and all(g == 'neutral' for g in self.history[-2:]):
            self.current_gesture = 'neutral'

        return self.current_gesture


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]


def draw_hand_landmarks(frame, landmarks, color=(0, 255, 128), line_color=(255, 200, 0)):
    import cv2
    h, w, _ = frame.shape
    coords = []
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        coords.append((cx, cy))

    for start_idx, end_idx in HAND_CONNECTIONS:
        if start_idx < len(coords) and end_idx < len(coords):
            cv2.line(frame, coords[start_idx], coords[end_idx], line_color, 2, cv2.LINE_AA)

    for i, (cx, cy) in enumerate(coords):
        if i in (4, 8, 12, 16, 20):
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 8, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (cx, cy), 4, color, -1, cv2.LINE_AA)
