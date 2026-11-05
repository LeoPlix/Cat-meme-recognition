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

    # Dedo está dobrado se a ponta estiver recolhida próxima ao MCP ou pulso
    return (d_tip_mcp < 1.35 * d_pip_mcp) or (d_tip_wrist < d_pip_wrist * 1.15)


def is_index_pointing_up(landmarks) -> bool:
    wrist = landmarks[0]
    mcp = landmarks[5]
    pip = landmarks[6]
    tip = landmarks[8]

    # No OpenCV/MediaPipe, y=0 é o topo e y=1 é a base.
    # Para apontar para cima, tip.y < pip.y < mcp.y
    dy = mcp.y - tip.y
    dx = abs(tip.x - mcp.x)

    pointing_up = (tip.y < pip.y) and (pip.y < mcp.y) and (dy > 0.08)
    mostly_vertical = dy > 1.2 * dx
    extended = is_finger_extended(landmarks, 8, 6, 5, 0)

    # Outros dedos (médio, anelar, mínimo) devem estar dobrados
    middle_folded = is_finger_folded(landmarks, 12, 10, 9, 0)
    ring_folded = is_finger_folded(landmarks, 16, 14, 13, 0)
    pinky_folded = is_finger_folded(landmarks, 20, 18, 17, 0)

    # Polegar não deve estar esticado para cima como um 'L' alto
    thumb_tip = landmarks[4]
    thumb_not_up = (thumb_tip.y > tip.y + 0.04)

    return pointing_up and mostly_vertical and extended and middle_folded and ring_folded and pinky_folded and thumb_not_up


def is_fist_facing_camera(landmarks) -> bool:
    # Todos os 4 dedos principais devem estar dobrados
    idx_folded = is_finger_folded(landmarks, 8, 6, 5, 0)
    mid_folded = is_finger_folded(landmarks, 12, 10, 9, 0)
    rng_folded = is_finger_folded(landmarks, 16, 14, 13, 0)
    pky_folded = is_finger_folded(landmarks, 20, 18, 17, 0)

    idx_ext = is_finger_extended(landmarks, 8, 6, 5, 0)
    mid_ext = is_finger_extended(landmarks, 12, 10, 9, 0)
    rng_ext = is_finger_extended(landmarks, 16, 14, 13, 0)
    pky_ext = is_finger_extended(landmarks, 20, 18, 17, 0)

    # Nenhum dedo esticado
    if idx_ext or mid_ext or rng_ext or pky_ext:
        return False

    all_four_folded = idx_folded and mid_folded and rng_folded and pky_folded

    # Polegar recolhido junto aos dedos ou palma
    wrist = landmarks[0]
    thumb_tip = landmarks[4]
    pinky_mcp = landmarks[17]
    middle_mcp = landmarks[9]

    d_wrist_mid = dist2d(wrist, middle_mcp)
    d_thumb_pky = dist2d(thumb_tip, pinky_mcp)
    thumb_folded = (d_thumb_pky < d_wrist_mid * 1.5) or (dist2d(thumb_tip, wrist) < d_wrist_mid * 1.3)

    # Proporção compacta do punho (não espalhado)
    xs = [p.x for p in landmarks]
    ys = [p.y for p in landmarks]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    aspect = width / max(height, 1e-5)
    compact = 0.4 < aspect < 2.2

    return all_four_folded and thumb_folded and compact


def is_shy_pose(all_hand_landmarks: List[List]) -> bool:
    """
    Verifica a pose 'shy' (👉👈):
    Duas mãos com os indicadores apontando horizontalmente um em direção ao outro,
    e os outros dedos dobrados.
    """
    if len(all_hand_landmarks) < 2:
        return False

    # Pega as duas mãos
    h1 = all_hand_landmarks[0]
    h2 = all_hand_landmarks[1]

    # Ordena as mãos da esquerda para a direita na tela (menor x = esquerda)
    # Usamos o MCP do indicador (5) ou o pulso (0)
    if h1[5].x < h2[5].x:
        left_hand, right_hand = h1, h2
    else:
        left_hand, right_hand = h2, h1

    # Ambas devem ter o indicador esticado e os outros dedos dobrados
    for hand in [left_hand, right_hand]:
        if not is_finger_extended(hand, 8, 6, 5, 0):
            return False
        if not is_finger_folded(hand, 12, 10, 9, 0):
            return False
        if not is_finger_folded(hand, 16, 14, 13, 0):
            return False
        if not is_finger_folded(hand, 20, 18, 17, 0):
            return False

    # Mão esquerda: o indicador aponta para a direita (x da ponta > x da base)
    left_mcp = left_hand[5]
    left_tip = left_hand[8]
    dx_left = left_tip.x - left_mcp.x
    dy_left = abs(left_tip.y - left_mcp.y)

    # Mão direita: o indicador aponta para a esquerda (x da ponta < x da base)
    right_mcp = right_hand[5]
    right_tip = right_hand[8]
    dx_right = right_tip.x - right_mcp.x
    dy_right = abs(right_tip.y - right_mcp.y)

    # Verificações de direção horizontal
    left_points_right = (dx_left > 0.02) and (dx_left > dy_left * 0.4)
    right_points_left = (dx_right < -0.02) and (abs(dx_right) > dy_right * 0.4)

    # Indicadores devem estar mais ou menos na mesma altura vertical
    same_horizontal_level = abs(left_tip.y - right_tip.y) < 0.30

    # Distância entre as pontas deve ser próxima ou menor que a distância entre as bases (convergindo)
    d_tips = dist2d(left_tip, right_tip)
    d_mcps = dist2d(left_mcp, right_mcp)
    converging = (d_tips < d_mcps * 1.2) or (d_tips < 0.30)

    return left_points_right and right_points_left and same_horizontal_level and converging


def classify_gesture(all_hand_landmarks: List[List]) -> str:
    """
    Retorna uma das seguintes tags:
    - 'shy': duas mãos com indicadores apontando um para o outro (👉👈)
    - 'erhm_actually': mão com indicador apontando para cima (☝️)
    - 'punch': mão em punho fechado virado para a câmara (👊)
    - 'neutral': mão neutra/outra pose ou nenhuma mão visível
    """
    if not all_hand_landmarks or len(all_hand_landmarks) == 0:
        return 'neutral'

    # 1. Checa pose shy (requer 2 mãos)
    if len(all_hand_landmarks) >= 2 and is_shy_pose(all_hand_landmarks):
        return 'shy'

    # 2. Checa gestos de uma mão (se houver mais de uma mão, checa em cada uma)
    for landmarks in all_hand_landmarks:
        if is_index_pointing_up(landmarks):
            return 'erhm_actually'

    for landmarks in all_hand_landmarks:
        if is_fist_facing_camera(landmarks):
            return 'punch'

    # 3. Se nenhuma condição ativa foi satisfeita -> neutro
    return 'neutral'


class GestureSmoother:
    """
    Filtro de suavização temporal para evitar que o meme oscile ou pisque
    rapidamente entre quadros por ruído na detecção.
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

        # Se os últimos `min_consecutive` quadros forem iguais, muda de estado
        recent = self.history[-self.min_consecutive:]
        if len(recent) == self.min_consecutive and all(g == recent[0] for g in recent):
            self.current_gesture = recent[0]
        elif detected_gesture == 'neutral' and all(g == 'neutral' for g in self.history[-2:]):
            # Transição mais rápida de volta para neutro se a mão sumir
            self.current_gesture = 'neutral'

        return self.current_gesture


HAND_CONNECTIONS = [
    # Polegar
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Indicador
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Médio
    (9, 10), (10, 11), (11, 12),
    # Anelar
    (13, 14), (14, 15), (15, 16),
    # Mínimo
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palma
    (5, 9), (9, 13), (13, 17)
]


def draw_hand_landmarks(frame, landmarks, color=(0, 255, 128), line_color=(255, 200, 0)):
    """
    Desenha os landmarks e as linhas de conexão da mão diretamente na imagem OpenCV.
    """
    import cv2
    h, w, _ = frame.shape
    coords = []
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        coords.append((cx, cy))

    # Desenha conexões
    for start_idx, end_idx in HAND_CONNECTIONS:
        if start_idx < len(coords) and end_idx < len(coords):
            cv2.line(frame, coords[start_idx], coords[end_idx], line_color, 2, cv2.LINE_AA)

    # Desenha pontos (landmarks)
    for i, (cx, cy) in enumerate(coords):
        # Destaque para as pontas dos dedos
        if i in (4, 8, 12, 16, 20):
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 8, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (cx, cy), 4, color, -1, cv2.LINE_AA)

