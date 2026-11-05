import math
from typing import List


def dist2d(p1, p2) -> float:
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def is_finger_extended(landmarks, tip_idx: int, pip_idx: int, mcp_idx: int, wrist_idx: int = 0) -> bool:
    wrist = landmarks[wrist_idx]
    mcp = landmarks[mcp_idx]
    pip = landmarks[pip_idx]
    tip = landmarks[tip_idx]
    return (dist2d(tip, wrist) > dist2d(pip, wrist) * 1.1) and (dist2d(tip, mcp) > 1.3 * dist2d(pip, mcp))


def is_finger_folded(landmarks, tip_idx: int, pip_idx: int, mcp_idx: int, wrist_idx: int = 0) -> bool:
    wrist = landmarks[wrist_idx]
    mcp = landmarks[mcp_idx]
    pip = landmarks[pip_idx]
    tip = landmarks[tip_idx]
    return (dist2d(tip, mcp) < 1.35 * dist2d(pip, mcp)) or (dist2d(tip, wrist) < dist2d(pip, wrist) * 1.15)


def is_index_pointing_up(landmarks) -> bool:
    mcp = landmarks[5]
    pip = landmarks[6]
    tip = landmarks[8]

    dy = mcp.y - tip.y
    dx = abs(tip.x - mcp.x)

    pointing_up = (tip.y < pip.y < mcp.y) and (dy > 0.08) and (dy > 1.2 * dx)
    extended = is_finger_extended(landmarks, 8, 6, 5, 0)

    other_folded = (
        is_finger_folded(landmarks, 12, 10, 9, 0)
        and is_finger_folded(landmarks, 16, 14, 13, 0)
        and is_finger_folded(landmarks, 20, 18, 17, 0)
    )

    thumb_not_up = landmarks[4].y > tip.y + 0.04
    return pointing_up and extended and other_folded and thumb_not_up


def is_thumbs_up(landmarks) -> bool:
    thumb_mcp = landmarks[2]
    thumb_ip = landmarks[3]
    thumb_tip = landmarks[4]

    thumb_up = (thumb_tip.y < thumb_ip.y < thumb_mcp.y) and ((thumb_mcp.y - thumb_tip.y) > 0.05)
    others_folded = (
        is_finger_folded(landmarks, 8, 6, 5, 0)
        and is_finger_folded(landmarks, 12, 10, 9, 0)
        and is_finger_folded(landmarks, 16, 14, 13, 0)
        and is_finger_folded(landmarks, 20, 18, 17, 0)
    )
    not_index_ext = not is_finger_extended(landmarks, 8, 6, 5, 0)
    d_thumb_idx = dist2d(thumb_tip, landmarks[8])

    return thumb_up and others_folded and not_index_ext and (d_thumb_idx > 0.07)


def is_rock_sign(landmarks) -> bool:
    idx_ext = is_finger_extended(landmarks, 8, 6, 5, 0)
    pky_ext = is_finger_extended(landmarks, 20, 18, 17, 0)
    mid_folded = is_finger_folded(landmarks, 12, 10, 9, 0)
    rng_folded = is_finger_folded(landmarks, 16, 14, 13, 0)

    upward = (landmarks[8].y < landmarks[6].y) and (landmarks[20].y < landmarks[18].y)
    return idx_ext and pky_ext and mid_folded and rng_folded and upward


def is_peace_sign(landmarks) -> bool:
    idx_ext = is_finger_extended(landmarks, 8, 6, 5, 0)
    mid_ext = is_finger_extended(landmarks, 12, 10, 9, 0)
    rng_folded = is_finger_folded(landmarks, 16, 14, 13, 0)
    pky_folded = is_finger_folded(landmarks, 20, 18, 17, 0)

    return idx_ext and mid_ext and rng_folded and pky_folded


def is_fist_facing_camera(landmarks) -> bool:
    if any(is_finger_extended(landmarks, t, p, m, 0) for t, p, m in [(8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17)]):
        return False

    all_four_folded = (
        is_finger_folded(landmarks, 8, 6, 5, 0)
        and is_finger_folded(landmarks, 12, 10, 9, 0)
        and is_finger_folded(landmarks, 16, 14, 13, 0)
        and is_finger_folded(landmarks, 20, 18, 17, 0)
    )

    wrist = landmarks[0]
    thumb_tip = landmarks[4]
    d_wrist_mid = dist2d(wrist, landmarks[9])
    thumb_folded = (dist2d(thumb_tip, landmarks[17]) < d_wrist_mid * 1.5) or (dist2d(thumb_tip, wrist) < d_wrist_mid * 1.3)

    xs = [p.x for p in landmarks]
    ys = [p.y for p in landmarks]
    aspect = (max(xs) - min(xs)) / max(max(ys) - min(ys), 1e-5)
    compact = 0.4 < aspect < 2.2

    return all_four_folded and thumb_folded and compact


def is_shy_pose(all_hand_landmarks: List[List]) -> bool:
    if len(all_hand_landmarks) < 2:
        return False

    h1, h2 = all_hand_landmarks[0], all_hand_landmarks[1]
    left_hand, right_hand = (h1, h2) if h1[5].x < h2[5].x else (h2, h1)

    for hand in [left_hand, right_hand]:
        if not is_finger_extended(hand, 8, 6, 5, 0):
            return False
        if any(not is_finger_folded(hand, t, p, m, 0) for t, p, m in [(12, 10, 9), (16, 14, 13), (20, 18, 17)]):
            return False

    left_mcp, left_tip = left_hand[5], left_hand[8]
    right_mcp, right_tip = right_hand[5], right_hand[8]

    dx_left = left_tip.x - left_mcp.x
    dy_left = abs(left_tip.y - left_mcp.y)
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
    if not all_hand_landmarks:
        return 'neutral'

    if len(all_hand_landmarks) >= 2 and is_shy_pose(all_hand_landmarks):
        return 'shy'

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
        if is_peace_sign(landmarks):
            return 'peace_sign'

    for landmarks in all_hand_landmarks:
        if is_fist_facing_camera(landmarks):
            return 'punch'

    return 'neutral'


class GestureSmoother:
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
    coords = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for start_idx, end_idx in HAND_CONNECTIONS:
        if start_idx < len(coords) and end_idx < len(coords):
            cv2.line(frame, coords[start_idx], coords[end_idx], line_color, 2, cv2.LINE_AA)

    for i, pt in enumerate(coords):
        if i in (4, 8, 12, 16, 20):
            cv2.circle(frame, pt, 6, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 8, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.circle(frame, pt, 4, color, -1, cv2.LINE_AA)
