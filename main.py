#!/usr/bin/env python3
import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cat'

import sys
import time
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import gesture_detector as gd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
MODEL_PATH = os.path.join(BASE_DIR, 'hand_landmarker.task')
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

MEME_FILES = {
    'neutral': os.path.join(ASSETS_DIR, 'neutral.jpg'),
    'erhm_actually': os.path.join(ASSETS_DIR, 'erhm_actually.jpg'),
    'punch': os.path.join(ASSETS_DIR, 'punch.jpg'),
    'shy': os.path.join(ASSETS_DIR, 'shy.jpg'),
    'thumbs_up': os.path.join(ASSETS_DIR, 'thumbs_up.png'),
    'rock_guitar': os.path.join(ASSETS_DIR, 'rock_guitar.jpg'),
    'peace_sign': os.path.join(ASSETS_DIR, 'peace_sign.jpg'),
}


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def load_memes():
    images = {}
    for key, path in MEME_FILES.items():
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                images[key] = img
                continue
        images[key] = np.zeros((600, 600, 3), dtype=np.uint8)
    return images


def fit_box(img, target_w, target_h):
    h, w, _ = img.shape
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = resized
    return canvas


def main():
    ensure_model()
    meme_images = load_memes()

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)

    camera_id = 0
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Erro ao abrir câmara {camera_id}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    window_name = "Cat Meme Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 600)

    smoother = gd.GestureSmoother(history_size=6, min_consecutive=3)
    view_mode = "split"
    show_pip = True
    show_landmarks = True

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02)
            continue

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)
        hands_landmarks = detection_result.hand_landmarks

        if show_landmarks:
            for landmarks in hands_landmarks:
                gd.draw_hand_landmarks(frame, landmarks)

        raw_gesture = gd.classify_gesture(hands_landmarks)
        current_gesture = smoother.update(raw_gesture)
        meme_img = meme_images.get(current_gesture, meme_images['neutral'])

        if view_mode == "split":
            cam_display = fit_box(frame, 640, 600)
            meme_display = fit_box(meme_img, 640, 600)
            final_screen = np.hstack([cam_display, meme_display])
        else:
            meme_display = fit_box(meme_img, 900, 700)
            if show_pip:
                pip_w, pip_h = 240, 180
                small_cam = cv2.resize(frame, (pip_w, pip_h), interpolation=cv2.INTER_AREA)
                cv2.rectangle(small_cam, (0, 0), (pip_w - 1, pip_h - 1), (80, 80, 80), 1)
                meme_display[15:15+pip_h, 900-pip_w-15:900-15] = small_cam
            final_screen = meme_display

        cv2.imshow(window_name, final_screen)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('m'):
            view_mode = "meme_only" if view_mode == "split" else "split"
        elif key == ord('p'):
            show_pip = not show_pip
        elif key == ord('l'):
            show_landmarks = not show_landmarks
        elif key == ord('c'):
            camera_id = (camera_id + 1) % 4
            cap.release()
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                camera_id = 0
                cap = cv2.VideoCapture(camera_id)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
