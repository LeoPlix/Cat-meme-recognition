#!/usr/bin/env python3
"""
Cat Meme Recognition - Detecção de Poses de Mão com Memes de Gato
----------------------------------------------------------------
Gestos reconhecidos:
  1. Neutro / sem mãos: Gato parado (neutral.jpg)
  2. Indicador para cima: Meme 'Erm Ackshuallay' (erhm_actually.jpg)
  3. Punho fechado virado para a câmara: Gato a dar murro (punch.jpg)
  4. Duas mãos com indicadores apontando um para o outro: Gato tímido corado (shy.jpg)

Controlos no teclado:
  - 'q' ou ESC: Sair
  - 'm': Alternar entre visualização lado a lado e ecrã de meme com PiP
  - 'p': Alternar miniatura da câmara (PiP) no modo meme
  - 'c': Trocar índice da câmara (0 -> 1 -> ...)
"""

import os
import sys
import time

# Evita avisos de permissão do Matplotlib
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cat'

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
    'finger_gun': os.path.join(ASSETS_DIR, 'finger_gun.jpg'),
    'finger_heart': os.path.join(ASSETS_DIR, 'finger_heart.png'),
}

MEME_TITLES = {
    'neutral': "Gato Parado (Neutro)",
    'erhm_actually': "Erm Ackshuallay! (☝️)",
    'punch': "Murro do Gato! (👊)",
    'shy': "Gato Corado (👉👈)",
    'thumbs_up': "Thumbs Up Cat! (👍)",
    'rock_guitar': "Rock Guitar Cat! (🤘🎸)",
    'finger_gun': "Finger Gun Cat! (👉🔫)",
    'finger_heart': "Finger Heart Cat! (🫰💖)",
}

MEME_COLORS = {
    'neutral': (180, 180, 180),       # Cinzento
    'erhm_actually': (0, 215, 255),    # Amarelo/Dourado
    'punch': (50, 50, 255),           # Vermelho vivo
    'shy': (200, 130, 255),           # Rosa / Blush
    'thumbs_up': (0, 255, 128),       # Verde positivo
    'rock_guitar': (0, 140, 255),     # Laranja elétrico
    'finger_gun': (255, 200, 50),     # Azul/Ciano claro
    'finger_heart': (180, 105, 255),  # Magenta coração
}


def ensure_model_downloaded():
    if not os.path.exists(MODEL_PATH):
        print(f"[+] Descarregando modelo do MediaPipe para {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[+] Modelo descarregado com sucesso!")


def load_meme_images():
    images = {}
    for key, path in MEME_FILES.items():
        if not os.path.exists(path):
            print(f"[!] AVISO: Imagem {path} não encontrada!")
            placeholder = np.zeros((600, 600, 3), dtype=np.uint8)
            cv2.putText(placeholder, key, (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            images[key] = placeholder
        else:
            img = cv2.imread(path)
            if img is None:
                print(f"[!] Erro ao carregar {path}")
                img = np.zeros((600, 600, 3), dtype=np.uint8)
            images[key] = img
    return images


def fit_image_to_box(img, target_w, target_h):
    """Redimensiona a imagem preservando a proporção de aspeto e adiciona letterbox."""
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
    ensure_model_downloaded()
    meme_images = load_meme_images()

    # Inicializa o detector MediaPipe
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
        print(f"[!] Não foi possível abrir a câmara {camera_id}.")
        print("[!] Tente especificar outro índice da câmara.")
        return

    # Tenta definir resolução 640x480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    window_name = "Reconhecimento de Gestos - Gato Meme"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 600)

    smoother = gd.GestureSmoother(history_size=6, min_consecutive=3)

    view_mode = "split"      # "split" ou "meme_only"
    show_pip = True          # Exibe miniatura da webcam no modo meme_only
    show_landmarks = True    # Exibe linhas de rastreio da mão na câmara

    print("\n" + "="*50)
    print(" 🐱 CAT MEME RECOGNITION INICIADO! 🐱 ")
    print("="*50)
    print("Gestos suportados:")
    print("  • Mão neutra ou sem mãos       -> Gato Parado")
    print("  • Indicador para cima (☝️)      -> Erm Ackshuallay")
    print("  • Punho virado p/ câmara (👊)   -> Gato Murro")
    print("  • Indicadores a apontar (👉👈)  -> Gato Corado")
    print("  • Polegar para cima (👍)        -> Thumbs Up Cat")
    print("  • Sinal do Rock (🤘)           -> Rock Guitar Cat")
    print("  • Pistola com dedos (👉 / 🔫)   -> Finger Gun Cat")
    print("  • Mini coração coreano (🫰)     -> Finger Heart Cat")
    print("\nTeclas de controlo:")
    print("  'm': Alternar vista (Lado a Lado / Só Meme)")
    print("  'p': Ligar/desligar miniatura (PiP) no modo meme")
    print("  'l': Ligar/desligar esqueleto da mão na câmara")
    print("  'c': Trocar índice de câmara")
    print("  'q' ou ESC: Sair")
    print("="*50 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Falha ao ler quadro da câmara.")
            time.sleep(0.05)
            continue

        # Inverte horizontalmente para efeito espelho natural
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Converte para formato do MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = detector.detect(mp_image)
        hands_landmarks = detection_result.hand_landmarks

        # Desenha esqueleto das mãos no frame da câmara (opcional)
        if show_landmarks:
            for landmarks in hands_landmarks:
                gd.draw_hand_landmarks(frame, landmarks)

        # Classifica e suaviza o gesto detectado
        raw_gesture = gd.classify_gesture(hands_landmarks)
        current_gesture = smoother.update(raw_gesture)

        # Pega a imagem do meme atual (sem texto)
        meme_img = meme_images.get(current_gesture, meme_images['neutral'])

        # Renderização limpa conforme o modo de visualização
        target_display_h = 600
        target_display_w = 640

        if view_mode == "split":
            cam_display = fit_image_to_box(frame, target_display_w, target_display_h)
            meme_display = fit_image_to_box(meme_img, target_display_w, target_display_h)
            final_screen = np.hstack([cam_display, meme_display])

        else:
            # Modo apenas meme em destaque
            full_w = 900
            full_h = 700
            meme_display = fit_image_to_box(meme_img, full_w, full_h)

            # Picture-in-Picture (PiP) da câmara no canto superior direito
            if show_pip:
                pip_w = 240
                pip_h = 180
                small_cam = cv2.resize(frame, (pip_w, pip_h), interpolation=cv2.INTER_AREA)
                cv2.rectangle(small_cam, (0, 0), (pip_w - 1, pip_h - 1), (100, 100, 100), 2)
                meme_display[15:15+pip_h, full_w-pip_w-15:full_w-15] = small_cam

            final_screen = meme_display

        cv2.imshow(window_name, final_screen)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # 'q' ou ESC
            break
        elif key == ord('m'):
            view_mode = "meme_only" if view_mode == "split" else "split"
        elif key == ord('p'):
            show_pip = not show_pip
        elif key == ord('l'):
            show_landmarks = not show_landmarks
        elif key == ord('c'):
            # Alterna para próxima câmara
            camera_id = (camera_id + 1) % 4
            cap.release()
            print(f"[*] Tentando alternar para câmara {camera_id}...")
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                print(f"[!] Câmara {camera_id} não disponível, voltando para câmara 0.")
                camera_id = 0
                cap = cv2.VideoCapture(camera_id)

    cap.release()
    cv2.destroyAllWindows()
    print("[+] Encerrado com sucesso!")


if __name__ == '__main__':
    main()
