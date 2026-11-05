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
}

MEME_TITLES = {
    'neutral': "Gato Parado (Neutro)",
    'erhm_actually': "Erm Ackshuallay! (☝️)",
    'punch': "Murro do Gato! (👊)",
    'shy': "Gato Corado (👉👈)",
}

MEME_COLORS = {
    'neutral': (180, 180, 180),       # Cinzento claro
    'erhm_actually': (0, 215, 255),    # Amarelo/Dourado
    'punch': (50, 50, 255),           # Vermelho vivo
    'shy': (200, 130, 255),           # Rosa / Blush
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

    view_mode = "split"   # "split" ou "meme_only"
    show_pip = True       # Exibe miniatura da webcam no modo meme_only

    prev_time = time.time()
    fps = 0.0

    print("\n" + "="*50)
    print(" 🐱 CAT MEME RECOGNITION INICIADO! 🐱 ")
    print("="*50)
    print("Gestos suportados:")
    print("  • Mão neutra ou sem mãos   -> Gato Parado")
    print("  • Indicador para cima (☝️)  -> Erm Ackshuallay")
    print("  • Punho virado p/ câmara (👊) -> Gato Murro")
    print("  • Indicadores a apontar (👉👈) -> Gato Corado")
    print("\nTeclas de controlo:")
    print("  'm': Alternar vista (Lado a Lado / Só Meme)")
    print("  'p': Ligar/desligar miniatura (PiP) no modo meme")
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

        # Desenha esqueleto das mãos no frame da câmara
        for landmarks in hands_landmarks:
            gd.draw_hand_landmarks(frame, landmarks)

        # Classifica e suaviza o gesto detectado
        raw_gesture = gd.classify_gesture(hands_landmarks)
        current_gesture = smoother.update(raw_gesture)

        # Calcula FPS
        cur_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(cur_time - prev_time, 1e-4))
        prev_time = cur_time

        # Pega a imagem do meme atual
        meme_img = meme_images.get(current_gesture, meme_images['neutral'])
        gesture_title = MEME_TITLES.get(current_gesture, "Neutro")
        gesture_color = MEME_COLORS.get(current_gesture, (255, 255, 255))

        # Adiciona overlay de status no frame da câmara
        # Barra superior com fundo escuro semi-transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Informações de texto
        cv2.putText(frame, f"Gesto: {gesture_title}", (15, 38),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, gesture_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {int(fps)}", (w - 110, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

        # Barra inferior com instruções
        cv2.rectangle(frame, (0, h - 30), (w, h), (15, 15, 15), -1)
        cv2.putText(frame, "[m] Modo Vista | [p] PiP | [c] Cam | [q] Sair", (15, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # Renderização conforme o modo de visualização
        target_display_h = 600
        target_display_w = 640

        if view_mode == "split":
            cam_display = fit_image_to_box(frame, target_display_w, target_display_h)
            meme_display = fit_image_to_box(meme_img, target_display_w, target_display_h)

            # Borda decorativa colorida no meme atual
            cv2.rectangle(meme_display, (4, 4), (target_display_w - 4, target_display_h - 4), gesture_color, 4)

            # Título sobreposto no meme
            cv2.rectangle(meme_display, (0, target_display_h - 45), (target_display_w, target_display_h), (20, 20, 20), -1)
            cv2.putText(meme_display, gesture_title, (20, target_display_h - 15),
                        cv2.FONT_HERSHEY_DUPLEX, 0.85, gesture_color, 2, cv2.LINE_AA)

            final_screen = np.hstack([cam_display, meme_display])

        else:
            # Modo apenas meme com imagem em destaque
            full_w = 900
            full_h = 700
            meme_display = fit_image_to_box(meme_img, full_w, full_h)
            cv2.rectangle(meme_display, (4, 4), (full_w - 4, full_h - 4), gesture_color, 4)

            # Legenda inferior
            cv2.rectangle(meme_display, (0, full_h - 55), (full_w, full_h), (20, 20, 20), -1)
            cv2.putText(meme_display, gesture_title, (30, full_h - 18),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, gesture_color, 2, cv2.LINE_AA)

            # Picture-in-Picture (PiP) da câmara no canto superior direito
            if show_pip:
                pip_w = 220
                pip_h = 165
                small_cam = cv2.resize(frame, (pip_w, pip_h), interpolation=cv2.INTER_AREA)
                # Adiciona borda no PiP
                cv2.rectangle(small_cam, (0, 0), (pip_w - 1, pip_h - 1), gesture_color, 2)
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
