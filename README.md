# 🐱 Cat Meme Recognition

Aplicação em Python que utiliza visão computacional (OpenCV + MediaPipe) para reconhecer a posição das mãos pela câmara e exibir memes de gatos em tempo real correspondentes a cada gesto!

---

## ✋ Gestos e Memes Correspondentes

| Gesto da Mão | Meme Exibido | Descrição |
| :--- | :---: | :--- |
| **Mão neutra ou sem mãos na câmara** | **Gato Parado** (`neutral.jpg`) | O gato laranja a olhar fixamente com orelhas baixas. |
| **Indicador levantado para cima (☝️)** | **Erm Ackshuallay** (`erhm_actually.jpg`) | O gato com óculos e dentinhos de esquilo apontando para cima. |
| **Punho fechado virado para a câmara (👊)** | **Murro do Gato** (`punch.jpg`) | O gato a desferir um soco na direção do ecrã. |
| **Duas mãos com indicadores a apontar um para o outro (👉👈)** | **Gato Corado / Tímido** (`shy.jpg`) | O gato laranja com bochechas coradas e dedos indicadores a tocarem-se. |

---

## 🚀 Como Executar

### Opção 1: Execução direta com o script facilitador
```bash
./run.sh
```

### Opção 2: Utilizando o ambiente virtual
```bash
# Ativar o ambiente virtual e executar
source .venv/bin/activate
python3 main.py
```

---

## ⌨️ Controlos de Teclado Durante a Execução

- **`m`**: Alterna entre a visualização **Split** (Câmara com esqueleto da mão ao lado do Meme) e **Meme em Ecrã Completo**.
- **`p`**: Liga ou desliga a miniatura da câmara (PiP - *Picture-in-Picture*) no modo de ecrã completo.
- **`c`**: Alterna o índice da câmara (caso possua mais do que uma webcam conectada).
- **`q`** ou **`ESC`**: Fecha a aplicação.

---

## 🛠️ Tecnologias Utilizadas

- **Python 3**
- **MediaPipe Tasks Vision** (`HandLandmarker`) para extração precisa dos 21 landmarks tridimensionais de cada mão.
- **OpenCV** para captura e processamento do fluxo de vídeo em tempo real.
- **Gesture Smoothing Buffer** para garantir transições suaves entre memes sem cintilação.
