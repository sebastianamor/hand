import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ====== CONFIGURA AQUI ======
CAMARA_INDICE = 1          # pon el numero de tu webcam
MODELO = "hand_landmarker.task"
# ============================

CONEXIONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# Colores en BGR
COLORES = {
    "PEACE": (0, 215, 255),
    "PUNO": (60, 60, 255),
    "MANO ABIERTA": (80, 220, 80),
    "SENALANDO": (255, 160, 40),
    "ROCK": (255, 60, 200),
    "LIKE": (60, 255, 255),
    "OK": (200, 255, 80),
    "...": (180, 180, 180),
}


def dist(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def dedos_levantados(hand):
    """Devuelve [pulgar, indice, medio, anular, menique] con True/False."""
    muneca = hand[0]
    estado = []

    # Pulgar: la punta debe estar mas lejos de la base del menique que la articulacion
    estado.append(dist(hand[4], hand[17]) > dist(hand[3], hand[17]))

    # Resto: la punta mas lejos de la muneca que la articulacion media
    for punta, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        estado.append(dist(hand[punta], muneca) > dist(hand[pip], muneca))
    return estado


def reconocer_gesto(hand):
    p, i, m, a, k = dedos_levantados(hand)
    total = sum([p, i, m, a, k])

    # OK: pulgar e indice juntos, los otros tres arriba
    if dist(hand[4], hand[8]) < 0.05 and m and a and k:
        return "OK", [p, i, m, a, k]
    if total == 0:
        return "PUNO", [p, i, m, a, k]
    if total == 5:
        return "MANO ABIERTA", [p, i, m, a, k]
    if i and m and not a and not k:
        return "PEACE", [p, i, m, a, k]
    if i and not m and not a and not k and not p:
        return "SENALANDO", [p, i, m, a, k]
    if i and k and not m and not a:
        return "ROCK", [p, i, m, a, k]
    if p and not i and not m and not a and not k and hand[4].y < hand[0].y:
        return "LIKE", [p, i, m, a, k]
    return "...", [p, i, m, a, k]


def dibujar_mano(frame, hand, color):
    h, w, _ = frame.shape
    puntos = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
    for a, b in CONEXIONES:
        cv2.line(frame, puntos[a], puntos[b], color, 2, cv2.LINE_AA)
    for idx, (x, y) in enumerate(puntos):
        radio = 6 if idx in (4, 8, 12, 16, 20) else 4
        cv2.circle(frame, (x, y), radio, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), radio, color, 1, cv2.LINE_AA)


def dibujar_panel(frame, x, y, titulo, gesto, dedos, color):
    ancho, alto = 270, 120
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + ancho, y + alto), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.rectangle(frame, (x, y), (x + 6, y + alto), color, -1)          # barra de color
    cv2.rectangle(frame, (x, y), (x + ancho, y + alto), color, 1)       # borde

    cv2.putText(frame, titulo, (x + 18, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, gesto, (x + 18, y + 65),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2, cv2.LINE_AA)

    # Indicadores de los 5 dedos
    nombres = ["P", "I", "M", "A", "K"]
    for n, (nombre, arriba) in enumerate(zip(nombres, dedos)):
        cx = x + 30 + n * 45
        cv2.circle(frame, (cx, y + 97), 11,
                   color if arriba else (80, 80, 80), -1, cv2.LINE_AA)
        cv2.putText(frame, nombre, (cx - 6, y + 102),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)


def main():
    opciones = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODELO),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    detector = vision.HandLandmarker.create_from_options(opciones)

    cap = cv2.VideoCapture(CAMARA_INDICE, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("No se pudo abrir la camara. Revisa CAMARA_INDICE.")
        return

    inicio = time.time()
    t_anterior = inicio

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts = int((time.time() - inicio) * 1000)
        resultado = detector.detect_for_video(mp_image, ts)

        for n, hand in enumerate(resultado.hand_landmarks):
            gesto, dedos = reconocer_gesto(hand)
            color = COLORES.get(gesto, (255, 255, 255))
            dibujar_mano(frame, hand, color)
            dibujar_panel(frame, 15, 15 + n * 135, f"Mano {n + 1}", gesto, dedos, color)

        # FPS
        ahora = time.time()
        fps = 1 / max(ahora - t_anterior, 1e-6)
        t_anterior = ahora
        cv2.putText(frame, f"FPS: {fps:.0f}", (frame.shape[1] - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "ESC para salir", (15, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("Gestos", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()