import cv2

for i in range(6):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Índice {i}: no disponible")
        cap.release()
        continue

    print(f"Índice {i}: abierta. Pulsa Q para pasar a la siguiente.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.putText(frame, f"INDICE {i}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.imshow("Test camara", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()