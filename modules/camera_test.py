import cv2
import time

CAMERA_INDEX = 0

WIDTH = 1920
HEIGHT = 1080
FPS = 30

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720


def get_codec(cap):
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    return "".join(
        chr((fourcc >> (8 * i)) & 0xFF)
        for i in range(4)
    )


def main():

    # Windows: DirectShow
    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    if not cap.isOpened():
        print("FEHLER: Kamera konnte nicht geöffnet werden.")
        return

    # MJPEG zuerst setzen
    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        HEIGHT
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        FPS
    )

    # Kamera kurz anlaufen lassen
    for _ in range(10):
        cap.read()

    actual_width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    actual_height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    camera_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    codec = get_codec(cap)

    print("Kamera geöffnet")
    print()
    print("Width :", actual_width)
    print("Height:", actual_height)
    print("FPS   :", camera_fps)
    print("Codec :", codec)

    # FPS-Messung
    frame_counter = 0
    fps_start = time.perf_counter()
    real_fps = 0.0

    while True:

        success, frame = cap.read()

        if not success:
            print("Framefehler")
            break

        # =========================
        # reale FPS bestimmen
        # =========================

        frame_counter += 1

        now = time.perf_counter()
        elapsed = now - fps_start

        if elapsed >= 1.0:

            real_fps = frame_counter / elapsed

            frame_counter = 0
            fps_start = now

        # =========================
        # Informationen anzeigen
        # =========================

        cv2.putText(
            frame,
            f"Camera: {actual_width} x {actual_height}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"FPS: {real_fps:.1f}",
            (30, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Codec: {codec}",
            (30, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # =========================
        # Fadenkreuz
        # =========================

        center_x = actual_width // 2
        center_y = actual_height // 2

        cv2.line(
            frame,
            (center_x - 30, center_y),
            (center_x + 30, center_y),
            (0, 0, 255),
            2
        )

        cv2.line(
            frame,
            (center_x, center_y - 30),
            (center_x, center_y + 30),
            (0, 0, 255),
            2
        )

        # =========================
        # Anzeige verkleinern
        # =========================

        display = cv2.resize(
            frame,
            (DISPLAY_WIDTH, DISPLAY_HEIGHT)
        )

        cv2.imshow(
            "3D Scanner - Camera Test",
            display
        )

        # =========================
        # Tastatur
        # =========================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

        # S = Bild speichern
        if key == ord("s"):

            filename = (
                "camera_test_"
                + time.strftime("%Y%m%d_%H%M%S")
                + ".png"
            )

            cv2.imwrite(
                filename,
                frame
            )

            print("Bild gespeichert:", filename)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()