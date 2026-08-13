import cv2
import json
import time
from pathlib import Path


# =========================
# Einstellungen
# =========================

CAMERA_INDEX = 0

WIDTH = 1920
HEIGHT = 1080
FPS = 30

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

CONFIG_PATH = Path("output/charuco/charuco_config.json")

DEBUG = True


# =========================
# ChArUco Board laden
# =========================

def load_charuco_board():

    if not CONFIG_PATH.exists():
        print("FEHLER: Config nicht gefunden:")
        print(CONFIG_PATH)
        return None, None

    config = json.loads(
        CONFIG_PATH.read_text(encoding="utf-8")
    )

    print("ChArUco Config geladen:")
    print(config)

    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_4X4_50
    )

    board = cv2.aruco.CharucoBoard(
        (
            int(config["squares_x"]),
            int(config["squares_y"])
        ),
        float(config["square_length_mm"]),
        float(config["marker_length_mm"]),
        dictionary
    )

    return dictionary, board


# =========================
# Kamera öffnen
# =========================

def open_camera():

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    if not cap.isOpened():
        print("FEHLER: Kamera konnte nicht geöffnet werden.")
        return None

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

    # Kamera anlaufen lassen
    for _ in range(10):
        cap.read()

    actual_width = cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )

    actual_height = cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )

    actual_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    fourcc = int(
        cap.get(cv2.CAP_PROP_FOURCC)
    )

    codec = "".join(
        chr((fourcc >> (8 * i)) & 0xFF)
        for i in range(4)
    )

    print()
    print("Kamera geöffnet:")
    print("Width :", actual_width)
    print("Height:", actual_height)
    print("FPS   :", actual_fps)
    print("Codec :", codec)

    return cap


# =========================
# Hauptprogramm
# =========================

def main():

    print("OpenCV Version:")
    print(cv2.__version__)

    if not hasattr(cv2, "aruco"):
        print("FEHLER: cv2.aruco fehlt")
        print("opencv-contrib-python installieren")
        return

    dictionary, board = load_charuco_board()

    if board is None:
        return

    # =========================
    # Detector erstellen
    # =========================

    detector = cv2.aruco.CharucoDetector(
        board
    )

    print()
    print("CharucoDetector erstellt")

    cap = open_camera()

    if cap is None:
        return

    # =========================
    # FPS
    # =========================

    frame_counter = 0
    fps_start = time.perf_counter()
    real_fps = 0.0

    debug_counter = 0

    # =========================
    # Loop
    # =========================

    while True:

        success, frame = cap.read()

        if not success:
            print("Framefehler")
            break

        # =========================
        # FPS messen
        # =========================

        frame_counter += 1

        now = time.perf_counter()
        elapsed = now - fps_start

        if elapsed >= 1.0:

            real_fps = frame_counter / elapsed

            frame_counter = 0
            fps_start = now

        # =========================
        # Graustufen
        # =========================

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # =========================
        # ChArUco erkennen
        # =========================

        try:

            result = detector.detectBoard(gray)

        except Exception as e:

            print()
            print("FEHLER detectBoard:")
            print(e)

            continue

        # =========================
        # Debug Rückgabe
        # =========================

        if DEBUG and debug_counter % 60 == 0:

            print()
            print("detectBoard Rückgabe:")

            print("Type:", type(result))

            try:
                print("Anzahl Elemente:", len(result))
            except:
                pass

        debug_counter += 1

        # =========================
        # Ergebnisse entpacken
        # =========================

        try:

            (
                charuco_corners,
                charuco_ids,
                marker_corners,
                marker_ids
            ) = result

        except Exception as e:

            print("FEHLER beim Entpacken:")
            print(e)

            continue

        # =========================
        # Debug Shapes
        # =========================

        if DEBUG and debug_counter % 60 == 0:

            print()
            print("--- DEBUG ---")

            if charuco_corners is None:
                print("charuco_corners: None")
            else:
                print(
                    "charuco_corners:",
                    type(charuco_corners),
                    charuco_corners.shape
                )

            if charuco_ids is None:
                print("charuco_ids: None")
            else:
                print(
                    "charuco_ids:",
                    type(charuco_ids),
                    charuco_ids.shape
                )

            if marker_ids is None:
                print("marker_ids: None")
            else:
                print(
                    "marker_ids:",
                    type(marker_ids),
                    marker_ids.shape
                )

        # =========================
        # Marker zeichnen
        # =========================

        marker_count = 0

        if (
            marker_ids is not None
            and marker_corners is not None
        ):

            marker_count = len(marker_ids)

            try:

                cv2.aruco.drawDetectedMarkers(
                    frame,
                    marker_corners,
                    marker_ids
                )

            except Exception as e:

                print("Fehler drawDetectedMarkers:")
                print(e)

        # =========================
        # ChArUco Corners prüfen
        # =========================

        corner_count = 0

        if (
            charuco_corners is not None
            and charuco_ids is not None
        ):

            n_corners = len(charuco_corners)
            n_ids = len(charuco_ids)

            # =========================
            # Debug Inkonsistenz
            # =========================

            if n_corners != n_ids:

                print()
                print("WARNUNG:")
                print(
                    "ChArUco Corners:",
                    n_corners
                )

                print(
                    "ChArUco IDs:",
                    n_ids
                )

                print(
                    "Shape Corners:",
                    charuco_corners.shape
                )

                print(
                    "Shape IDs:",
                    charuco_ids.shape
                )

            # =========================
            # fürs Zeichnen kürzen
            # =========================

            n = min(
                n_corners,
                n_ids
            )

            if n > 0:

                corners_draw = (
                    charuco_corners[:n]
                )

                ids_draw = (
                    charuco_ids[:n]
                )

                corner_count = n

                try:

                    cv2.aruco.drawDetectedCornersCharuco(
                        frame,
                        corners_draw,
                        ids_draw,
                        (0, 0, 255)
                    )

                except Exception as e:

                    print()
                    print(
                        "Fehler drawDetectedCornersCharuco:"
                    )

                    print(e)

                    print(
                        "Corners Shape:",
                        corners_draw.shape
                    )

                    print(
                        "IDs Shape:",
                        ids_draw.shape
                    )

        # =========================
        # Statusanzeige
        # =========================

        cv2.putText(
            frame,
            f"FPS: {real_fps:.1f}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"ArUco Marker: {marker_count}",
            (30, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"ChArUco Corners: {corner_count}",
            (30, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # =========================
        # Statusfarbe
        # =========================

        if corner_count >= 30:

            text = "ERKENNUNG: GUT"
            color = (0, 255, 0)

        elif corner_count >= 15:

            text = "ERKENNUNG: MITTEL"
            color = (0, 255, 255)

        else:

            text = "ERKENNUNG: SCHWACH"
            color = (0, 0, 255)

        cv2.putText(
            frame,
            text,
            (30, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
            cv2.LINE_AA
        )

        # =========================
        # Fadenkreuz
        # =========================

        center_x = frame.shape[1] // 2
        center_y = frame.shape[0] // 2

        cv2.line(
            frame,
            (center_x - 20, center_y),
            (center_x + 20, center_y),
            (255, 0, 0),
            2
        )

        cv2.line(
            frame,
            (center_x, center_y - 20),
            (center_x, center_y + 20),
            (255, 0, 0),
            2
        )

        # =========================
        # Anzeige verkleinern
        # =========================

        display = cv2.resize(
            frame,
            (
                DISPLAY_WIDTH,
                DISPLAY_HEIGHT
            )
        )

        cv2.imshow(
            "3D Scanner - ChArUco Live",
            display
        )

        # =========================
        # Tastatur
        # =========================

        key = cv2.waitKey(1) & 0xFF

        # Q oder ESC
        if (
            key == ord("q")
            or key == 27
        ):
            break

        # =========================
        # Screenshot
        # =========================

        if key == ord("s"):

            filename = (
                "charuco_test_"
                + time.strftime(
                    "%Y%m%d_%H%M%S"
                )
                + ".png"
            )

            cv2.imwrite(
                filename,
                frame
            )

            print()
            print(
                "Screenshot gespeichert:",
                filename
            )

    # =========================
    # Ende
    # =========================

    cap.release()

    cv2.destroyAllWindows()

    print()
    print("Programm beendet")


if __name__ == "__main__":
    main()