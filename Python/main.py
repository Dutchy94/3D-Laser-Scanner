import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODULES = ROOT / "modules"

class ScannerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Laser Scanner")
        self.geometry("560x500")
        self.resizable(False, False)

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        outer = ttk.Frame(self, padding=20)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer,
            text="Laser-Triangulationsscanner",
            font=("Segoe UI", 18, "bold")
        ).pack(anchor="w")

        ttk.Label(
            outer,
            text="Webcam + Linienlaser + ChArUco",
            font=("Segoe UI", 10)
        ).pack(anchor="w", pady=(0, 18))

        self.camera_index = tk.IntVar(value=0)

        row = ttk.Frame(outer)
        row.pack(fill="x", pady=(0, 16))
        ttk.Label(row, text="Kamera-Index:").pack(side="left")
        ttk.Spinbox(
            row, from_=0, to=10, width=5,
            textvariable=self.camera_index
        ).pack(side="left", padx=8)

        self.add_button(
            outer,
            "1. ChArUco A4 erzeugen",
            lambda: self.run_module("charuco_board.py")
        )
        self.add_button(
            outer,
            "2. Webcam testen (1080p)",
            lambda: self.run_module("camera_test.py", str(self.camera_index.get()))
        )
        self.add_button(
            outer,
            "3. ChArUco live erkennen",
            lambda: self.run_module("charuco_live.py", str(self.camera_index.get()))
        )

        ttk.Separator(outer).pack(fill="x", pady=14)

        self.add_button(
            outer,
            "4. Kamerakalibrierung",
            lambda: self.run_module("camera_calibration.py", str(self.camera_index.get()))
        )
        self.add_button(
            outer,
            "5. Laser-Ebene kalibrieren",
            lambda: self.run_module("laser_calibration.py", str(self.camera_index.get()))
        )
        self.add_button(
            outer,
            "6. 3D Scan / Punktwolke",
            lambda: self.run_module("scan_live.py", str(self.camera_index.get()))
        )

        ttk.Separator(outer).pack(fill="x", pady=14)

        ttk.Button(
            outer,
            text="Output-Ordner öffnen",
            command=self.open_output
        ).pack(fill="x", ipady=5)

        ttk.Label(
            outer,
            text="ESC oder Q schließt die OpenCV-Fenster.",
            foreground="#666666"
        ).pack(anchor="w", pady=(14, 0))

    def add_button(self, parent, text, command):
        ttk.Button(parent, text=text, command=command).pack(
            fill="x", pady=4, ipady=7
        )

    def run_module(self, filename, *args):
        path = MODULES / filename
        if not path.exists():
            messagebox.showerror("Fehler", f"Datei fehlt:\n{path}")
            return

        try:
            subprocess.Popen(
                [sys.executable, str(path), *args],
                cwd=str(ROOT)
            )
        except Exception as e:
            messagebox.showerror("Fehler", str(e))

    def not_ready(self, name):
        messagebox.showinfo(
            name,
            "Der Menüpunkt ist vorbereitet.\n"
            "Dieses Modul bauen wir als nächsten Schritt ein."
        )

    def open_output(self):
        path = ROOT / "output"
        path.mkdir(exist_ok=True)

        try:
            if sys.platform.startswith("win"):
                import os
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as e:
            messagebox.showerror("Fehler", str(e))

if __name__ == "__main__":
    app = ScannerGUI()
    app.mainloop()
'''

files["modules/charuco_board.py"] = r'''
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "charuco"
OUT.mkdir(parents=True, exist_ok=True)

# ----- Physische Board-Geometrie -----
SQUARES_X = 6
SQUARES_Y = 9
SQUARE_MM = 28.0
MARKER_MM = 20.0

# A4 bei 300 dpi
DPI = 300
A4_W_MM = 210.0
A4_H_MM = 297.0

DICT_ID = cv2.aruco.DICT_4X4_50
DICT_NAME = "DICT_4X4_50"

def mm_to_px(mm):
    return int(round(mm / 25.4 * DPI))

def main():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "cv2.aruco fehlt. Installiere 'opencv-contrib-python'."
        )

    dictionary = cv2.aruco.getPredefinedDictionary(DICT_ID)

    # OpenCV nutzt hier beliebige konsistente Längeneinheiten.
    # Wir verwenden Millimeter.
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_MM,
        MARKER_MM,
        dictionary
    )

    board_w_mm = SQUARES_X * SQUARE_MM
    board_h_mm = SQUARES_Y * SQUARE_MM

    board_w_px = mm_to_px(board_w_mm)
    board_h_px = mm_to_px(board_h_mm)

    board_img = board.generateImage(
        (board_w_px, board_h_px),
        marginSize=0,
        borderBits=1
    )

    page_w = mm_to_px(A4_W_MM)
    page_h = mm_to_px(A4_H_MM)
    page = np.full((page_h, page_w), 255, dtype=np.uint8)

    x0 = (page_w - board_w_px) // 2
    # Etwas mehr Rand oben/unten vermeiden; zentriert reicht für A4.
    y0 = (page_h - board_h_px) // 2

    page[y0:y0 + board_h_px, x0:x0 + board_w_px] = board_img

    # Kontrollstrecke: 100 mm
    line_y = page_h - mm_to_px(10)
    line_x0 = mm_to_px(20)
    line_x1 = line_x0 + mm_to_px(100)
    cv2.line(page, (line_x0, line_y), (line_x1, line_y), 0, 4)
    cv2.line(page, (line_x0, line_y - 15), (line_x0, line_y + 15), 0, 4)
    cv2.line(page, (line_x1, line_y - 15), (line_x1, line_y + 15), 0, 4)

    png_path = OUT / "charuco_A4_6x9_28mm.png"
    pdf_path = OUT / "charuco_A4_6x9_28mm.pdf"
    json_path = OUT / "charuco_A4_6x9_28mm.json"

    cv2.imwrite(str(png_path), page)

    pil = Image.fromarray(page).convert("L")
    pil.save(
        pdf_path,
        "PDF",
        resolution=DPI
    )

    meta = {
        "dictionary": DICT_NAME,
        "squares_x": SQUARES_X,
        "squares_y": SQUARES_Y,
        "square_length_mm": SQUARE_MM,
        "marker_length_mm": MARKER_MM,
        "board_width_mm": board_w_mm,
        "board_height_mm": board_h_mm,
        "page": "A4",
        "dpi": DPI,
        "print_scale": "100%",
        "control_line_mm": 100.0
    }

    json_path.write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8"
    )

    print("ChArUco erstellt:")
    print(png_path)
    print(pdf_path)
    print(json_path)
    print()
    print("WICHTIG: PDF bei 100 % / Tatsächliche Größe drucken.")
    print("Die Kontrollstrecke unten muss nach dem Druck 100 mm messen.")

    # Vorschau
    preview = cv2.resize(page, (0, 0), fx=0.25, fy=0.25)
    cv2.imshow("ChArUco A4 - Vorschau", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
'''

files["modules/camera_test.py"] = r'''
import sys
import time
import cv2

def main():
    camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    # Unter Windows funktioniert DirectShow bei vielen USB-Webcams gut.
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"Kamera {camera_index} konnte nicht geöffnet werden.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Kamera: {camera_index}")
    print(f"Auflösung: {actual_w} x {actual_h}")
    print(f"Gemeldete FPS: {actual_fps:.1f}")

    last = time.perf_counter()
    frames = 0
    shown_fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frames += 1
        now = time.perf_counter()

        if now - last >= 1.0:
            shown_fps = frames / (now - last)
            frames = 0
            last = now

        text = f"{frame.shape[1]}x{frame.shape[0]}  FPS: {shown_fps:.1f}"
        cv2.putText(
            frame, text, (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            (0, 255, 0), 2, cv2.LINE_AA
        )

        # Nur für die Anzeige verkleinern, Aufnahme bleibt in voller Auflösung.
        display = cv2.resize(frame, (1280, 720))
        cv2.imshow("Webcam Test - ESC/Q zum Beenden", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
'''

files["modules/charuco_live.py"] = r'''
import sys
import json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
META_PATH = ROOT / "output" / "charuco" / "charuco_A4_6x9_28mm.json"

def load_board():
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    else:
        meta = {
            "dictionary": "DICT_4X4_50",
            "squares_x": 6,
            "squares_y": 9,
            "square_length_mm": 28.0,
            "marker_length_mm": 20.0,
        }

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    board = cv2.aruco.CharucoBoard(
        (int(meta["squares_x"]), int(meta["squares_y"])),
        float(meta["square_length_mm"]),
        float(meta["marker_length_mm"]),
        dictionary
    )

    return dictionary, board

def main():
    camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    dictionary, board = load_board()

    detector_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(
        dictionary,
        detector_params
    )

    # Moderne OpenCV-Versionen:
    charuco_detector = None
    if hasattr(cv2.aruco, "CharucoDetector"):
        try:
            charuco_detector = cv2.aruco.CharucoDetector(board)
        except Exception:
            charuco_detector = None

    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"Kamera {camera_index} konnte nicht geöffnet werden.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        charuco_corners = None
        charuco_ids = None
        marker_corners = None
        marker_ids = None

        if charuco_detector is not None:
            try:
                result = charuco_detector.detectBoard(gray)
                charuco_corners, charuco_ids, marker_corners, marker_ids = result
            except Exception:
                charuco_detector = None

        if charuco_detector is None:
            marker_corners, marker_ids, _ = aruco_detector.detectMarkers(gray)

            if marker_ids is not None and len(marker_ids) > 0:
                try:
                    count, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                        marker_corners,
                        marker_ids,
                        gray,
                        board
                    )
                except Exception:
                    charuco_corners = None
                    charuco_ids = None

        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)

        num_charuco = 0
        if charuco_ids is not None and len(charuco_ids) > 0:
            num_charuco = len(charuco_ids)
            cv2.aruco.drawDetectedCornersCharuco(
                frame,
                charuco_corners,
                charuco_ids,
                (0, 0, 255)
            )

        num_markers = 0 if marker_ids is None else len(marker_ids)

        cv2.putText(
            frame,
            f"ArUco: {num_markers}   ChArUco corners: {num_charuco}",
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        display = cv2.resize(frame, (1280, 720))
        cv2.imshow("ChArUco Live - ESC/Q zum Beenden", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
