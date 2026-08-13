from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image

# =========================
# Einstellungen
# =========================

SQUARES_X = 6
SQUARES_Y = 9

SQUARE_MM = 28.0
MARKER_MM = 20.0

DPI = 300

A4_W_MM = 210.0
A4_H_MM = 297.0

DICT_ID = cv2.aruco.DICT_4X4_50
DICT_NAME = "DICT_4X4_50"


# =========================
# Hilfsfunktion
# =========================

def mm_to_px(mm):
    return int(round(mm / 25.4 * DPI))


# =========================
# Hauptprogramm
# =========================

def main():

    if not hasattr(cv2, "aruco"):
        print("FEHLER: cv2.aruco fehlt.")
        print("Installiere:")
        print("pip install opencv-contrib-python")
        return

    output_dir = Path("output/charuco")
    output_dir.mkdir(parents=True, exist_ok=True)

    dictionary = cv2.aruco.getPredefinedDictionary(DICT_ID)

    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_MM,
        MARKER_MM,
        dictionary
    )

    # =========================
    # Boardgröße
    # =========================

    board_w_mm = SQUARES_X * SQUARE_MM
    board_h_mm = SQUARES_Y * SQUARE_MM

    board_w_px = mm_to_px(board_w_mm)
    board_h_px = mm_to_px(board_h_mm)

    print("Boardgröße:")
    print(f"{board_w_mm:.1f} x {board_h_mm:.1f} mm")

    # =========================
    # ChArUco erzeugen
    # =========================

    board_img = board.generateImage(
        (board_w_px, board_h_px),
        marginSize=0,
        borderBits=1
    )

    # =========================
    # A4-Seite erzeugen
    # =========================

    page_w = mm_to_px(A4_W_MM)
    page_h = mm_to_px(A4_H_MM)

    page = np.full(
        (page_h, page_w),
        255,
        dtype=np.uint8
    )

    x0 = (page_w - board_w_px) // 2
    y0 = (page_h - board_h_px) // 2

    page[
        y0:y0 + board_h_px,
        x0:x0 + board_w_px
    ] = board_img

    # =========================
    # 100-mm Kontrolllinie
    # =========================

    line_y = page_h - mm_to_px(8)

    line_x0 = mm_to_px(20)
    line_x1 = line_x0 + mm_to_px(100)

    cv2.line(
        page,
        (line_x0, line_y),
        (line_x1, line_y),
        0,
        4
    )

    cv2.line(
        page,
        (line_x0, line_y - 15),
        (line_x0, line_y + 15),
        0,
        4
    )

    cv2.line(
        page,
        (line_x1, line_y - 15),
        (line_x1, line_y + 15),
        0,
        4
    )

    # =========================
    # Dateien speichern
    # =========================

    png_path = output_dir / "charuco_A4.png"
    pdf_path = output_dir / "charuco_A4.pdf"
    json_path = output_dir / "charuco_config.json"

    cv2.imwrite(
        str(png_path),
        page
    )

    Image.fromarray(page).convert("L").save(
        pdf_path,
        "PDF",
        resolution=DPI
    )

    # =========================
    # Konfiguration speichern
    # =========================

    config = {
        "dictionary": DICT_NAME,

        "squares_x": SQUARES_X,
        "squares_y": SQUARES_Y,

        "square_length_mm": SQUARE_MM,
        "marker_length_mm": MARKER_MM,

        "board_width_mm": board_w_mm,
        "board_height_mm": board_h_mm,

        "dpi": DPI
    }

    json_path.write_text(
        json.dumps(
            config,
            indent=4
        )
    )

    # =========================
    # Ausgabe
    # =========================

    print()
    print("ChArUco erzeugt:")
    print(png_path)
    print(pdf_path)
    print(json_path)

    print()
    print("WICHTIG:")
    print("PDF mit 100 % / Tatsächliche Größe drucken.")
    print("Nicht 'An Seite anpassen' verwenden.")
    print("Die Kontrolllinie muss exakt 100 mm messen.")


if __name__ == "__main__":
    main()