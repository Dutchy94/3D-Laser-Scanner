import cv2
import json
import math
import time
import numpy as np
from pathlib import Path


# ==========================================================
# EINSTELLUNGEN
# ==========================================================

CAMERA_INDEX = 0

WIDTH = 1920
HEIGHT = 1080
FPS = 30

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

MIN_CORNERS = 18

STABLE_FRAMES_REQUIRED = 8

ANGLE_TOLERANCE_DEG = 10.0
DISTANCE_TOLERANCE_MM = 100.0
POSITION_TOLERANCE_NORM = 0.2

CAPTURE_COOLDOWN_S = 0.65

AUTO_CALIBRATE_WHEN_DONE = True
AUTO_REJECT_BAD_IMAGES = True
MIN_IMAGES_AFTER_REJECTION = 20

CONFIG_PATH = Path("output/charuco/charuco_config.json")

OUTPUT_DIR = Path("output/calibration")
IMAGE_DIR = OUTPUT_DIR / "camera_images_auto"

CALIBRATION_FILE = OUTPUT_DIR / "camera_calibration.json"


# ==========================================================
# ZIELPOSEN
#
# azimuth:
#   - = von links
#   + = von rechts
#
# elevation:
#   - = von oben
#   + = von unten
#
# image_x / image_y:
#   gewünschte Lage des Board-Mittelpunkts im Bild
# ==========================================================

TARGETS = [

    # ------------------------------------------------------
    # FRONTAL - BILDFLÄCHE ABDECKEN
    # ------------------------------------------------------

    {
        "text": "FRONTAL - BOARD MITTIG",
        "azimuth_deg": 0,
        "elevation_deg": 0,
        "distance_mm": 400,
        "image_x": 0.50,
        "image_y": 0.50
    },

    {
        "text": "FRONTAL - BOARD OBEN LINKS",
        "azimuth_deg": 0,
        "elevation_deg": 0,
        "distance_mm": 400,
        "image_x": 0.28,
        "image_y": 0.28
    },

    {
        "text": "FRONTAL - BOARD OBEN RECHTS",
        "azimuth_deg": 0,
        "elevation_deg": 0,
        "distance_mm": 400,
        "image_x": 0.72,
        "image_y": 0.28
    },

    {
        "text": "FRONTAL - BOARD UNTEN LINKS",
        "azimuth_deg": 0,
        "elevation_deg": 0,
        "distance_mm": 400,
        "image_x": 0.28,
        "image_y": 0.72
    },

    {
        "text": "FRONTAL - BOARD UNTEN RECHTS",
        "azimuth_deg": 0,
        "elevation_deg": 0,
        "distance_mm": 400,
        "image_x": 0.72,
        "image_y": 0.72
    },


    # ------------------------------------------------------
    # LINKS / RECHTS
    # ------------------------------------------------------

    {
        "text": "CA. 15 GRAD VON LINKS",
        "azimuth_deg": -15,
        "elevation_deg": 0,
        "distance_mm": 400,
        "image_x": 0.50,
        "image_y": 0.50
    },

    {
        "text": "CA. 25 GRAD VON LINKS",
        "azimuth_deg": -25,
        "elevation_deg": 0,
        "distance_mm": 400,
        "image_x": 0.50,
        "image_y": 0.50
    },

    {
        "text": "CA. 15 GRAD VON RECHTS",
        "azimuth_deg": 15,
        "elevation_deg": 0,
        "distance_mm": 400,
        "image_x": 0.50,
        "image_y": 0.50
    },

    {
        "text": "CA. 25 GRAD VON RECHTS",
        "azimuth_deg": 25,
        "elevation_deg": 0,
        "distance_mm": 400,
        "image_x": 0.50,
        "image_y": 0.50
    },


    # ------------------------------------------------------
    # OBEN / UNTEN
    # ------------------------------------------------------

    {
        "text": "CA. 15 GRAD VON OBEN",
        "azimuth_deg": 0,
        "elevation_deg": -15,
        "distance_mm": 400,
        "image_x": 0.50,
        "image_y": 0.50
    },

    {
        "text": "CA. 25 GRAD VON OBEN",
        "azimuth_deg": 0,
        "elevation_deg": -25,
        "distance_mm": 400,
        "image_x": 0.50,
        "image_y": 0.50
    },

    {
        "text": "CA. 15 GRAD VON UNTEN",
        "azimuth_deg": 0,
        "elevation_deg": 15,
        "distance_mm": 400,
        "image_x": 0.50,
        "image_y": 0.50
    },

    {
        "text": "CA. 25 GRAD VON UNTEN",
        "azimuth_deg": 0,
        "elevation_deg": 25,
        "distance_mm": 400,
        "image_x": 0.50,
        "image_y": 0.50
    },


    # ------------------------------------------------------
    # DIAGONAL
    # ------------------------------------------------------

    {
        "text": "20 GRAD LINKS + 15 GRAD OBEN",
        "azimuth_deg": -20,
        "elevation_deg": -15,
        "distance_mm": 400,
        "image_x": 0.50,
        "image_y": 0.50
    },

    {
        "text": "20 GRAD RECHTS + 15 GRAD OBEN",
        "azimuth_deg": 20,
        "elevation_deg": -15,
        "distance_mm": 400,
        "image_x": 0.50,
        "image_y": 0.50
    },

    {
        "text": "20 GRAD LINKS + 15 GRAD UNTEN",
        "azimuth_deg": -20,
        "elevation_deg": 15,
        "distance_mm": 400,
        "image_x": 0.50,
        "image_y": 0.50
    },

    {
        "text": "20 GRAD RECHTS + 15 GRAD UNTEN",
        "azimuth_deg": 20,
        "elevation_deg": 15,
        "distance_mm": 400,
        "image_x": 0.50,
        "image_y": 0.50
    },


    # ------------------------------------------------------
    # NAH
    # ------------------------------------------------------

    {
        "text": "NAH ~300 MM - FRONTAL",
        "azimuth_deg": 0,
        "elevation_deg": 0,
        "distance_mm": 300,
        "image_x": 0.50,
        "image_y": 0.50
    },

    {
        "text": "NAH ~300 MM - 20 GRAD VON LINKS",
        "azimuth_deg": -20,
        "elevation_deg": 0,
        "distance_mm": 300,
        "image_x": 0.50,
        "image_y": 0.50
    },

    {
        "text": "NAH ~300 MM - 20 GRAD VON RECHTS",
        "azimuth_deg": 20,
        "elevation_deg": 0,
        "distance_mm": 300,
        "image_x": 0.50,
        "image_y": 0.50
    },

    {
        "text": "NAH ~300 MM - 18 GRAD VON OBEN",
        "azimuth_deg": 0,
        "elevation_deg": -18,
        "distance_mm": 300,
        "image_x": 0.50,
        "image_y": 0.50
    },

    {
        "text": "NAH ~300 MM - 18 GRAD VON UNTEN",
        "azimuth_deg": 0,
        "elevation_deg": 18,
        "distance_mm": 300,
        "image_x": 0.50,
        "image_y": 0.50
    },


    # ------------------------------------------------------
    # WEIT
    # ------------------------------------------------------

    {
        "text": "WEIT ~500 MM - FRONTAL",
        "azimuth_deg": 0,
        "elevation_deg": 0,
        "distance_mm": 500,
        "image_x": 0.50,
        "image_y": 0.50
    },

    {
        "text": "WEIT ~500 MM - 20 GRAD VON LINKS",
        "azimuth_deg": -20,
        "elevation_deg": 0,
        "distance_mm": 500,
        "image_x": 0.50,
        "image_y": 0.50
    },

    {
        "text": "WEIT ~500 MM - 20 GRAD VON RECHTS",
        "azimuth_deg": 20,
        "elevation_deg": 0,
        "distance_mm": 500,
        "image_x": 0.50,
        "image_y": 0.50
    },

    {
        "text": "WEIT ~500 MM - 18 GRAD VON OBEN",
        "azimuth_deg": 0,
        "elevation_deg": -18,
        "distance_mm": 500,
        "image_x": 0.50,
        "image_y": 0.50
    },

    {
        "text": "WEIT ~500 MM - 18 GRAD VON UNTEN",
        "azimuth_deg": 0,
        "elevation_deg": 18,
        "distance_mm": 500,
        "image_x": 0.50,
        "image_y": 0.50
    },


    # ------------------------------------------------------
    # RANDABDECKUNG MIT LEICHTER NEIGUNG
    # ------------------------------------------------------

    {
        "text": "BOARD LINKS + LEICHT VON RECHTS",
        "azimuth_deg": 12,
        "elevation_deg": 0,
        "distance_mm": 400,
        "image_x": 0.27,
        "image_y": 0.50
    },

    {
        "text": "BOARD RECHTS + LEICHT VON LINKS",
        "azimuth_deg": -12,
        "elevation_deg": 0,
        "distance_mm": 400,
        "image_x": 0.73,
        "image_y": 0.50
    },

    {
        "text": "BOARD OBEN + LEICHT VON UNTEN",
        "azimuth_deg": 0,
        "elevation_deg": 12,
        "distance_mm": 400,
        "image_x": 0.50,
        "image_y": 0.27
    },

    {
        "text": "BOARD UNTEN + LEICHT VON OBEN",
        "azimuth_deg": 0,
        "elevation_deg": -12,
        "distance_mm": 400,
        "image_x": 0.50,
        "image_y": 0.73
    }
]


# ==========================================================
# BOARD LADEN
# ==========================================================

def load_charuco_board():

    if not CONFIG_PATH.exists():

        print("FEHLER:")
        print("Config nicht gefunden:")
        print(CONFIG_PATH)

        return None, None, None


    config = json.loads(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )


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


    return dictionary, board, config


# ==========================================================
# KAMERA
# ==========================================================

def open_camera():

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )


    if not cap.isOpened():

        print("Kamera konnte nicht geöffnet werden.")

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


    for _ in range(10):

        cap.read()


    print()
    print(
        "Kamera:",
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "x",
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "@",
        cap.get(cv2.CAP_PROP_FPS),
        "FPS"
    )


    return cap


# ==========================================================
# ALTE KALIBRIERUNG NUR ALS POSE-HILFE
# ==========================================================

def load_guide_camera():

    if CALIBRATION_FILE.exists():

        try:

            data = json.loads(
                CALIBRATION_FILE.read_text(
                    encoding="utf-8"
                )
            )


            K = np.asarray(
                data["camera_matrix"],
                dtype=np.float64
            )


            dist = np.asarray(
                data["dist_coeffs"],
                dtype=np.float64
            )


            print(
                "Vorhandene Kalibrierung wird nur "
                "zur Benutzerführung verwendet."
            )


            return K, dist

        except Exception:

            pass


    # Fallback nur für grobe Pose-Führung

    fx = WIDTH * 0.78
    fy = WIDTH * 0.78

    cx = WIDTH / 2.0
    cy = HEIGHT / 2.0


    K = np.array(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ],
        dtype=np.float64
    )


    dist = np.zeros(
        (1, 5),
        dtype=np.float64
    )


    return K, dist


# ==========================================================
# DETECTION PRÜFEN
# ==========================================================

def clean_detection(
    corners,
    ids
):

    if (
        corners is None
        or
        ids is None
    ):

        return None, None


    if len(corners) != len(ids):

        return None, None


    if len(corners) == 0:

        return None, None


    return (
        corners.copy(),
        ids.copy()
    )


# ==========================================================
# GUIDE POSE
# ==========================================================

def estimate_guide_pose(
    board,
    corners,
    ids,
    camera_matrix,
    dist_coeffs
):

    if (
        corners is None
        or
        ids is None
        or
        len(ids) < MIN_CORNERS
    ):

        return None


    try:

        obj_points, img_points = (
            board.matchImagePoints(
                corners,
                ids
            )
        )

    except Exception:

        return None


    if (
        obj_points is None
        or
        img_points is None
    ):

        return None


    obj_points = np.asarray(
        obj_points,
        dtype=np.float64
    ).reshape(-1, 3)


    img_points = np.asarray(
        img_points,
        dtype=np.float64
    ).reshape(-1, 2)


    if len(obj_points) < 6:

        return None


    try:

        success, rvec, tvec = cv2.solvePnP(
            obj_points,
            img_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

    except cv2.error:

        return None


    if not success:

        return None


    R, _ = cv2.Rodrigues(
        rvec
    )


    camera_position = (
        -R.T
        @ tvec.reshape(3)
    )


    min_xyz = np.min(
        obj_points,
        axis=0
    )

    max_xyz = np.max(
        obj_points,
        axis=0
    )


    board_center = (
        min_xyz + max_xyz
    ) / 2.0


    delta = (
        camera_position
        - board_center
    )


    depth = abs(
        delta[2]
    )


    if depth < 1:

        return None


    azimuth_deg = math.degrees(
        math.atan2(
            delta[0],
            depth
        )
    )


    elevation_deg = math.degrees(
        math.atan2(
            delta[1],
            depth
        )
    )


    distance_mm = float(
        np.linalg.norm(
            delta
        )
    )


    image_center = np.mean(
        img_points,
        axis=0
    )


    image_x = float(
        image_center[0]
        / WIDTH
    )


    image_y = float(
        image_center[1]
        / HEIGHT
    )


    return {
        "azimuth_deg":
            float(azimuth_deg),

        "elevation_deg":
            float(elevation_deg),

        "distance_mm":
            distance_mm,

        "image_x":
            image_x,

        "image_y":
            image_y
    }


# ==========================================================
# PRÜFEN OB POSE PASST
# ==========================================================

def pose_matches_target(
    pose,
    target
):

    if pose is None:

        return False


    return (

        abs(
            pose["azimuth_deg"]
            - target["azimuth_deg"]
        )
        <= ANGLE_TOLERANCE_DEG

        and

        abs(
            pose["elevation_deg"]
            - target["elevation_deg"]
        )
        <= ANGLE_TOLERANCE_DEG

        and

        abs(
            pose["distance_mm"]
            - target["distance_mm"]
        )
        <= DISTANCE_TOLERANCE_MM

        and

        abs(
            pose["image_x"]
            - target["image_x"]
        )
        <= POSITION_TOLERANCE_NORM

        and

        abs(
            pose["image_y"]
            - target["image_y"]
        )
        <= POSITION_TOLERANCE_NORM
    )


# ==========================================================
# SOLL-POSITION IM BILD ZEICHNEN
# ==========================================================

def draw_target_board_position(
    frame,
    target
):

    h, w = frame.shape[:2]


    cx = int(
        target["image_x"]
        * w
    )

    cy = int(
        target["image_y"]
        * h
    )


    distance = float(
        target["distance_mm"]
    )


    # grobe visuelle Größe
    # bei 400 mm ca. 47 % der Bildhöhe

    scale = (
        400.0
        / max(
            distance,
            1.0
        )
    )


    box_h = int(
        h
        * 0.47
        * scale
    )


    # Board ist 168 x 252 mm
    aspect = (
        168.0
        / 252.0
    )


    box_w = int(
        box_h
        * aspect
    )


    box_h = int(
        np.clip(
            box_h,
            230,
            h * 0.78
        )
    )


    box_w = int(
        np.clip(
            box_w,
            150,
            w * 0.55
        )
    )


    x1 = int(
        cx - box_w / 2
    )

    y1 = int(
        cy - box_h / 2
    )

    x2 = int(
        cx + box_w / 2
    )

    y2 = int(
        cy + box_h / 2
    )


    x1 = max(
        5,
        x1
    )

    y1 = max(
        5,
        y1
    )

    x2 = min(
        w - 5,
        x2
    )

    y2 = min(
        h - 5,
        y2
    )


    # ------------------------------------------------------
    # dunkle transparente Fläche
    # ------------------------------------------------------

    overlay = frame.copy()


    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        (0, 255, 255),
        -1
    )


    frame[:] = cv2.addWeighted(
        overlay,
        0.12,
        frame,
        0.88,
        0
    )


    # ------------------------------------------------------
    # dicke gelbe Umrandung
    # ------------------------------------------------------

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 255),
        5
    )


    # ------------------------------------------------------
    # Ecken extra markieren
    # ------------------------------------------------------

    corner_len = 40


    # oben links
    cv2.line(
        frame,
        (x1, y1),
        (x1 + corner_len, y1),
        (0, 255, 255),
        9
    )

    cv2.line(
        frame,
        (x1, y1),
        (x1, y1 + corner_len),
        (0, 255, 255),
        9
    )


    # oben rechts
    cv2.line(
        frame,
        (x2, y1),
        (x2 - corner_len, y1),
        (0, 255, 255),
        9
    )

    cv2.line(
        frame,
        (x2, y1),
        (x2, y1 + corner_len),
        (0, 255, 255),
        9
    )


    # unten links
    cv2.line(
        frame,
        (x1, y2),
        (x1 + corner_len, y2),
        (0, 255, 255),
        9
    )

    cv2.line(
        frame,
        (x1, y2),
        (x1, y2 - corner_len),
        (0, 255, 255),
        9
    )


    # unten rechts
    cv2.line(
        frame,
        (x2, y2),
        (x2 - corner_len, y2),
        (0, 255, 255),
        9
    )

    cv2.line(
        frame,
        (x2, y2),
        (x2, y2 - corner_len),
        (0, 255, 255),
        9
    )


    # ------------------------------------------------------
    # großes Fadenkreuz
    # ------------------------------------------------------

    cv2.line(
        frame,
        (cx - 45, cy),
        (cx + 45, cy),
        (0, 255, 255),
        4
    )


    cv2.line(
        frame,
        (cx, cy - 45),
        (cx, cy + 45),
        (0, 255, 255),
        4
    )


    cv2.circle(
        frame,
        (cx, cy),
        10,
        (0, 255, 255),
        3
    )


    # ------------------------------------------------------
    # Label
    # ------------------------------------------------------

    label_y = max(
        35,
        y1 - 15
    )


    cv2.putText(
        frame,
        "BOARD HIER POSITIONIEREN",
        (
            x1,
            label_y
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 255),
        3,
        cv2.LINE_AA
    )


# ==========================================================
# ZENTRIERTER TEXT
# ==========================================================

def draw_centered_text(
    image,
    text,
    y,
    scale,
    color,
    thickness
):

    font = cv2.FONT_HERSHEY_SIMPLEX


    size, _ = cv2.getTextSize(
        text,
        font,
        scale,
        thickness
    )


    x = int(
        (
            image.shape[1]
            - size[0]
        )
        / 2
    )


    cv2.putText(
        image,
        text,
        (x, y),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


# ==========================================================
# BILD SPEICHERN
# ==========================================================

def store_calibration_image(
    raw_frame,
    corners,
    ids,
    target,
    all_corners,
    all_ids,
    captured_targets
):

    all_corners.append(
        corners.copy()
    )


    all_ids.append(
        ids.copy()
    )


    captured_targets.append(
        target.copy()
    )


    number = len(
        all_corners
    )


    filename = (
        IMAGE_DIR
        / f"calibration_{number:02d}.png"
    )


    cv2.imwrite(
        str(filename),
        raw_frame
    )


    print(
        f"Bild {number:02d} gespeichert: "
        f"{target['text']}"
    )


# ==========================================================
# KALIBRIERUNG
# ==========================================================

def run_calibration(
    board,
    all_corners,
    all_ids,
    image_size,
    selected_indices
):

    object_points = []
    image_points = []
    source_indices = []


    for source_index in selected_indices:

        try:

            obj_points, img_points = (
                board.matchImagePoints(
                    all_corners[source_index],
                    all_ids[source_index]
                )
            )

        except Exception:

            continue


        if (
            obj_points is None
            or
            img_points is None
            or
            len(obj_points) < 4
        ):

            continue


        object_points.append(
            np.asarray(
                obj_points,
                dtype=np.float32
            )
        )


        image_points.append(
            np.asarray(
                img_points,
                dtype=np.float32
            )
        )


        source_indices.append(
            source_index
        )


    if len(
        object_points
    ) < 10:

        return None


    try:

        (
            rms,
            camera_matrix,
            dist_coeffs,
            rvecs,
            tvecs
        ) = cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            None,
            None
        )

    except cv2.error:

        return None


    per_image_errors = []


    for i in range(
        len(object_points)
    ):

        projected, _ = cv2.projectPoints(
            object_points[i],
            rvecs[i],
            tvecs[i],
            camera_matrix,
            dist_coeffs
        )


        measured = image_points[
            i
        ].reshape(-1, 2)


        projected = projected.reshape(
            -1,
            2
        )


        error = np.linalg.norm(
            measured - projected,
            axis=1
        )


        per_image_errors.append(
            float(
                np.mean(
                    error
                )
            )
        )


    return {

        "rms":
            float(rms),

        "camera_matrix":
            camera_matrix,

        "dist_coeffs":
            dist_coeffs,

        "per_image_errors":
            per_image_errors,

        "source_indices":
            source_indices
    }


# ==========================================================
# FINALE KALIBRIERUNG
# ==========================================================

def calibrate_camera_auto(
    board,
    all_corners,
    all_ids,
    captured_targets,
    image_size
):

    print()
    print(
        "Starte finale Kamerakalibrierung ..."
    )


    all_indices = list(
        range(
            len(all_corners)
        )
    )


    first = run_calibration(
        board,
        all_corners,
        all_ids,
        image_size,
        all_indices
    )


    if first is None:

        print(
            "Kalibrierung fehlgeschlagen."
        )

        return None


    final = first

    rejected_indices = []


    if AUTO_REJECT_BAD_IMAGES:

        errors = np.asarray(
            first["per_image_errors"],
            dtype=np.float64
        )


        median = float(
            np.median(
                errors
            )
        )


        mad = float(
            np.median(
                np.abs(
                    errors - median
                )
            )
        )


        threshold = max(
            0.75,
            median
            + 3.0
            * max(
                mad,
                0.03
            )
        )


        good_local = np.where(
            errors <= threshold
        )[0]


        if (
            len(good_local)
            >= MIN_IMAGES_AFTER_REJECTION
            and
            len(good_local)
            < len(errors)
        ):

            good_source = [
                first["source_indices"][i]
                for i
                in good_local
            ]


            second = run_calibration(
                board,
                all_corners,
                all_ids,
                image_size,
                good_source
            )


            if second is not None:

                final = second


                rejected_indices = [
                    i
                    for i
                    in all_indices
                    if i
                    not in good_source
                ]


    K = final[
        "camera_matrix"
    ]


    dist = final[
        "dist_coeffs"
    ]


    errors = np.asarray(
        final["per_image_errors"],
        dtype=np.float64
    )


    fx = float(
        K[0, 0]
    )

    fy = float(
        K[1, 1]
    )

    cx = float(
        K[0, 2]
    )

    cy = float(
        K[1, 2]
    )


    fov_x = math.degrees(
        2.0
        * math.atan(
            image_size[0]
            / (2.0 * fx)
        )
    )


    fov_y = math.degrees(
        2.0
        * math.atan(
            image_size[1]
            / (2.0 * fy)
        )
    )


    result = {

        "image_width":
            int(image_size[0]),

        "image_height":
            int(image_size[1]),

        "camera_matrix":
            K.tolist(),

        "dist_coeffs":
            dist.tolist(),

        "rms":
            float(
                final["rms"]
            ),

        "mean_reprojection_error_px":
            float(
                np.mean(errors)
            ),

        "median_reprojection_error_px":
            float(
                np.median(errors)
            ),

        "max_reprojection_error_px":
            float(
                np.max(errors)
            ),

        "per_image_errors_px":
            final["per_image_errors"],

        "number_of_images":
            len(
                final["source_indices"]
            ),

        "number_of_images_captured":
            len(
                all_corners
            ),

        "used_capture_indices":
            [
                int(i + 1)
                for i
                in final["source_indices"]
            ],

        "rejected_capture_indices":
            [
                int(i + 1)
                for i
                in rejected_indices
            ],

        "fx":
            fx,

        "fy":
            fy,

        "cx":
            cx,

        "cy":
            cy,

        "fov_x_deg":
            float(fov_x),

        "fov_y_deg":
            float(fov_y),

        "capture_targets":
            captured_targets
    }


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    CALIBRATION_FILE.write_text(
        json.dumps(
            result,
            indent=4
        ),
        encoding="utf-8"
    )


    print()
    print(
        "================================="
    )

    print(
        "KALIBRIERUNG FERTIG"
    )

    print(
        "================================="
    )

    print(
        "RMS:",
        result["rms"],
        "px"
    )

    print(
        "Mean:",
        result["mean_reprojection_error_px"],
        "px"
    )

    print(
        "Median:",
        result["median_reprojection_error_px"],
        "px"
    )

    print(
        "Max:",
        result["max_reprojection_error_px"],
        "px"
    )

    print(
        "Verwendete Bilder:",
        result["number_of_images"]
    )

    print(
        "Verworfene Bilder:",
        result["rejected_capture_indices"]
    )

    print(
        "Gespeichert:",
        CALIBRATION_FILE
    )


    return result


# ==========================================================
# MAIN
# ==========================================================

def main():

    print(
        "OpenCV:",
        cv2.__version__
    )


    dictionary, board, config = (
        load_charuco_board()
    )


    if board is None:

        return


    detector = (
        cv2.aruco.CharucoDetector(
            board
        )
    )


    guide_K, guide_dist = (
        load_guide_camera()
    )


    cap = open_camera()


    if cap is None:

        return


    IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # alte Auto-Bilder löschen
    for f in IMAGE_DIR.glob(
        "calibration_*.png"
    ):

        try:

            f.unlink()

        except Exception:

            pass


    all_corners = []

    all_ids = []

    captured_targets = []


    target_index = 0

    stable_frames = 0

    last_capture_time = 0.0

    flash_until = 0.0


    calibration_done = False


    while True:

        success, frame = (
            cap.read()
        )


        if not success:

            break


        raw_frame = (
            frame.copy()
        )


        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        # ==================================================
        # DETECTION
        # ==================================================

        try:

            (
                charuco_corners,
                charuco_ids,
                marker_corners,
                marker_ids
            ) = detector.detectBoard(
                gray
            )

        except Exception:

            charuco_corners = None
            charuco_ids = None
            marker_corners = None
            marker_ids = None


        corners_clean, ids_clean = (
            clean_detection(
                charuco_corners,
                charuco_ids
            )
        )


        corner_count = (
            0
            if ids_clean is None
            else len(ids_clean)
        )


        # ==================================================
        # MARKER / CORNERS ZEICHNEN
        # ==================================================

        if (
            marker_ids is not None
            and
            marker_corners is not None
        ):

            try:

                cv2.aruco.drawDetectedMarkers(
                    frame,
                    marker_corners,
                    marker_ids
                )

            except Exception:

                pass


        if (
            corners_clean is not None
            and
            ids_clean is not None
        ):

            try:

                cv2.aruco.drawDetectedCornersCharuco(
                    frame,
                    corners_clean,
                    ids_clean,
                    (0, 0, 255)
                )

            except Exception:

                pass


        now = time.perf_counter()


        # ==================================================
        # FERTIG?
        # ==================================================

        if (
            target_index
            >= len(TARGETS)
        ):

            if (
                AUTO_CALIBRATE_WHEN_DONE
                and
                not calibration_done
            ):

                calibration_done = True


                image_size = (
                    raw_frame.shape[1],
                    raw_frame.shape[0]
                )


                calibrate_camera_auto(
                    board,
                    all_corners,
                    all_ids,
                    captured_targets,
                    image_size
                )


            cv2.rectangle(
                frame,
                (0, 0),
                (frame.shape[1], 170),
                (0, 0, 0),
                -1
            )


            draw_centered_text(
                frame,
                "KALIBRIERUNG FERTIG",
                75,
                1.4,
                (0, 255, 0),
                3
            )


            draw_centered_text(
                frame,
                "Q = BEENDEN",
                130,
                0.9,
                (255, 255, 255),
                2
            )


        else:

            target = TARGETS[
                target_index
            ]


            # ==================================================
            # WICHTIG:
            # GROSSES SOLL-KÄSTCHEN ZEICHNEN
            # ==================================================

            draw_target_board_position(
                frame,
                target
            )


            # ==================================================
            # GUIDE POSE
            # ==================================================

            pose = estimate_guide_pose(
                board,
                corners_clean,
                ids_clean,
                guide_K,
                guide_dist
            )


            enough_corners = (
                corner_count
                >= MIN_CORNERS
            )


            matches = (
                enough_corners
                and
                pose_matches_target(
                    pose,
                    target
                )
            )


            if matches:

                stable_frames += 1

            else:

                stable_frames = 0


            cooldown_ok = (
                now
                - last_capture_time
                >= CAPTURE_COOLDOWN_S
            )


            # ==================================================
            # AUTO CAPTURE
            # ==================================================

            if (
                stable_frames
                >= STABLE_FRAMES_REQUIRED
                and
                cooldown_ok
                and
                corners_clean is not None
                and
                ids_clean is not None
            ):

                store_calibration_image(
                    raw_frame,
                    corners_clean,
                    ids_clean,
                    target,
                    all_corners,
                    all_ids,
                    captured_targets
                )


                target_index += 1

                stable_frames = 0

                last_capture_time = now

                flash_until = (
                    now + 0.30
                )


            # ==================================================
            # OBERER INFOBALKEN
            # ==================================================

            overlay = (
                frame.copy()
            )


            cv2.rectangle(
                overlay,
                (0, 0),
                (frame.shape[1], 170),
                (0, 0, 0),
                -1
            )


            frame = cv2.addWeighted(
                overlay,
                0.65,
                frame,
                0.35,
                0
            )


            if now < flash_until:

                text = (
                    "AUFGENOMMEN"
                )

                color = (
                    0,
                    255,
                    0
                )

            else:

                text = (
                    target["text"]
                )

                color = (
                    0,
                    255,
                    255
                )


            draw_centered_text(
                frame,
                text,
                70,
                1.15,
                color,
                3
            )


            draw_centered_text(
                frame,
                (
                    f"ZIEL "
                    f"{target_index + 1} / "
                    f"{len(TARGETS)}"
                ),
                120,
                0.85,
                (255, 255, 255),
                2
            )


            # ==================================================
            # STABIL-FORTSCHRITT
            # ==================================================

            if matches:

                progress = min(
                    1.0,
                    stable_frames
                    / STABLE_FRAMES_REQUIRED
                )


                bar_width = int(
                    WIDTH * 0.45
                )

                bar_height = 18


                x0 = int(
                    (
                        WIDTH
                        - bar_width
                    )
                    / 2
                )

                y0 = 140


                cv2.rectangle(
                    frame,
                    (x0, y0),
                    (
                        x0 + bar_width,
                        y0 + bar_height
                    ),
                    (100, 100, 100),
                    2
                )


                cv2.rectangle(
                    frame,
                    (x0, y0),
                    (
                        x0
                        + int(
                            bar_width
                            * progress
                        ),
                        y0
                        + bar_height
                    ),
                    (0, 255, 0),
                    -1
                )


        # ==================================================
        # DISPLAY
        # ==================================================

        display = cv2.resize(
            frame,
            (
                DISPLAY_WIDTH,
                DISPLAY_HEIGHT
            )
        )


        cv2.imshow(
            "3D Scanner - Auto Camera Calibration",
            display
        )


        key = (
            cv2.waitKey(1)
            & 0xFF
        )


        if (
            key == ord("q")
            or
            key == 27
        ):

            break


        if key == ord("r"):

            all_corners = []

            all_ids = []

            captured_targets = []

            target_index = 0

            stable_frames = 0

            calibration_done = False


            print(
                "Kalibrierungssequenz zurückgesetzt."
            )


    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()
