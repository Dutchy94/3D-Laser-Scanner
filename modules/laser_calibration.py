import cv2
import json
import math
import time
import numpy as np
from pathlib import Path


# ==========================================================
# PFADE
# ==========================================================

ROOT = Path(__file__).resolve().parents[1]

CHARUCO_CONFIG = (
    ROOT
    / "output"
    / "charuco"
    / "charuco_config.json"
)

CAMERA_CALIBRATION = (
    ROOT
    / "output"
    / "calibration"
    / "camera_calibration.json"
)

OUTPUT_DIR = (
    ROOT
    / "output"
    / "calibration"
)

LASER_CALIBRATION_FILE = (
    OUTPUT_DIR
    / "laser_calibration.json"
)

IMAGE_DIR = (
    OUTPUT_DIR
    / "laser_images_auto"
)


# ==========================================================
# KAMERA
# ==========================================================

CAMERA_INDEX = 0

WIDTH = 1920
HEIGHT = 1080
FPS = 30

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720


# ==========================================================
# ERKENNUNGSBEDINGUNGEN
# ==========================================================

MIN_CHARUCO_CORNERS = 18

MIN_LASER_PIXELS = 120

MAX_POSE_ERROR_PX = 1.0


# ==========================================================
# AUTOMATISCHE AUFNAHME
# ==========================================================

STABLE_FRAMES_REQUIRED = 5

ANGLE_TOLERANCE_DEG = 10.0

DISTANCE_TOLERANCE_MM = 90.0

POSITION_TOLERANCE_NORM = 0.20

CAPTURE_COOLDOWN_S = 0.7

AUTO_CALIBRATE_WHEN_DONE = True


# ==========================================================
# LASER-ERKENNUNG
# ==========================================================

MIN_RED = 160

MIN_RED_DIFFERENCE = 60

MAX_Y_DEVIATION = 10

MAX_GAP = 15

# ==========================================================
# ROBUSTE LASER-MITTELLINIE
# ==========================================================

MIN_LINE_WIDTH_PX = 1
MAX_LINE_WIDTH_PX = 14
LASER_BACKGROUND_MARGIN_PX = 10
MIN_PROFILE_PROMINENCE = 35.0
MAX_PROFILE_SATURATED_RATIO = 0.70
PROFILE_RELATIVE_FLOOR = 0.20
MIN_PROFILE_QUALITY = 0.30


# ==========================================================
# ROBUSTER EBENENFIT
# ==========================================================

ROBUST_PLANE_FIT = True

ROBUST_MIN_THRESHOLD_MM = 0.6

ROBUST_MAD_FACTOR = 3.5


# ==========================================================
# DEBUG
# ==========================================================

DEBUG = False


# ==========================================================
# GEFÜHRTE ZIELPOSEN
#
# WICHTIG:
# Kamera + Laser bilden zusammen den Scannerkopf.
# Der Anwender bewegt also den gesamten Scannerkopf.
#
# azimuth:
#   negativ = Scanner links vom Board
#   positiv = Scanner rechts vom Board
#
# elevation:
#   negativ = Scanner oberhalb des Boards
#   positiv = Scanner unterhalb des Boards
#
# image_x / image_y:
#   gewünschte Lage des Board-Mittelpunkts im Kamerabild
#
# guidance:
#   kurze konkrete Anweisung im Livebild
# ==========================================================

TARGETS = [

    {
        "text": "1/18  FRONTAL - MITTLERER ABSTAND",
        "guidance": "Scanner frontal auf das Board richten",
        "azimuth_deg": 0,
        "elevation_deg": 0,
        "distance_mm": 400,
        "image_x": 0.50,
        "image_y": 0.50
    },

    {
        "text": "2/18  CA. 20 GRAD VON LINKS",
        "guidance": "Scanner nach links bewegen und zum Board drehen",
        "azimuth_deg": -20,
        "elevation_deg": 0,
        "distance_mm": 400,
        "image_x": 0.50,
        "image_y": 0.50
    },

    {
        "text": "3/18  CA. 20 GRAD VON RECHTS",
        "guidance": "Scanner nach rechts bewegen und zum Board drehen",
        "azimuth_deg": 20,
        "elevation_deg": 0,
        "distance_mm": 400,
        "image_x": 0.50,
        "image_y": 0.50
    },

    {
        "text": "4/18  CA. 20 GRAD VON OBEN",
        "guidance": "Scanner oberhalb halten und nach unten aufs Board schauen",
        "azimuth_deg": 0,
        "elevation_deg": -20,
        "distance_mm": 400,
        "image_x": 0.50,
        "image_y": 0.50
    },

    {
        "text": "5/18  CA. 20 GRAD VON UNTEN",
        "guidance": "Scanner unterhalb halten und nach oben aufs Board schauen",
        "azimuth_deg": 0,
        "elevation_deg": 20,
        "distance_mm": 400,
        "image_x": 0.50,
        "image_y": 0.50
    },

    {
        "text": "6/18  LINKS + OBEN",
        "guidance": "Scanner links und etwas oberhalb des Boards halten",
        "azimuth_deg": -18,
        "elevation_deg": -15,
        "distance_mm": 400,
        "image_x": 0.50,
        "image_y": 0.50
    },

    {
        "text": "7/18  RECHTS + OBEN",
        "guidance": "Scanner rechts und etwas oberhalb des Boards halten",
        "azimuth_deg": 18,
        "elevation_deg": -15,
        "distance_mm": 400,
        "image_x": 0.50,
        "image_y": 0.50
    },

    {
        "text": "8/18  LINKS + UNTEN",
        "guidance": "Scanner links und etwas unterhalb des Boards halten",
        "azimuth_deg": -18,
        "elevation_deg": 15,
        "distance_mm": 400,
        "image_x": 0.50,
        "image_y": 0.50
    },

    {
        "text": "9/18  RECHTS + UNTEN",
        "guidance": "Scanner rechts und etwas unterhalb des Boards halten",
        "azimuth_deg": 18,
        "elevation_deg": 15,
        "distance_mm": 400,
        "image_x": 0.50,
        "image_y": 0.50
    },

    {
        "text": "10/18  NAH - FRONTAL",
        "guidance": "Scanner naeher heran, etwa 300 mm",
        "azimuth_deg": 0,
        "elevation_deg": 0,
        "distance_mm": 300,
        "image_x": 0.50,
        "image_y": 0.50
    },

    {
        "text": "11/18  NAH - VON LINKS",
        "guidance": "Etwa 300 mm Abstand und ca. 18 Grad von links",
        "azimuth_deg": -18,
        "elevation_deg": 0,
        "distance_mm": 300,
        "image_x": 0.50,
        "image_y": 0.50
    },

    {
        "text": "12/18  NAH - VON RECHTS",
        "guidance": "Etwa 300 mm Abstand und ca. 18 Grad von rechts",
        "azimuth_deg": 18,
        "elevation_deg": 0,
        "distance_mm": 300,
        "image_x": 0.50,
        "image_y": 0.50
    },

    {
        "text": "13/18  WEIT - FRONTAL",
        "guidance": "Scanner weiter weg, etwa 500 mm",
        "azimuth_deg": 0,
        "elevation_deg": 0,
        "distance_mm": 500,
        "image_x": 0.50,
        "image_y": 0.50
    },

    {
        "text": "14/18  WEIT - VON LINKS",
        "guidance": "Etwa 500 mm Abstand und ca. 18 Grad von links",
        "azimuth_deg": -18,
        "elevation_deg": 0,
        "distance_mm": 500,
        "image_x": 0.50,
        "image_y": 0.50
    },

    {
        "text": "15/18  WEIT - VON RECHTS",
        "guidance": "Etwa 500 mm Abstand und ca. 18 Grad von rechts",
        "azimuth_deg": 18,
        "elevation_deg": 0,
        "distance_mm": 500,
        "image_x": 0.50,
        "image_y": 0.50
    },

    {
        "text": "16/18  BOARD LINKS IM BILD",
        "guidance": "Scanner so verschieben, dass das Board links im Bild liegt",
        "azimuth_deg": 10,
        "elevation_deg": 0,
        "distance_mm": 400,
        "image_x": 0.28,
        "image_y": 0.50
    },

    {
        "text": "17/18  BOARD RECHTS IM BILD",
        "guidance": "Scanner so verschieben, dass das Board rechts im Bild liegt",
        "azimuth_deg": -10,
        "elevation_deg": 0,
        "distance_mm": 400,
        "image_x": 0.72,
        "image_y": 0.50
    },

    {
        "text": "18/18  BOARD UNTEN IM BILD",
        "guidance": "Scanner anheben, Board soll unten im Kamerabild liegen",
        "azimuth_deg": 0,
        "elevation_deg": -10,
        "distance_mm": 400,
        "image_x": 0.50,
        "image_y": 0.70
    }
]


# ==========================================================
# DATEIEN LADEN
# ==========================================================

def load_data():

    if not CHARUCO_CONFIG.exists():

        print("FEHLER:")
        print("ChArUco Config fehlt:")
        print(CHARUCO_CONFIG)

        return None


    if not CAMERA_CALIBRATION.exists():

        print("FEHLER:")
        print("Kamerakalibrierung fehlt:")
        print(CAMERA_CALIBRATION)

        return None


    charuco_config = json.loads(
        CHARUCO_CONFIG.read_text(
            encoding="utf-8"
        )
    )


    camera_config = json.loads(
        CAMERA_CALIBRATION.read_text(
            encoding="utf-8"
        )
    )


    camera_matrix = np.array(
        camera_config["camera_matrix"],
        dtype=np.float64
    )


    dist_coeffs = np.array(
        camera_config["dist_coeffs"],
        dtype=np.float64
    )


    print()
    print("Kamerakalibrierung geladen")

    if "mean_reprojection_error_px" in camera_config:

        print(
            "Camera Reprojection Error:",
            camera_config["mean_reprojection_error_px"],
            "px"
        )


    return (
        charuco_config,
        camera_matrix,
        dist_coeffs,
        camera_config
    )


# ==========================================================
# CHARUCO ERZEUGEN
# ==========================================================

def create_charuco(config):

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


    detector = cv2.aruco.CharucoDetector(
        board
    )


    return board, detector


# ==========================================================
# KAMERA OEFFNEN
# ==========================================================

def open_camera():

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )


    if not cap.isOpened():

        print("FEHLER:")
        print("Kamera konnte nicht geoeffnet werden")

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
# BOARD-MASKE
# ==========================================================

def create_board_mask(
    frame_shape,
    charuco_corners
):

    mask = np.zeros(
        frame_shape[:2],
        dtype=np.uint8
    )


    if (
        charuco_corners is None
        or
        len(charuco_corners) < 4
    ):

        return mask


    points = charuco_corners.reshape(
        -1,
        2
    )


    hull = cv2.convexHull(
        points.astype(
            np.float32
        )
    )


    hull = hull.astype(
        np.int32
    )


    # leicht vergroessern, damit die Laserlinie an den Randfeldern
    # nicht sofort abgeschnitten wird

    cv2.fillConvexPoly(
        mask,
        hull,
        255
    )


    kernel = np.ones(
        (25, 25),
        dtype=np.uint8
    )


    mask = cv2.dilate(
        mask,
        kernel,
        iterations=1
    )


    return mask


# ==========================================================
# HILFE: ZUSAMMENHAENGENDE PIXEL-RUNS
# ==========================================================

def find_runs(values):
    if len(values) == 0:
        return []
    runs = []
    start = 0
    for i in range(1, len(values)):
        if values[i] != values[i - 1] + 1:
            runs.append(values[start:i])
            start = i
    runs.append(values[start:])
    return runs


# ==========================================================
# ROBUSTES PROFILZENTRUM EINER LASERLINIE
# ==========================================================

def get_laser_profile_center(red_difference_column, red_column, run):
    if len(run) == 0:
        return None
    height = len(red_difference_column)
    run_start = int(run[0])
    run_end = int(run[-1])
    bg_start = max(0, run_start - LASER_BACKGROUND_MARGIN_PX)
    bg_end = min(height - 1, run_end + LASER_BACKGROUND_MARGIN_PX)
    background_values = []
    if bg_start < run_start:
        background_values.extend(red_difference_column[bg_start:run_start].astype(np.float64).tolist())
    if run_end + 1 <= bg_end:
        background_values.extend(red_difference_column[run_end + 1:bg_end + 1].astype(np.float64).tolist())
    background_level = float(np.median(background_values)) if background_values else 0.0
    y_values = np.asarray(run, dtype=np.float64)
    profile = red_difference_column[run].astype(np.float64) - background_level
    profile[profile < 0] = 0.0
    peak = float(np.max(profile))
    if peak < MIN_PROFILE_PROMINENCE:
        return None
    floor = peak * PROFILE_RELATIVE_FLOOR
    weights = profile - floor
    weights[weights < 0] = 0.0
    total = float(np.sum(weights))
    if total <= 0:
        return None
    center_y = float(np.sum(y_values * weights) / total)
    variance = float(np.sum(weights * (y_values - center_y) ** 2) / total)
    sigma = float(np.sqrt(max(variance, 0.0)))
    run_red = red_column[run]
    saturated_ratio = float(np.mean(run_red >= 250))
    if saturated_ratio > MAX_PROFILE_SATURATED_RATIO:
        return None
    left = float(np.sum(weights[y_values < center_y]))
    right = float(np.sum(weights[y_values > center_y]))
    denom = left + right
    asymmetry = abs(left-right)/denom if denom > 1e-9 else 1.0
    quality = 1.0
    quality *= float(np.clip(1.0 - saturated_ratio, 0.0, 1.0))
    quality *= float(np.clip(1.0 - 0.7*asymmetry, 0.0, 1.0))
    quality *= float(np.clip(1.0 - max(0.0, len(run)-6.0)/12.0, 0.25, 1.0))
    if quality < MIN_PROFILE_QUALITY:
        return None
    return {
        'center_y': center_y,
        'sigma_px': sigma,
        'width_px': float(len(run)),
        'peak': peak,
        'background': background_level,
        'saturated_ratio': saturated_ratio,
        'quality': quality
    }


# ==========================================================
# GEMEINSAME ROBUSTE LASERERKENNUNG
# ==========================================================

def detect_laser_profile_center(frame, board_mask=None, x_step=1, min_points=80, max_x_gap=12, max_y_jump_px=35, min_segment_points=20):
    b = frame[:,:,0].astype(np.int16)
    g = frame[:,:,1].astype(np.int16)
    r = frame[:,:,2].astype(np.int16)
    red_difference = r - np.maximum(g,b)
    rgb_mask = ((r >= MIN_RED) & (red_difference >= MIN_RED_DIFFERENCE)).astype(np.uint8)*255
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv1_min = globals().get('HSV1_MIN',(0,100,120))
    hsv1_max = globals().get('HSV1_MAX',(12,255,255))
    hsv2_min = globals().get('HSV2_MIN',(168,100,120))
    hsv2_max = globals().get('HSV2_MAX',(179,255,255))
    mask1 = cv2.inRange(hsv,np.array(hsv1_min,dtype=np.uint8),np.array(hsv1_max,dtype=np.uint8))
    mask2 = cv2.inRange(hsv,np.array(hsv2_min,dtype=np.uint8),np.array(hsv2_max,dtype=np.uint8))
    mask = cv2.bitwise_and(rgb_mask, cv2.bitwise_or(mask1,mask2))
    if board_mask is not None:
        mask = cv2.bitwise_and(mask, board_mask)
    h,w = mask.shape
    candidates=[]; widths=[]; sigmas=[]; qualities=[]
    for x in range(0,w,x_step):
        ys=np.where(mask[:,x]>0)[0]
        if len(ys)==0: continue
        best_score=None; best=None
        for run in find_runs(ys):
            lw=len(run)
            if lw < MIN_LINE_WIDTH_PX or lw > MAX_LINE_WIDTH_PX: continue
            res=get_laser_profile_center(red_difference[:,x], r[:,x], run)
            if res is None: continue
            score=res['quality']*res['peak']
            if best_score is None or score > best_score:
                best_score=score; best=res
        if best is not None:
            candidates.append([float(x),float(best['center_y'])])
            widths.append(best['width_px']); sigmas.append(best['sigma_px']); qualities.append(best['quality'])
    diagnostics={
        'mean_width_px': float(np.mean(widths)) if widths else None,
        'median_width_px': float(np.median(widths)) if widths else None,
        'mean_sigma_px': float(np.mean(sigmas)) if sigmas else None,
        'mean_quality': float(np.mean(qualities)) if qualities else None
    }
    if len(candidates) < min_points:
        return np.empty((0,2),dtype=np.float32), mask, False, diagnostics
    candidates=np.asarray(candidates,dtype=np.float32)
    segments=[]; current=[candidates[0]]
    for i in range(1,len(candidates)):
        p0,p1=candidates[i-1],candidates[i]
        if abs(p1[0]-p0[0]) <= max_x_gap and abs(p1[1]-p0[1]) <= max_y_jump_px:
            current.append(p1)
        else:
            if len(current) >= min_segment_points: segments.append(current)
            current=[p1]
    if len(current) >= min_segment_points: segments.append(current)
    final=[]
    for seg in segments: final.extend(seg)
    final=np.asarray(final,dtype=np.float32)
    if len(final) < min_points:
        return final, mask, False, diagnostics
    return final, mask, True, diagnostics


# ==========================================================
# WRAPPER FUER LASER-KALIBRIERUNG
# ==========================================================

def detect_laser(frame, board_mask=None):
    return detect_laser_profile_center(
        frame,
        board_mask=board_mask,
        x_step=1,
        min_points=MIN_LASER_PIXELS,
        max_x_gap=MAX_GAP,
        max_y_jump_px=35,
        min_segment_points=20
    )


# ==========================================================
# BOARD-POSE ROBUST
# ==========================================================

def get_board_pose(
    board,
    charuco_corners,
    charuco_ids,
    camera_matrix,
    dist_coeffs
):

    if (
        charuco_corners is None
        or
        charuco_ids is None
    ):

        return None


    if len(
        charuco_corners
    ) != len(
        charuco_ids
    ):

        return None


    if len(
        charuco_ids
    ) < MIN_CHARUCO_CORNERS:

        return None


    try:

        (
            object_points,
            image_points
        ) = board.matchImagePoints(
            charuco_corners,
            charuco_ids
        )

    except Exception:

        return None


    if (
        object_points is None
        or
        image_points is None
    ):

        return None


    object_points = np.asarray(
        object_points,
        dtype=np.float64
    ).reshape(
        -1,
        3
    )


    image_points = np.asarray(
        image_points,
        dtype=np.float64
    ).reshape(
        -1,
        2
    )


    if len(
        object_points
    ) < 6:

        return None


    try:

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

    except cv2.error:

        return None


    if not success:

        return None


    try:

        rvec, tvec = cv2.solvePnPRefineLM(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            rvec,
            tvec
        )

    except cv2.error:

        pass


    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs
    )


    projected = projected.reshape(
        -1,
        2
    )


    errors = np.linalg.norm(
        image_points
        - projected,
        axis=1
    )


    mean_error = float(
        np.mean(
            errors
        )
    )


    if mean_error > MAX_POSE_ERROR_PX:

        return None


    R, _ = cv2.Rodrigues(
        rvec
    )


    points_camera = (
        R
        @ object_points.T
    ).T + tvec.reshape(
        1,
        3
    )


    if np.any(
        points_camera[:, 2] <= 0
    ):

        return None


    return {
        "rvec":
            rvec,

        "tvec":
            tvec,

        "error":
            mean_error,

        "object_points":
            object_points,

        "image_points":
            image_points
    }


# ==========================================================
# GUIDE-METRIKEN AUS POSE
# ==========================================================

def get_pose_guidance(
    pose,
    image_points
):

    if pose is None:

        return None


    rvec = pose[
        "rvec"
    ]

    tvec = pose[
        "tvec"
    ]


    R, _ = cv2.Rodrigues(
        rvec
    )


    camera_position = (
        -R.T
        @ tvec.reshape(
            3
        )
    )


    obj = pose[
        "object_points"
    ]


    board_center = (
        np.min(
            obj,
            axis=0
        )
        +
        np.max(
            obj,
            axis=0
        )
    ) / 2.0


    delta = (
        camera_position
        - board_center
    )


    depth = abs(
        delta[2]
    )


    if depth < 1.0:

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
        image_points,
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
            float(
                azimuth_deg
            ),

        "elevation_deg":
            float(
                elevation_deg
            ),

        "distance_mm":
            distance_mm,

        "image_x":
            image_x,

        "image_y":
            image_y
    }


# ==========================================================
# BOARD-EBENE IN KAMERAKOORDINATEN
# ==========================================================

def calculate_board_plane(
    rvec,
    tvec
):

    R, _ = cv2.Rodrigues(
        rvec
    )


    n_board = np.array(
        [
            0.0,
            0.0,
            1.0
        ],
        dtype=np.float64
    )


    n_camera = (
        R
        @ n_board
    )


    n_camera = (
        n_camera
        / np.linalg.norm(
            n_camera
        )
    )


    p0 = tvec.reshape(
        3
    )


    d = -float(
        np.dot(
            n_camera,
            p0
        )
    )


    return (
        n_camera,
        d
    )


# ==========================================================
# LASERPIXEL -> 3D AUF BOARD-EBENE
# ==========================================================

def laser_pixels_to_3d(
    pixels,
    camera_matrix,
    dist_coeffs,
    plane_normal,
    plane_d
):

    if len(
        pixels
    ) == 0:

        return np.empty(
            (0, 3),
            dtype=np.float64
        )


    pixel_array = pixels.reshape(
        -1,
        1,
        2
    ).astype(
        np.float64
    )


    undistorted = cv2.undistortPoints(
        pixel_array,
        camera_matrix,
        dist_coeffs
    ).reshape(
        -1,
        2
    )


    points_3d = []


    for p in undistorted:

        ray = np.array(
            [
                p[0],
                p[1],
                1.0
            ],
            dtype=np.float64
        )


        denominator = float(
            np.dot(
                plane_normal,
                ray
            )
        )


        if abs(
            denominator
        ) < 1e-8:

            continue


        distance = (
            -plane_d
            / denominator
        )


        if distance <= 0:

            continue


        point = (
            ray
            * distance
        )


        points_3d.append(
            point
        )


    return np.asarray(
        points_3d,
        dtype=np.float64
    )


# ==========================================================
# EBENE FITTEN
# ==========================================================

def fit_plane_basic(
    points
):

    if len(
        points
    ) < 3:

        return None


    centroid = np.mean(
        points,
        axis=0
    )


    centered = (
        points
        - centroid
    )


    _, _, vh = np.linalg.svd(
        centered,
        full_matrices=False
    )


    normal = vh[
        -1
    ]


    normal = (
        normal
        / np.linalg.norm(
            normal
        )
    )


    d = -float(
        np.dot(
            normal,
            centroid
        )
    )


    if normal[2] < 0:

        normal = -normal

        d = -d


    distances = np.abs(
        points
        @ normal
        + d
    )


    return {
        "normal":
            normal,

        "d":
            d,

        "centroid":
            centroid,

        "distances":
            distances,

        "mean_error":
            float(
                np.mean(
                    distances
                )
            ),

        "rms_error":
            float(
                np.sqrt(
                    np.mean(
                        distances ** 2
                    )
                )
            ),

        "max_error":
            float(
                np.max(
                    distances
                )
            )
    }


# ==========================================================
# ROBUSTER EBENENFIT
# ==========================================================

def fit_plane(
    points
):

    first = fit_plane_basic(
        points
    )


    if first is None:

        return None


    if not ROBUST_PLANE_FIT:

        first[
            "used_points"
        ] = points

        first[
            "rejected_points"
        ] = 0

        first[
            "threshold_mm"
        ] = None

        return first


    residuals = first[
        "distances"
    ]


    median = float(
        np.median(
            residuals
        )
    )


    mad = float(
        np.median(
            np.abs(
                residuals
                - median
            )
        )
    )


    threshold = max(
        ROBUST_MIN_THRESHOLD_MM,
        median
        +
        ROBUST_MAD_FACTOR
        * max(
            mad,
            0.05
        )
    )


    good_mask = (
        residuals
        <= threshold
    )


    good_points = points[
        good_mask
    ]


    if len(
        good_points
    ) < max(
        100,
        int(
            len(points)
            * 0.60
        )
    ):

        first[
            "used_points"
        ] = points

        first[
            "rejected_points"
        ] = 0

        first[
            "threshold_mm"
        ] = threshold

        return first


    final = fit_plane_basic(
        good_points
    )


    if final is None:

        return first


    final[
        "used_points"
    ] = good_points

    final[
        "rejected_points"
    ] = int(
        len(points)
        - len(
            good_points
        )
    )

    final[
        "threshold_mm"
    ] = float(
        threshold
    )


    return final


# ==========================================================
# LASER-KALIBRIERUNG SPEICHERN
# ==========================================================

def save_laser_calibration(
    all_points,
    pose_records,
    camera_config
):

    result = fit_plane(
        all_points
    )


    if result is None:

        print(
            "Nicht genug Punkte fuer Ebenenfit."
        )

        return None


    normal = result[
        "normal"
    ]

    d = result[
        "d"
    ]

    centroid = result[
        "centroid"
    ]


    data = {

        "coordinate_system":
            "camera",

        "unit":
            "mm",

        "laser_center_method":
            "robust_intensity_centroid",

        "profile_relative_floor":
            float(PROFILE_RELATIVE_FLOOR),

        "min_profile_quality":
            float(MIN_PROFILE_QUALITY),

        "plane": {
            "a":
                float(
                    normal[0]
                ),

            "b":
                float(
                    normal[1]
                ),

            "c":
                float(
                    normal[2]
                ),

            "d":
                float(
                    d
                )
        },

        "normal":
            normal.tolist(),

        "centroid_mm":
            centroid.tolist(),

        "number_of_points_raw":
            int(
                len(
                    all_points
                )
            ),

        "number_of_points":
            int(
                len(
                    result[
                        "used_points"
                    ]
                )
            ),

        "rejected_points":
            int(
                result[
                    "rejected_points"
                ]
            ),

        "robust_threshold_mm":
            result[
                "threshold_mm"
            ],

        "number_of_board_poses":
            int(
                len(
                    pose_records
                )
            ),

        "mean_plane_error_mm":
            float(
                result[
                    "mean_error"
                ]
            ),

        "rms_plane_error_mm":
            float(
                result[
                    "rms_error"
                ]
            ),

        "max_plane_error_mm":
            float(
                result[
                    "max_error"
                ]
            ),

        "pose_records":
            pose_records,

        "camera_calibration_reference": {
            "fx":
                camera_config.get(
                    "fx"
                ),

            "fy":
                camera_config.get(
                    "fy"
                ),

            "cx":
                camera_config.get(
                    "cx"
                ),

            "cy":
                camera_config.get(
                    "cy"
                ),

            "mean_reprojection_error_px":
                camera_config.get(
                    "mean_reprojection_error_px"
                )
        }
    }


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    LASER_CALIBRATION_FILE.write_text(
        json.dumps(
            data,
            indent=4
        ),
        encoding="utf-8"
    )


    print()
    print(
        "========================================"
    )
    print(
        "LASER-KALIBRIERUNG FERTIG"
    )
    print(
        "========================================"
    )

    print(
        "Posen:",
        len(
            pose_records
        )
    )

    print(
        "Punkte roh:",
        len(
            all_points
        )
    )

    print(
        "Punkte verwendet:",
        len(
            result[
                "used_points"
            ]
        )
    )

    print(
        "Punkte verworfen:",
        result[
            "rejected_points"
        ]
    )

    print()
    print(
        "Laserebene:"
    )

    print(
        f"{normal[0]:.9f} * X + "
        f"{normal[1]:.9f} * Y + "
        f"{normal[2]:.9f} * Z + "
        f"{d:.9f} = 0"
    )

    print()
    print(
        "Mean Plane Error:",
        result[
            "mean_error"
        ],
        "mm"
    )

    print(
        "RMS Plane Error:",
        result[
            "rms_error"
        ],
        "mm"
    )

    print(
        "Max Plane Error:",
        result[
            "max_error"
        ],
        "mm"
    )

    print()
    print(
        "Gespeichert:"
    )

    print(
        LASER_CALIBRATION_FILE
    )


    return data


# ==========================================================
# ZIELPOSITION IM BILD ZEICHNEN
# ==========================================================

def draw_target_board_position(
    frame,
    target
):

    h, w = frame.shape[
        :2
    ]


    cx = int(
        target[
            "image_x"
        ]
        * w
    )


    cy = int(
        target[
            "image_y"
        ]
        * h
    )


    distance = float(
        target[
            "distance_mm"
        ]
    )


    scale = (
        400.0
        / max(
            distance,
            1.0
        )
    )


    box_h = int(
        h
        * 0.45
        * scale
    )


    box_w = int(
        box_h
        * (
            168.0
            / 252.0
        )
    )


    box_h = int(
        np.clip(
            box_h,
            220,
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
        cx
        - box_w / 2
    )

    y1 = int(
        cy
        - box_h / 2
    )

    x2 = int(
        cx
        + box_w / 2
    )

    y2 = int(
        cy
        + box_h / 2
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


    overlay = (
        frame.copy()
    )


    cv2.rectangle(
        overlay,
        (
            x1,
            y1
        ),
        (
            x2,
            y2
        ),
        (
            0,
            255,
            255
        ),
        -1
    )


    frame[:] = cv2.addWeighted(
        overlay,
        0.10,
        frame,
        0.90,
        0
    )


    cv2.rectangle(
        frame,
        (
            x1,
            y1
        ),
        (
            x2,
            y2
        ),
        (
            0,
            255,
            255
        ),
        4
    )


    cv2.line(
        frame,
        (
            cx - 35,
            cy
        ),
        (
            cx + 35,
            cy
        ),
        (
            0,
            255,
            255
        ),
        3
    )


    cv2.line(
        frame,
        (
            cx,
            cy - 35
        ),
        (
            cx,
            cy + 35
        ),
        (
            0,
            255,
            255
        ),
        3
    )


# ==========================================================
# ZENTRIERTER TEXT
# ==========================================================

def draw_centered_text(
    frame,
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
            frame.shape[1]
            - size[0]
        )
        / 2
    )


    cv2.putText(
        frame,
        text,
        (
            x,
            y
        ),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


# ==========================================================
# IST-ZUSTAND -> KURZE HANDLUNGSANWEISUNG
# ==========================================================

def get_adjustment_text(
    pose_guidance,
    target,
    charuco_count,
    laser_detected,
    laser_count
):

    if charuco_count < MIN_CHARUCO_CORNERS:

        return (
            "Board vollstaendiger ins Bild bringen"
        )


    if pose_guidance is None:

        return (
            "Board ruhig halten - Pose wird gesucht"
        )


    distance_error = (
        pose_guidance[
            "distance_mm"
        ]
        - target[
            "distance_mm"
        ]
    )


    if distance_error > DISTANCE_TOLERANCE_MM:

        return (
            "SCANNER NAEHER ZUM BOARD"
        )


    if distance_error < -DISTANCE_TOLERANCE_MM:

        return (
            "SCANNER WEITER VOM BOARD WEG"
        )


    az_error = (
        pose_guidance[
            "azimuth_deg"
        ]
        - target[
            "azimuth_deg"
        ]
    )


    if az_error > ANGLE_TOLERANCE_DEG:

        return (
            "MEHR NACH LINKS GEHEN"
        )


    if az_error < -ANGLE_TOLERANCE_DEG:

        return (
            "MEHR NACH RECHTS GEHEN"
        )


    el_error = (
        pose_guidance[
            "elevation_deg"
        ]
        - target[
            "elevation_deg"
        ]
    )


    if el_error > ANGLE_TOLERANCE_DEG:

        return (
            "SCANNER HOEHER HALTEN"
        )


    if el_error < -ANGLE_TOLERANCE_DEG:

        return (
            "SCANNER TIEFER HALTEN"
        )


    x_error = (
        pose_guidance[
            "image_x"
        ]
        - target[
            "image_x"
        ]
    )


    if x_error > POSITION_TOLERANCE_NORM:

        return (
            "BOARD MUSS WEITER LINKS IM BILD SEIN"
        )


    if x_error < -POSITION_TOLERANCE_NORM:

        return (
            "BOARD MUSS WEITER RECHTS IM BILD SEIN"
        )


    y_error = (
        pose_guidance[
            "image_y"
        ]
        - target[
            "image_y"
        ]
    )


    if y_error > POSITION_TOLERANCE_NORM:

        return (
            "BOARD MUSS WEITER OBEN IM BILD SEIN"
        )


    if y_error < -POSITION_TOLERANCE_NORM:

        return (
            "BOARD MUSS WEITER UNTEN IM BILD SEIN"
        )


    if not laser_detected:

        return (
            "LASERLINIE AUF DAS BOARD RICHTEN"
        )


    if laser_count < MIN_LASER_PIXELS:

        return (
            "MEHR LASERLINIE AUF DEM BOARD SICHTBAR MACHEN"
        )


    return (
        "RICHTIG - RUHIG HALTEN"
    )


# ==========================================================
# ZIELBEDINGUNG
# ==========================================================

def pose_matches_target(
    pose_guidance,
    target
):

    if pose_guidance is None:

        return False


    return (

        abs(
            pose_guidance[
                "azimuth_deg"
            ]
            - target[
                "azimuth_deg"
            ]
        )
        <= ANGLE_TOLERANCE_DEG

        and

        abs(
            pose_guidance[
                "elevation_deg"
            ]
            - target[
                "elevation_deg"
            ]
        )
        <= ANGLE_TOLERANCE_DEG

        and

        abs(
            pose_guidance[
                "distance_mm"
            ]
            - target[
                "distance_mm"
            ]
        )
        <= DISTANCE_TOLERANCE_MM

        and

        abs(
            pose_guidance[
                "image_x"
            ]
            - target[
                "image_x"
            ]
        )
        <= POSITION_TOLERANCE_NORM

        and

        abs(
            pose_guidance[
                "image_y"
            ]
            - target[
                "image_y"
            ]
        )
        <= POSITION_TOLERANCE_NORM
    )



# ==========================================================
# SOLLWERTE / TOLERANZBEREICHE ALS TEXT
# ==========================================================

def get_target_bounds_text(
    target
):

    az_min = (
        target["azimuth_deg"]
        - ANGLE_TOLERANCE_DEG
    )

    az_max = (
        target["azimuth_deg"]
        + ANGLE_TOLERANCE_DEG
    )


    el_min = (
        target["elevation_deg"]
        - ANGLE_TOLERANCE_DEG
    )

    el_max = (
        target["elevation_deg"]
        + ANGLE_TOLERANCE_DEG
    )


    dist_min = (
        target["distance_mm"]
        - DISTANCE_TOLERANCE_MM
    )

    dist_max = (
        target["distance_mm"]
        + DISTANCE_TOLERANCE_MM
    )


    x_min = (
        target["image_x"]
        - POSITION_TOLERANCE_NORM
    )

    x_max = (
        target["image_x"]
        + POSITION_TOLERANCE_NORM
    )


    y_min = (
        target["image_y"]
        - POSITION_TOLERANCE_NORM
    )

    y_max = (
        target["image_y"]
        + POSITION_TOLERANCE_NORM
    )


    line1 = (
        f"Soll L/R: "
        f"{az_min:+.1f}..{az_max:+.1f} deg | "
        f"O/U: "
        f"{el_min:+.1f}..{el_max:+.1f} deg | "
        f"Abstand: "
        f"{dist_min:.0f}..{dist_max:.0f} mm"
    )


    line2 = (
        f"Soll Bildpos X: "
        f"{x_min:.2f}..{x_max:.2f} | "
        f"Y: "
        f"{y_min:.2f}..{y_max:.2f}"
    )


    return (
        line1,
        line2
    )


# ==========================================================
# IST/SOLL STATUS PRO WERT
# ==========================================================

def value_ok(
    value,
    minimum,
    maximum
):

    return (
        value >= minimum
        and
        value <= maximum
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    print()
    print(
        "OpenCV:",
        cv2.__version__
    )


    loaded = load_data()


    if loaded is None:

        return


    (
        charuco_config,
        camera_matrix,
        dist_coeffs,
        camera_config
    ) = loaded


    board, detector = create_charuco(
        charuco_config
    )


    cap = open_camera()


    if cap is None:

        return


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    for old in IMAGE_DIR.glob(
        "laser_pose_*.png"
    ):

        try:

            old.unlink()

        except Exception:

            pass


    collected_points = []

    pose_records = []


    target_index = 0

    stable_frames = 0

    last_capture_time = 0.0

    flash_until = 0.0

    calibration_done = False


    show_mask = False


    while True:

        success, frame = cap.read()


        if not success:

            print(
                "Framefehler"
            )

            break


        raw_frame = (
            frame.copy()
        )


        gray = cv2.cvtColor(
            raw_frame,
            cv2.COLOR_BGR2GRAY
        )


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


        charuco_count = (
            0
            if charuco_ids is None
            else len(
                charuco_ids
            )
        )


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
            charuco_corners is not None
            and
            charuco_ids is not None
            and
            len(
                charuco_corners
            )
            ==
            len(
                charuco_ids
            )
        ):

            try:

                cv2.aruco.drawDetectedCornersCharuco(
                    frame,
                    charuco_corners,
                    charuco_ids,
                    (
                        0,
                        0,
                        255
                    )
                )

            except Exception:

                pass


        board_mask = create_board_mask(
            raw_frame.shape,
            charuco_corners
        )


        (
            laser_pixels,
            laser_mask,
            laser_detected,
            laser_diagnostics
        ) = detect_laser(
            raw_frame,
            board_mask
        )


        laser_count = len(
            laser_pixels
        )

        laser_width_text = (
            "-" if laser_diagnostics.get("mean_width_px") is None
            else f"{laser_diagnostics['mean_width_px']:.2f} px"
        )

        laser_quality_text = (
            "-" if laser_diagnostics.get("mean_quality") is None
            else f"{laser_diagnostics['mean_quality']:.2f}"
        )


        for p in laser_pixels[
            ::4
        ]:

            cv2.circle(
                frame,
                (
                    int(
                        round(
                            p[0]
                        )
                    ),
                    int(
                        round(
                            p[1]
                        )
                    )
                ),
                2,
                (
                    0,
                    255,
                    255
                ),
                -1
            )


        pose = get_board_pose(
            board,
            charuco_corners,
            charuco_ids,
            camera_matrix,
            dist_coeffs
        )


        pose_guidance = None

        pose_error = None

        current_3d = np.empty(
            (
                0,
                3
            ),
            dtype=np.float64
        )


        if pose is not None:

            pose_error = pose[
                "error"
            ]


            pose_guidance = get_pose_guidance(
                pose,
                pose[
                    "image_points"
                ]
            )


            (
                board_normal,
                board_d
            ) = calculate_board_plane(
                pose[
                    "rvec"
                ],
                pose[
                    "tvec"
                ]
            )


            if (
                laser_detected
                and
                laser_count
                >= MIN_LASER_PIXELS
            ):

                current_3d = laser_pixels_to_3d(
                    laser_pixels,
                    camera_matrix,
                    dist_coeffs,
                    board_normal,
                    board_d
                )


        now = time.perf_counter()


        # ==================================================
        # ALLE ZIELE ERLEDIGT
        # ==================================================

        if target_index >= len(
            TARGETS
        ):

            if (
                AUTO_CALIBRATE_WHEN_DONE
                and
                not calibration_done
            ):

                calibration_done = True


                if len(
                    collected_points
                ) > 0:

                    all_points = np.vstack(
                        collected_points
                    )


                    save_laser_calibration(
                        all_points,
                        pose_records,
                        camera_config
                    )


            overlay = frame.copy()


            cv2.rectangle(
                overlay,
                (
                    0,
                    0
                ),
                (
                    frame.shape[1],
                    220
                ),
                (
                    0,
                    0,
                    0
                ),
                -1
            )


            frame = cv2.addWeighted(
                overlay,
                0.70,
                frame,
                0.30,
                0
            )


            draw_centered_text(
                frame,
                "LASER-KALIBRIERUNG FERTIG",
                85,
                1.35,
                (
                    0,
                    255,
                    0
                ),
                3
            )


            draw_centered_text(
                frame,
                "laser_calibration.json wurde gespeichert",
                145,
                0.85,
                (
                    255,
                    255,
                    255
                ),
                2
            )


            draw_centered_text(
                frame,
                "Q = BEENDEN",
                195,
                0.8,
                (
                    255,
                    255,
                    255
                ),
                2
            )


        else:

            target = TARGETS[
                target_index
            ]


            draw_target_board_position(
                frame,
                target
            )


            target_pose_ok = pose_matches_target(
                pose_guidance,
                target
            )


            ready = (
                target_pose_ok
                and
                pose is not None
                and
                charuco_count
                >= MIN_CHARUCO_CORNERS
                and
                laser_detected
                and
                laser_count
                >= MIN_LASER_PIXELS
                and
                len(
                    current_3d
                )
                >= MIN_LASER_PIXELS
            )


            if ready:

                stable_frames += 1

            else:

                stable_frames = 0


            cooldown_ok = (
                now
                - last_capture_time
                >= CAPTURE_COOLDOWN_S
            )


            if (
                ready
                and
                stable_frames
                >= STABLE_FRAMES_REQUIRED
                and
                cooldown_ok
            ):

                collected_points.append(
                    current_3d.copy()
                )


                record = {
                    "target_index":
                        int(
                            target_index + 1
                        ),

                    "target_text":
                        target[
                            "text"
                        ],

                    "guidance":
                        target[
                            "guidance"
                        ],

                    "charuco_corners":
                        int(
                            charuco_count
                        ),

                    "laser_points":
                        int(
                            len(
                                current_3d
                            )
                        ),

                    "laser_profile": laser_diagnostics,

                    "pose_error_px":
                        float(
                            pose_error
                        ),

                    "measured_pose": {
                        "azimuth_deg":
                            float(
                                pose_guidance[
                                    "azimuth_deg"
                                ]
                            ),

                        "elevation_deg":
                            float(
                                pose_guidance[
                                    "elevation_deg"
                                ]
                            ),

                        "distance_mm":
                            float(
                                pose_guidance[
                                    "distance_mm"
                                ]
                            ),

                        "image_x":
                            float(
                                pose_guidance[
                                    "image_x"
                                ]
                            ),

                        "image_y":
                            float(
                                pose_guidance[
                                    "image_y"
                                ]
                            )
                    }
                }


                pose_records.append(
                    record
                )


                filename = (
                    IMAGE_DIR
                    / f"laser_pose_{target_index + 1:02d}.png"
                )


                cv2.imwrite(
                    str(
                        filename
                    ),
                    raw_frame
                )


                print(
                    f"Pose {target_index + 1:02d} automatisch gespeichert"
                )


                target_index += 1

                stable_frames = 0

                last_capture_time = now

                flash_until = (
                    now
                    + 0.35
                )


            # ==================================================
            # INFOBALKEN
            # ==================================================

            overlay = frame.copy()


            cv2.rectangle(
                overlay,
                (
                    0,
                    0
                ),
                (
                    frame.shape[1],
                    255
                ),
                (
                    0,
                    0,
                    0
                ),
                -1
            )


            frame = cv2.addWeighted(
                overlay,
                0.68,
                frame,
                0.32,
                0
            )


            if now < flash_until:

                main_text = (
                    "AUFGENOMMEN - NAECHSTE POSITION"
                )

                main_color = (
                    0,
                    255,
                    0
                )

            else:

                main_text = (
                    target[
                        "text"
                    ]
                )

                main_color = (
                    0,
                    255,
                    255
                )


            draw_centered_text(
                frame,
                main_text,
                65,
                1.1,
                main_color,
                3
            )


            draw_centered_text(
                frame,
                target[
                    "guidance"
                ],
                115,
                0.82,
                (
                    255,
                    255,
                    255
                ),
                2
            )


            adjustment = get_adjustment_text(
                pose_guidance,
                target,
                charuco_count,
                laser_detected,
                laser_count
            )


            if ready:

                adjustment_color = (
                    0,
                    255,
                    0
                )

            else:

                adjustment_color = (
                    0,
                    165,
                    255
                )


            draw_centered_text(
                frame,
                adjustment,
                165,
                0.95,
                adjustment_color,
                3
            )


            info = (
                f"Corners {charuco_count} | "
                f"Laser {laser_count} | "
                f"Breite {laser_width_text} | "
                f"Qual {laser_quality_text} | "
                f"3D {len(current_3d)}"
            )


            draw_centered_text(
                frame,
                info,
                215,
                0.72,
                (
                    255,
                    255,
                    0
                ),
                2
            )


            # ==================================================
            # AKTUELLE POSE UNTEN LINKS
            # ==================================================

            if pose_guidance is not None:

                az_min = (
                    target["azimuth_deg"]
                    - ANGLE_TOLERANCE_DEG
                )

                az_max = (
                    target["azimuth_deg"]
                    + ANGLE_TOLERANCE_DEG
                )


                el_min = (
                    target["elevation_deg"]
                    - ANGLE_TOLERANCE_DEG
                )

                el_max = (
                    target["elevation_deg"]
                    + ANGLE_TOLERANCE_DEG
                )


                dist_min = (
                    target["distance_mm"]
                    - DISTANCE_TOLERANCE_MM
                )

                dist_max = (
                    target["distance_mm"]
                    + DISTANCE_TOLERANCE_MM
                )


                az_ok = value_ok(
                    pose_guidance["azimuth_deg"],
                    az_min,
                    az_max
                )


                el_ok = value_ok(
                    pose_guidance["elevation_deg"],
                    el_min,
                    el_max
                )


                dist_ok = value_ok(
                    pose_guidance["distance_mm"],
                    dist_min,
                    dist_max
                )


                ist_color = (
                    0,
                    255,
                    0
                ) if (
                    az_ok
                    and
                    el_ok
                    and
                    dist_ok
                ) else (
                    0,
                    165,
                    255
                )


                cv2.putText(
                    frame,
                    (
                        f"Ist: "
                        f"L/R {pose_guidance['azimuth_deg']:+.1f} deg | "
                        f"O/U {pose_guidance['elevation_deg']:+.1f} deg | "
                        f"Abstand {pose_guidance['distance_mm']:.0f} mm"
                    ),
                    (
                        30,
                        frame.shape[0] - 125
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    ist_color,
                    2,
                    cv2.LINE_AA
                )


                (
                    soll_line1,
                    soll_line2
                ) = get_target_bounds_text(
                    target
                )


                cv2.putText(
                    frame,
                    soll_line1,
                    (
                        30,
                        frame.shape[0] - 90
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.67,
                    (
                        255,
                        255,
                        0
                    ),
                    2,
                    cv2.LINE_AA
                )


                cv2.putText(
                    frame,
                    soll_line2,
                    (
                        30,
                        frame.shape[0] - 58
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.67,
                    (
                        255,
                        255,
                        0
                    ),
                    2,
                    cv2.LINE_AA
                )


            # ==================================================
            # STABIL-FORTSCHRITT
            # ==================================================

            if ready:

                progress = min(
                    1.0,
                    stable_frames
                    / STABLE_FRAMES_REQUIRED
                )


                bar_width = int(
                    WIDTH
                    * 0.42
                )

                x0 = int(
                    (
                        WIDTH
                        - bar_width
                    )
                    / 2
                )

                y0 = 230


                cv2.rectangle(
                    frame,
                    (
                        x0,
                        y0
                    ),
                    (
                        x0 + bar_width,
                        y0 + 14
                    ),
                    (
                        100,
                        100,
                        100
                    ),
                    2
                )


                cv2.rectangle(
                    frame,
                    (
                        x0,
                        y0
                    ),
                    (
                        x0
                        + int(
                            bar_width
                            * progress
                        ),
                        y0 + 14
                    ),
                    (
                        0,
                        255,
                        0
                    ),
                    -1
                )


        # ==================================================
        # DISPLAY
        # ==================================================

        cv2.putText(
            frame,
            (
                "M = Lasermaske | R = Neustart | Q = Ende"
            ),
            (
                30,
                frame.shape[0] - 30
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (
                255,
                255,
                255
            ),
            2,
            cv2.LINE_AA
        )


        display = cv2.resize(
            frame,
            (
                DISPLAY_WIDTH,
                DISPLAY_HEIGHT
            )
        )


        cv2.imshow(
            "3D Scanner - Auto Laser Calibration",
            display
        )


        if show_mask:

            mask_display = cv2.resize(
                laser_mask,
                (
                    DISPLAY_WIDTH,
                    DISPLAY_HEIGHT
                )
            )


            cv2.imshow(
                "Laser Mask",
                mask_display
            )


        key = (
            cv2.waitKey(
                1
            )
            & 0xFF
        )


        if (
            key == ord(
                "q"
            )
            or
            key == 27
        ):

            break


        if key == ord(
            "m"
        ):

            show_mask = (
                not show_mask
            )


            if not show_mask:

                try:

                    cv2.destroyWindow(
                        "Laser Mask"
                    )

                except Exception:

                    pass


        if key == ord(
            "r"
        ):

            collected_points = []

            pose_records = []

            target_index = 0

            stable_frames = 0

            calibration_done = False

            last_capture_time = 0.0

            flash_until = 0.0


            print()
            print(
                "Laser-Kalibrierung zurueckgesetzt."
            )


    cap.release()

    cv2.destroyAllWindows()


    print()
    print(
        "Laser-Kalibrierung beendet."
    )


if __name__ == "__main__":

    main()