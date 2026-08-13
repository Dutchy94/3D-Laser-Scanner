import cv2
import json
import time
import ctypes
import threading
import queue
import numpy as np
import open3d as o3d

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

LASER_CALIBRATION = (
    ROOT
    / "output"
    / "calibration"
    / "laser_calibration.json"
)

SCAN_DIR = (
    ROOT
    / "output"
    / "scans"
)

SCAN_DIR.mkdir(
    parents=True,
    exist_ok=True
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
# TRACKING
# ==========================================================

MIN_CHARUCO_CORNERS = 14

# Reprojection darf nicht zu schlecht sein
MAX_POSE_ERROR_PX = 1.00

# ----------------------------------------------------------
# Anti-Jitter / Bruchfestes Tracking
# ----------------------------------------------------------

# Kleine Bewegungen werden direkt akzeptiert
SOFT_TRANSLATION_JUMP_MM = 6.0
SOFT_ROTATION_JUMP_DEG = 2.0

# Größere Bewegungen müssen in zwei aufeinanderfolgenden
# Frames konsistent sein. Dadurch verschwinden Einzelspikes.
PENDING_CONFIRM_TRANSLATION_MM = 8.0
PENDING_CONFIRM_ROTATION_DEG = 3.0

# Harte physikalische Geschwindigkeitsgrenzen.
# Diese skalieren automatisch mit der realen Framezeit.
MAX_TRANSLATION_SPEED_MM_S = 1800.0
MAX_ROTATION_SPEED_DEG_S = 220.0

# Absolute Notbremse, unabhängig von dt
MAX_TRANSLATION_JUMP_MM = 45.0
MAX_ROTATION_JUMP_DEG = 14.0

# Nach mehreren schlechten Frames Pose komplett neu initialisieren
POSE_RESET_AFTER_BAD_FRAMES = 8

# Wenn nur ein einzelner Frame schlecht ist, bleibt die letzte
# gültige Pose intern erhalten. Es werden aber KEINE Punkte
# dieses ungültigen Frames gespeichert.


# ==========================================================
# LASER
# ==========================================================

MIN_RED = 170
MIN_RED_DIFFERENCE = 75

HSV1_MIN = (0, 110, 130)
HSV1_MAX = (12, 255, 255)

HSV2_MIN = (168, 110, 130)
HSV2_MAX = (179, 255, 255)


# ==========================================================
# REFLEXIONSFILTER
# ==========================================================

MIN_LINE_WIDTH_PX = 1

MAX_LINE_WIDTH_PX = 8

MIN_PEAK_PROMINENCE = 35

SATURATION_VALUE = 250

MAX_SATURATED_RATIO = 0.65


# ==========================================================
# LINIENKONTINUITÄT
# ==========================================================

LASER_X_STEP = 1

MIN_LASER_POINTS = 80

MAX_X_GAP = 12

MAX_Y_JUMP_PX = 35

MIN_SEGMENT_POINTS = 20


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
# MESSBEREICH
# ==========================================================

MIN_Z_MM = 200.0
MAX_Z_MM = 700.0


# ==========================================================
# PUNKTWOLKENFILTER
# ==========================================================

VOXEL_SIZE_MM = 0.5

STAT_NB_NEIGHBORS = 20
STAT_STD_RATIO = 1.5

DISPLAY_MAX_POINTS = 180000


# ==========================================================
# 3D ANSICHT
# ==========================================================

VIEW_TARGET_ALPHA = 0.15

VIEW_EYE_ALPHA = 0.12

VIEW_DISTANCE_FACTOR = 1.15

VIEW_MIN_DISTANCE_MM = 200.0
VIEW_MAX_DISTANCE_MM = 900.0


# ==========================================================
# WINDOWS HOTKEYS
# ==========================================================

VK_SPACE = 0x20
VK_F = 0x46
VK_S = 0x53
VK_R = 0x52
VK_Q = 0x51
VK_ESCAPE = 0x1B


def key_down(vk):

    return bool(
        ctypes.windll.user32.GetAsyncKeyState(
            vk
        )
        & 0x8000
    )


class KeyEdge:

    def __init__(self):

        self.old = {}


    def pressed(
        self,
        vk
    ):

        now = key_down(
            vk
        )

        old = self.old.get(
            vk,
            False
        )

        self.old[vk] = now

        return (
            now
            and not old
        )


# ==========================================================
# KALIBRIERUNGEN LADEN
# ==========================================================

def load_calibration():

    for path in [
        CHARUCO_CONFIG,
        CAMERA_CALIBRATION,
        LASER_CALIBRATION
    ]:

        if not path.exists():

            print()
            print(
                "FEHLER: Datei fehlt:"
            )

            print(
                path
            )

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


    laser_config = json.loads(
        LASER_CALIBRATION.read_text(
            encoding="utf-8"
        )
    )


    camera_matrix = np.array(
        camera_config[
            "camera_matrix"
        ],
        dtype=np.float64
    )


    dist_coeffs = np.array(
        camera_config[
            "dist_coeffs"
        ],
        dtype=np.float64
    )


    plane = laser_config[
        "plane"
    ]


    laser_normal = np.array(
        [
            plane["a"],
            plane["b"],
            plane["c"]
        ],
        dtype=np.float64
    )


    laser_normal /= np.linalg.norm(
        laser_normal
    )


    laser_d = float(
        plane["d"]
    )


    print()
    print(
        "Kalibrierungen geladen"
    )

    print(
        "Camera reprojection:",
        camera_config.get(
            "mean_reprojection_error_px",
            "?"
        ),
        "px"
    )

    print(
        "Laser RMS:",
        laser_config.get(
            "rms_plane_error_mm",
            "?"
        ),
        "mm"
    )


    return (
        charuco_config,
        camera_matrix,
        dist_coeffs,
        laser_normal,
        laser_d
    )


# ==========================================================
# CHARUCO
# ==========================================================

def create_charuco(
    config
):

    dictionary = (
        cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )
    )


    board = cv2.aruco.CharucoBoard(
        (
            int(
                config["squares_x"]
            ),
            int(
                config["squares_y"]
            )
        ),

        float(
            config[
                "square_length_mm"
            ]
        ),

        float(
            config[
                "marker_length_mm"
            ]
        ),

        dictionary
    )


    detector = (
        cv2.aruco.CharucoDetector(
            board
        )
    )


    return (
        board,
        detector
    )


# ==========================================================
# KAMERA
# ==========================================================

def open_camera():

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )


    if not cap.isOpened():

        print(
            "Kamera konnte nicht geöffnet werden"
        )

        return None


    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(
            *"MJPG"
        )
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


    for _ in range(
        10
    ):

        cap.read()


    print()
    print(
        "Kamera:",
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        ),
        "x",
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )


    return cap


# ==========================================================
# KAMERAPOSITION IM BOARD-KOORDINATENSYSTEM
# ==========================================================

def get_camera_position_world(
    rvec,
    tvec
):

    R, _ = cv2.Rodrigues(
        rvec
    )


    return (
        -R.T
        @ tvec.reshape(
            3
        )
    )


# ==========================================================
# ROTATIONSDIFFERENZ
# ==========================================================

def rotation_difference_deg(
    rvec1,
    rvec2
):

    R1, _ = cv2.Rodrigues(
        rvec1
    )

    R2, _ = cv2.Rodrigues(
        rvec2
    )


    relative = (
        R2
        @ R1.T
    )


    value = (
        np.trace(
            relative
        )
        - 1.0
    ) / 2.0


    value = np.clip(
        value,
        -1.0,
        1.0
    )


    angle = np.arccos(
        value
    )


    return float(
        np.degrees(
            angle
        )
    )


# ==========================================================
# POSE TRACKER - ROBUST / ANTI-Z-JITTER
# ==========================================================

class PoseTracker:

    def __init__(
        self,
        board,
        camera_matrix,
        dist_coeffs
    ):

        self.board = board

        self.camera_matrix = (
            camera_matrix
        )

        self.dist_coeffs = (
            dist_coeffs
        )

        self.previous_rvec = None
        self.previous_tvec = None

        self.previous_camera_position = None
        self.previous_time = None

        self.bad_frames = 0

        # Verdächtige größere Bewegung wird erst akzeptiert,
        # wenn der nächste Frame dieselbe Bewegung bestätigt.
        self.pending_rvec = None
        self.pending_tvec = None
        self.pending_camera_position = None
        self.pending_time = None

        self.last_status = "INIT"


    def reset(
        self
    ):

        self.previous_rvec = None
        self.previous_tvec = None

        self.previous_camera_position = None
        self.previous_time = None

        self.pending_rvec = None
        self.pending_tvec = None
        self.pending_camera_position = None
        self.pending_time = None

        self.bad_frames = 0

        self.last_status = "RESET"


    def clear_pending(
        self
    ):

        self.pending_rvec = None
        self.pending_tvec = None
        self.pending_camera_position = None
        self.pending_time = None


    def bad_frame(
        self,
        status="BAD"
    ):

        self.bad_frames += 1

        self.last_status = status

        if (
            self.bad_frames
            >= POSE_RESET_AFTER_BAD_FRAMES
        ):

            self.reset()


    def _solve_pose(
        self,
        object_points,
        image_points
    ):

        try:

            if (
                self.previous_rvec is None
                or
                self.previous_tvec is None
            ):

                success, rvec, tvec = cv2.solvePnP(
                    object_points,
                    image_points,
                    self.camera_matrix,
                    self.dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )

            else:

                success, rvec, tvec = cv2.solvePnP(
                    object_points,
                    image_points,
                    self.camera_matrix,
                    self.dist_coeffs,
                    rvec=self.previous_rvec.copy(),
                    tvec=self.previous_tvec.copy(),
                    useExtrinsicGuess=True,
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
                self.camera_matrix,
                self.dist_coeffs,
                rvec,
                tvec
            )

        except cv2.error:

            pass


        return (
            rvec,
            tvec
        )


    def _pose_quality(
        self,
        object_points,
        image_points,
        rvec,
        tvec
    ):

        try:

            projected, _ = cv2.projectPoints(
                object_points,
                rvec,
                tvec,
                self.camera_matrix,
                self.dist_coeffs
            )

        except cv2.error:

            return None


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


        median_error = float(
            np.median(
                errors
            )
        )


        max_error = float(
            np.max(
                errors
            )
        )


        return (
            mean_error,
            median_error,
            max_error
        )


    def _accept_pose(
        self,
        rvec,
        tvec,
        camera_position,
        now,
        mean_error,
        median_error,
        max_error,
        translation_jump,
        rotation_jump,
        translation_speed,
        rotation_speed,
        status
    ):

        self.previous_rvec = (
            rvec.copy()
        )

        self.previous_tvec = (
            tvec.copy()
        )

        self.previous_camera_position = (
            camera_position.copy()
        )

        self.previous_time = now

        self.bad_frames = 0

        self.clear_pending()

        self.last_status = status


        return {

            "rvec":
                rvec,

            "tvec":
                tvec,

            "error":
                mean_error,

            "median_error":
                median_error,

            "max_error":
                max_error,

            "camera_position":
                camera_position,

            "translation_jump":
                translation_jump,

            "rotation_jump":
                rotation_jump,

            "translation_speed":
                translation_speed,

            "rotation_speed":
                rotation_speed,

            "status":
                status
        }


    def estimate(
        self,
        corners,
        ids
    ):

        now = time.perf_counter()


        # ======================================================
        # EINGANGSDATEN PRÜFEN
        # ======================================================

        if (
            corners is None
            or
            ids is None
        ):

            self.bad_frame(
                "NO_BOARD"
            )

            return None


        if (
            len(corners)
            != len(ids)
        ):

            self.bad_frame(
                "CORNER_ID_MISMATCH"
            )

            return None


        if (
            len(ids)
            < MIN_CHARUCO_CORNERS
        ):

            self.bad_frame(
                "TOO_FEW_CORNERS"
            )

            return None


        try:

            (
                object_points,
                image_points
            ) = self.board.matchImagePoints(
                corners,
                ids
            )

        except Exception:

            self.bad_frame(
                "MATCH_FAILED"
            )

            return None


        if (
            object_points is None
            or
            image_points is None
        ):

            self.bad_frame(
                "NO_MATCH_POINTS"
            )

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

            self.bad_frame(
                "TOO_FEW_MATCH_POINTS"
            )

            return None


        # ======================================================
        # POSE LÖSEN
        # ======================================================

        solved = self._solve_pose(
            object_points,
            image_points
        )


        if solved is None:

            self.bad_frame(
                "PNP_FAILED"
            )

            return None


        rvec, tvec = solved


        # ======================================================
        # REPROJEKTIONSQUALITÄT
        # ======================================================

        quality = self._pose_quality(
            object_points,
            image_points,
            rvec,
            tvec
        )


        if quality is None:

            self.bad_frame(
                "PROJECT_FAILED"
            )

            return None


        (
            mean_error,
            median_error,
            max_error
        ) = quality


        if (
            mean_error
            > MAX_POSE_ERROR_PX
        ):

            self.bad_frame(
                "REPROJECTION_HIGH"
            )

            return None


        # ======================================================
        # BOARD MUSS VOR DER KAMERA LIEGEN
        # ======================================================

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
            points_camera[:, 2]
            <= 0
        ):

            self.bad_frame(
                "BOARD_BEHIND_CAMERA"
            )

            return None


        camera_position = (
            get_camera_position_world(
                rvec,
                tvec
            )
        )


        # ======================================================
        # ERSTE POSE DIREKT AKZEPTIEREN
        # ======================================================

        if (
            self.previous_rvec is None
            or
            self.previous_tvec is None
            or
            self.previous_camera_position is None
            or
            self.previous_time is None
        ):

            return self._accept_pose(
                rvec,
                tvec,
                camera_position,
                now,
                mean_error,
                median_error,
                max_error,
                0.0,
                0.0,
                0.0,
                0.0,
                "INIT_OK"
            )


        # ======================================================
        # FRAME-ZU-FRAME-BEWEGUNG
        # ======================================================

        dt = max(
            now
            - self.previous_time,
            1.0 / 120.0
        )


        translation_jump = float(
            np.linalg.norm(
                camera_position
                - self.previous_camera_position
            )
        )


        rotation_jump = (
            rotation_difference_deg(
                self.previous_rvec,
                rvec
            )
        )


        translation_speed = (
            translation_jump
            / dt
        )


        rotation_speed = (
            rotation_jump
            / dt
        )


        # ======================================================
        # HARTE NOTBREMSE
        # ======================================================

        if (
            translation_jump
            > MAX_TRANSLATION_JUMP_MM
            or
            rotation_jump
            > MAX_ROTATION_JUMP_DEG
            or
            translation_speed
            > MAX_TRANSLATION_SPEED_MM_S
            or
            rotation_speed
            > MAX_ROTATION_SPEED_DEG_S
        ):

            self.clear_pending()

            self.bad_frame(
                "HARD_JUMP_REJECTED"
            )

            return None


        # ======================================================
        # KLEINE / NORMALE BEWEGUNG
        #
        # Direkt akzeptieren -> kein Filter-Lag.
        # ======================================================

        if (
            translation_jump
            <= SOFT_TRANSLATION_JUMP_MM
            and
            rotation_jump
            <= SOFT_ROTATION_JUMP_DEG
        ):

            return self._accept_pose(
                rvec,
                tvec,
                camera_position,
                now,
                mean_error,
                median_error,
                max_error,
                translation_jump,
                rotation_jump,
                translation_speed,
                rotation_speed,
                "TRACK_OK"
            )


        # ======================================================
        # GRÖSSERE, ABER PLAUSIBLE BEWEGUNG
        #
        # Ein einzelner Frame kann ein Pose-Spike sein.
        # Deshalb erst merken, dann im nächsten Frame bestätigen.
        # ======================================================

        if (
            self.pending_rvec is None
            or
            self.pending_tvec is None
            or
            self.pending_camera_position is None
        ):

            self.pending_rvec = (
                rvec.copy()
            )

            self.pending_tvec = (
                tvec.copy()
            )

            self.pending_camera_position = (
                camera_position.copy()
            )

            self.pending_time = now

            self.last_status = (
                "WAIT_CONFIRM"
            )

            # Wichtig:
            # Dieser Frame wird NICHT in die Cloud geschrieben.
            return None


        # ======================================================
        # ZWEITER FRAME MUSS DIE GROSSE BEWEGUNG BESTÄTIGEN
        # ======================================================

        pending_translation = float(
            np.linalg.norm(
                camera_position
                - self.pending_camera_position
            )
        )


        pending_rotation = (
            rotation_difference_deg(
                self.pending_rvec,
                rvec
            )
        )


        if (
            pending_translation
            <= PENDING_CONFIRM_TRANSLATION_MM
            and
            pending_rotation
            <= PENDING_CONFIRM_ROTATION_DEG
        ):

            return self._accept_pose(
                rvec,
                tvec,
                camera_position,
                now,
                mean_error,
                median_error,
                max_error,
                translation_jump,
                rotation_jump,
                translation_speed,
                rotation_speed,
                "MOVE_CONFIRMED"
            )


        # Der zweite Frame bestätigt den ersten nicht:
        # Wahrscheinlich war mindestens einer ein Ausreißer.
        # Neuen Kandidaten als pending übernehmen.

        self.pending_rvec = (
            rvec.copy()
        )

        self.pending_tvec = (
            tvec.copy()
        )

        self.pending_camera_position = (
            camera_position.copy()
        )

        self.pending_time = now

        self.last_status = (
            "WAIT_CONFIRM"
        )

        return None


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
# WRAPPER FUER LIVE-SCANNER
# ==========================================================

def detect_laser(frame):
    return detect_laser_profile_center(
        frame,
        board_mask=None,
        x_step=LASER_X_STEP,
        min_points=MIN_LASER_POINTS,
        max_x_gap=MAX_X_GAP,
        max_y_jump_px=MAX_Y_JUMP_PX,
        min_segment_points=MIN_SEGMENT_POINTS
    )


# ==========================================================
# LASERPIXEL -> KAMERA XYZ
# ==========================================================

def laser_pixels_to_camera_points(
    pixels,
    camera_matrix,
    dist_coeffs,
    laser_normal,
    laser_d
):

    if len(
        pixels
    ) == 0:

        return (
            np.empty(
                (0, 3),
                dtype=np.float64
            ),

            np.empty(
                (0, 2),
                dtype=np.float32
            )
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


    points = []

    valid_pixels = []


    for i, p in enumerate(
        undistorted
    ):

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
                laser_normal,
                ray
            )
        )


        if abs(
            denominator
        ) < 1e-9:

            continue


        distance = (
            -laser_d
            / denominator
        )


        if distance <= 0:

            continue


        point = (
            ray
            * distance
        )


        if (
            point[2]
            < MIN_Z_MM
            or
            point[2]
            > MAX_Z_MM
        ):

            continue


        points.append(
            point
        )


        valid_pixels.append(
            pixels[i]
        )


    return (
        np.asarray(
            points,
            dtype=np.float64
        ),

        np.asarray(
            valid_pixels,
            dtype=np.float32
        )
    )


# ==========================================================
# KAMERA -> WELT
# ==========================================================

def camera_to_world(
    camera_points,
    rvec,
    tvec
):

    if len(
        camera_points
    ) == 0:

        return np.empty(
            (0, 3),
            dtype=np.float64
        )


    R, _ = cv2.Rodrigues(
        rvec
    )


    t = tvec.reshape(
        1,
        3
    )


    return (
        R.T
        @
        (
            camera_points
            - t
        ).T
    ).T


# ==========================================================
# LASERPUNKT AN KAMERAMITTE
# ==========================================================

def get_center_laser_point(
    pixels,
    world_points,
    camera_center_x
):

    if (
        pixels is None
        or
        world_points is None
        or
        len(pixels) == 0
        or
        len(world_points) == 0
    ):

        return None


    n = min(
        len(pixels),
        len(world_points)
    )


    pixels = pixels[
        :n
    ]

    world_points = world_points[
        :n
    ]


    index = int(
        np.argmin(
            np.abs(
                pixels[:, 0]
                - camera_center_x
            )
        )
    )


    return (
        world_points[
            index
        ].copy()
    )


# ==========================================================
# LIVE-ENTFERNUNG AN KAMERAMITTE
# ==========================================================

def get_center_laser_measurement(
    pixels,
    camera_points,
    camera_center_x
):

    if (
        pixels is None
        or
        camera_points is None
        or
        len(pixels) == 0
        or
        len(camera_points) == 0
    ):

        return None


    n = min(
        len(pixels),
        len(camera_points)
    )


    pixels = pixels[
        :n
    ]

    camera_points = camera_points[
        :n
    ]


    index = int(
        np.argmin(
            np.abs(
                pixels[:, 0]
                - camera_center_x
            )
        )
    )


    pixel = pixels[
        index
    ].copy()


    point = camera_points[
        index
    ].copy()


    distance_mm = float(
        np.linalg.norm(
            point
        )
    )


    z_mm = float(
        point[2]
    )


    return {

        "pixel":
            pixel,

        "point":
            point,

        "distance_mm":
            distance_mm,

        "z_mm":
            z_mm
    }


# ==========================================================
# LOOK AT MATRIX
# ==========================================================

def look_at_matrix(
    eye,
    target,
    up
):

    eye = np.asarray(
        eye,
        dtype=np.float64
    )

    target = np.asarray(
        target,
        dtype=np.float64
    )

    up = np.asarray(
        up,
        dtype=np.float64
    )


    z = (
        eye
        - target
    )


    norm_z = np.linalg.norm(
        z
    )


    if norm_z < 1e-9:

        return np.eye(
            4
        )


    z /= norm_z


    x = np.cross(
        up,
        z
    )


    if (
        np.linalg.norm(
            x
        ) < 1e-6
    ):

        up = np.array(
            [
                0.0,
                1.0,
                0.0
            ],
            dtype=np.float64
        )


        x = np.cross(
            up,
            z
        )


    x /= np.linalg.norm(
        x
    )


    y = np.cross(
        z,
        x
    )


    R = np.vstack(
        [
            x,
            y,
            z
        ]
    )


    t = (
        -R
        @ eye
    )


    extrinsic = np.eye(
        4,
        dtype=np.float64
    )


    extrinsic[
        :3,
        :3
    ] = R


    extrinsic[
        :3,
        3
    ] = t


    return extrinsic


# ==========================================================
# 3D ANSICHT FOLGT SCANNER
# ==========================================================

def follow_scanner_view(
    vis,
    target_point,
    scanner_position,
    old_target,
    old_eye
):

    if (
        target_point is None
        or
        scanner_position is None
    ):

        return (
            old_target,
            old_eye
        )


    scan_direction = (
        target_point
        - scanner_position
    )


    scan_distance = np.linalg.norm(
        scan_direction
    )


    if scan_distance < 1.0:

        return (
            old_target,
            old_eye
        )


    scan_direction /= (
        scan_distance
    )


    table_distance = abs(
        scanner_position[2]
    )


    if (
        table_distance
        < 100
    ):

        table_distance = (
            scan_distance
        )


    virtual_distance = (
        table_distance
        * VIEW_DISTANCE_FACTOR
    )


    virtual_distance = float(
        np.clip(
            virtual_distance,
            VIEW_MIN_DISTANCE_MM,
            VIEW_MAX_DISTANCE_MM
        )
    )


    new_eye = (
        target_point
        - scan_direction
        * virtual_distance
    )


    new_target = (
        target_point.copy()
    )


    if (
        old_target is None
        or
        old_eye is None
    ):

        smooth_target = (
            new_target
        )

        smooth_eye = (
            new_eye
        )

    else:

        smooth_target = (
            old_target
            * (
                1.0
                - VIEW_TARGET_ALPHA
            )
            +
            new_target
            * VIEW_TARGET_ALPHA
        )


        smooth_eye = (
            old_eye
            * (
                1.0
                - VIEW_EYE_ALPHA
            )
            +
            new_eye
            * VIEW_EYE_ALPHA
        )


    control = (
        vis.get_view_control()
    )


    try:

        parameters = (
            control.convert_to_pinhole_camera_parameters()
        )


        parameters.extrinsic = (
            look_at_matrix(
                smooth_eye,
                smooth_target,
                [
                    0.0,
                    0.0,
                    1.0
                ]
            )
        )


        control.convert_from_pinhole_camera_parameters(
            parameters,
            allow_arbitrary=True
        )


    except Exception:

        pass


    return (
        smooth_target,
        smooth_eye
    )


# ==========================================================
# CLOUD FILTER
# ==========================================================

def filter_cloud(
    points
):

    if len(
        points
    ) < 30:

        return points


    cloud = (
        o3d.geometry.PointCloud()
    )


    cloud.points = (
        o3d.utility.Vector3dVector(
            points
        )
    )


    cloud = (
        cloud.voxel_down_sample(
            voxel_size=VOXEL_SIZE_MM
        )
    )


    if (
        len(
            cloud.points
        )
        > STAT_NB_NEIGHBORS
    ):

        cloud, _ = (
            cloud.remove_statistical_outlier(
                nb_neighbors=STAT_NB_NEIGHBORS,
                std_ratio=STAT_STD_RATIO
            )
        )


    return np.asarray(
        cloud.points
    )


# ==========================================================
# SPEICHERN
# ==========================================================

def save_cloud(
    points,
    suffix=""
):

    if len(
        points
    ) == 0:

        return


    stamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )


    cloud = (
        o3d.geometry.PointCloud()
    )


    cloud.points = (
        o3d.utility.Vector3dVector(
            points
        )
    )


    ply_path = (
        SCAN_DIR
        /
        f"scan_{stamp}{suffix}.ply"
    )


    o3d.io.write_point_cloud(
        str(
            ply_path
        ),
        cloud
    )


    csv_path = (
        SCAN_DIR
        /
        f"scan_{stamp}{suffix}.csv"
    )


    np.savetxt(
        csv_path,
        points,
        delimiter=";",
        header="X_mm;Y_mm;Z_mm",
        comments=""
    )


    print(
        "Gespeichert:",
        ply_path
    )


# ==========================================================
# WORKER THREAD
# ==========================================================

class ScannerWorker(
    threading.Thread
):

    def __init__(
        self,
        output_queue,
        stop_event,
        board,
        detector,
        camera_matrix,
        dist_coeffs,
        laser_normal,
        laser_d
    ):

        super().__init__(
            daemon=True
        )


        self.output_queue = (
            output_queue
        )

        self.stop_event = (
            stop_event
        )

        self.board = (
            board
        )

        self.detector = (
            detector
        )

        self.camera_matrix = (
            camera_matrix
        )

        self.dist_coeffs = (
            dist_coeffs
        )

        self.laser_normal = (
            laser_normal
        )

        self.laser_d = (
            laser_d
        )


        self.pose_tracker = (
            PoseTracker(
                board,
                camera_matrix,
                dist_coeffs
            )
        )


    def put_latest(
        self,
        data
    ):

        try:

            while True:

                self.output_queue.get_nowait()

        except queue.Empty:

            pass


        try:

            self.output_queue.put_nowait(
                data
            )

        except queue.Full:

            pass


    def run(
        self
    ):

        cap = open_camera()


        if cap is None:

            return


        frame_counter = 0

        frame_id = 0

        fps_start = (
            time.perf_counter()
        )

        real_fps = 0.0


        while not self.stop_event.is_set():

            success, raw = (
                cap.read()
            )


            if not success:

                continue


            frame_id += 1

            frame_counter += 1


            now = (
                time.perf_counter()
            )


            if (
                now - fps_start
                >= 1.0
            ):

                real_fps = (
                    frame_counter
                    /
                    (
                        now
                        - fps_start
                    )
                )


                frame_counter = 0

                fps_start = (
                    now
                )


            # ==================================================
            # CHARUCO
            # ==================================================

            gray = cv2.cvtColor(
                raw,
                cv2.COLOR_BGR2GRAY
            )


            try:

                (
                    charuco_corners,
                    charuco_ids,
                    marker_corners,
                    marker_ids
                ) = self.detector.detectBoard(
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


            # ==================================================
            # POSE
            # ==================================================

            pose = (
                self.pose_tracker.estimate(
                    charuco_corners,
                    charuco_ids
                )
            )


            tracking_ok = (
                pose is not None
            )


            rvec = None

            tvec = None

            pose_error = None

            scanner_position = None

            translation_jump = None

            rotation_jump = None

            translation_speed = None
            rotation_speed = None

            pose_status = (
                self.pose_tracker.last_status
            )


            if tracking_ok:

                rvec = pose[
                    "rvec"
                ]

                tvec = pose[
                    "tvec"
                ]

                pose_error = pose[
                    "error"
                ]

                scanner_position = pose[
                    "camera_position"
                ]

                translation_jump = pose[
                    "translation_jump"
                ]

                rotation_jump = pose[
                    "rotation_jump"
                ]

                translation_speed = pose[
                    "translation_speed"
                ]

                rotation_speed = pose[
                    "rotation_speed"
                ]

                pose_status = pose[
                    "status"
                ]


            # ==================================================
            # LASER
            # ==================================================

            (
                laser_pixels,
                laser_mask,
                laser_ok,
                laser_diagnostics
            ) = detect_laser(
                raw
            )


            # ==================================================
            # XYZ
            # ==================================================

            camera_points = np.empty(
                (0, 3),
                dtype=np.float64
            )


            world_points = np.empty(
                (0, 3),
                dtype=np.float64
            )


            valid_pixels = np.empty(
                (0, 2),
                dtype=np.float32
            )


            if (
                tracking_ok
                and
                laser_ok
            ):

                (
                    camera_points,
                    valid_pixels
                ) = laser_pixels_to_camera_points(
                    laser_pixels,
                    self.camera_matrix,
                    self.dist_coeffs,
                    self.laser_normal,
                    self.laser_d
                )


                world_points = camera_to_world(
                    camera_points,
                    rvec,
                    tvec
                )


            packet = {

                "frame_id":
                    frame_id,

                "frame":
                    raw,

                "fps":
                    real_fps,

                "charuco_corners":
                    charuco_corners,

                "charuco_ids":
                    charuco_ids,

                "marker_corners":
                    marker_corners,

                "marker_ids":
                    marker_ids,

                "charuco_count":
                    charuco_count,

                "tracking_ok":
                    tracking_ok,

                "pose_error":
                    pose_error,

                "translation_jump":
                    translation_jump,

                "rotation_jump":
                    rotation_jump,

                "translation_speed":
                    translation_speed,

                "rotation_speed":
                    rotation_speed,

                "pose_status":
                    pose_status,

                "rvec":
                    rvec,

                "tvec":
                    tvec,

                "scanner_position":
                    scanner_position,

                "laser_pixels":
                    laser_pixels,

                "valid_pixels":
                    valid_pixels,

                "laser_mask":
                    laser_mask,

                "laser_ok":
                    laser_ok,

                "laser_diagnostics":
                    laser_diagnostics,

                "camera_points":
                    camera_points,

                "world_points":
                    world_points
            }


            self.put_latest(
                packet
            )


        cap.release()


# ==========================================================
# MAIN
# ==========================================================

def main():

    loaded = (
        load_calibration()
    )


    if loaded is None:

        return


    (
        charuco_config,
        camera_matrix,
        dist_coeffs,
        laser_normal,
        laser_d
    ) = loaded


    (
        board,
        detector
    ) = create_charuco(
        charuco_config
    )


    # ======================================================
    # WORKER
    # ======================================================

    output_queue = queue.Queue(
        maxsize=1
    )


    stop_event = (
        threading.Event()
    )


    worker = ScannerWorker(
        output_queue,
        stop_event,
        board,
        detector,
        camera_matrix,
        dist_coeffs,
        laser_normal,
        laser_d
    )


    worker.start()


    # ======================================================
    # OPEN3D
    # ======================================================

    vis = (
        o3d.visualization.Visualizer()
    )


    vis.create_window(
        window_name="3D Scanner - Live",
        width=900,
        height=700
    )


    cloud = (
        o3d.geometry.PointCloud()
    )


    vis.add_geometry(
        cloud
    )


    axis = (
        o3d.geometry.TriangleMesh
        .create_coordinate_frame(
            size=50.0
        )
    )


    vis.add_geometry(
        axis
    )


    # ======================================================
    # STATUS
    # ======================================================

    key_edge = (
        KeyEdge()
    )


    all_points = []

    filtered_display = False

    filtered_cache = None

    filtered_cache_count = 0


    view_target = None

    view_eye = None


    was_recording = False


    last_packet = None

    last_recorded_frame_id = -1


    # ======================================================
    # LETZTE GÜLTIGE OBJEKTENTFERNUNG
    # ======================================================

    last_object_distance_mm = None

    last_object_z_mm = None

    last_object_pixel = None


    try:

        while True:

            got_new_packet = False


            try:

                packet = (
                    output_queue.get(
                        timeout=0.005
                    )
                )


                last_packet = (
                    packet
                )

                got_new_packet = (
                    True
                )


            except queue.Empty:

                packet = (
                    last_packet
                )


            # ==================================================
            # HOTKEYS
            # ==================================================

            recording_requested = (
                key_down(
                    VK_SPACE
                )
            )


            if (
                key_edge.pressed(
                    VK_Q
                )
                or
                key_edge.pressed(
                    VK_ESCAPE
                )
            ):

                break


            if key_edge.pressed(
                VK_F
            ):

                filtered_display = (
                    not filtered_display
                )


                print(
                    "Filter:",
                    filtered_display
                )


            if key_edge.pressed(
                VK_R
            ):

                all_points = []

                filtered_cache = None

                filtered_cache_count = 0

                view_target = None

                view_eye = None

                last_recorded_frame_id = -1

                worker.pose_tracker.reset()


                cloud.clear()


                vis.update_geometry(
                    cloud
                )


                print()
                print(
                    "Cloud gelöscht"
                )


            if key_edge.pressed(
                VK_S
            ):

                if len(
                    all_points
                ) > 0:

                    combined = (
                        np.vstack(
                            all_points
                        )
                    )


                    save_cloud(
                        combined,
                        "_raw"
                    )


                    filtered = (
                        filter_cloud(
                            combined
                        )
                    )


                    save_cloud(
                        filtered,
                        "_filtered"
                    )


            if packet is None:

                vis.poll_events()

                vis.update_renderer()

                continue


            frame = packet[
                "frame"
            ].copy()


            tracking_ok = packet[
                "tracking_ok"
            ]


            laser_ok = packet[
                "laser_ok"
            ]


            world_points = packet[
                "world_points"
            ]


            camera_points = packet[
                "camera_points"
            ]


            valid_pixels = packet[
                "valid_pixels"
            ]


            frame_id = packet[
                "frame_id"
            ]


            # ==================================================
            # LIVE DISTANZ AN KAMERAMITTE
            # ==================================================

            center_measurement = (
                get_center_laser_measurement(
                    valid_pixels,
                    camera_points,
                    camera_matrix[
                        0,
                        2
                    ]
                )
            )


            if (
                center_measurement
                is not None
            ):

                last_object_distance_mm = (
                    center_measurement[
                        "distance_mm"
                    ]
                )


                last_object_z_mm = (
                    center_measurement[
                        "z_mm"
                    ]
                )


                last_object_pixel = (
                    center_measurement[
                        "pixel"
                    ].copy()
                )


            # ==================================================
            # AUFNAHME
            # ==================================================

            recording = (
                recording_requested
                and
                tracking_ok
                and
                laser_ok
                and
                len(
                    world_points
                ) > 0
            )


            if (
                recording
                and
                got_new_packet
                and
                frame_id
                != last_recorded_frame_id
            ):

                all_points.append(
                    world_points.copy()
                )


                last_recorded_frame_id = (
                    frame_id
                )


                center_point = (
                    get_center_laser_point(
                        valid_pixels,
                        world_points,
                        camera_matrix[
                            0,
                            2
                        ]
                    )
                )


                scanner_position = packet[
                    "scanner_position"
                ]


                if (
                    center_point is not None
                    and
                    scanner_position is not None
                ):

                    (
                        view_target,
                        view_eye
                    ) = follow_scanner_view(
                        vis,
                        center_point,
                        scanner_position,
                        view_target,
                        view_eye
                    )


            # ==================================================
            # START / STOP
            # ==================================================

            if (
                recording
                and
                not was_recording
            ):

                print()
                print(
                    "SCAN START"
                )


            if (
                not recording
                and
                was_recording
            ):

                print()
                print(
                    "SCAN STOP"
                )


            was_recording = (
                recording
            )


            # ==================================================
            # MARKER ZEICHNEN
            # ==================================================

            marker_ids = packet[
                "marker_ids"
            ]


            marker_corners = packet[
                "marker_corners"
            ]


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


            charuco_ids = packet[
                "charuco_ids"
            ]


            charuco_corners = packet[
                "charuco_corners"
            ]


            if (
                charuco_ids is not None
                and
                charuco_corners is not None
                and
                len(
                    charuco_ids
                )
                ==
                len(
                    charuco_corners
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


            # ==================================================
            # LASER ZEICHNEN
            # ==================================================

            laser_pixels = packet[
                "laser_pixels"
            ]


            for p in laser_pixels[
                ::4
            ]:

                cv2.circle(
                    frame,
                    (
                        int(
                            p[0]
                        ),
                        int(
                            p[1]
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


            # ==================================================
            # KAMERAMITTE
            # ==================================================

            cx = int(
                camera_matrix[
                    0,
                    2
                ]
            )


            cv2.line(
                frame,
                (
                    cx,
                    0
                ),
                (
                    cx,
                    frame.shape[0]
                ),
                (
                    255,
                    0,
                    0
                ),
                1
            )


            # ==================================================
            # AKTUELLER DISTANZPUNKT
            # ==================================================

            if (
                last_object_pixel
                is not None
            ):

                px = int(
                    last_object_pixel[
                        0
                    ]
                )

                py = int(
                    last_object_pixel[
                        1
                    ]
                )


                cv2.circle(
                    frame,
                    (
                        px,
                        py
                    ),
                    8,
                    (
                        0,
                        255,
                        255
                    ),
                    2
                )


                cv2.line(
                    frame,
                    (
                        px - 12,
                        py
                    ),
                    (
                        px + 12,
                        py
                    ),
                    (
                        0,
                        255,
                        255
                    ),
                    1
                )


            # ==================================================
            # LASERPROFIL-DIAGNOSE
            # ==================================================

            laser_diag = packet.get(
                "laser_diagnostics",
                {}
            )

            mean_width = laser_diag.get(
                "mean_width_px"
            )

            mean_quality = laser_diag.get(
                "mean_quality"
            )

            if (
                mean_width is not None
                and mean_quality is not None
            ):

                cv2.putText(
                    frame,
                    (
                        f"Laserbreite: {mean_width:.2f} px | "
                        f"Profilqualitaet: {mean_quality:.2f}"
                    ),
                    (30, 275),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (255, 255, 0),
                    2
                )


            # ==================================================
            # CLOUD
            # ==================================================

            total_points = sum(
                len(
                    p
                )
                for p in all_points
            )


            if len(
                all_points
            ) > 0:

                combined = (
                    np.vstack(
                        all_points
                    )
                )


                if filtered_display:

                    if (
                        filtered_cache is None
                        or
                        total_points
                        - filtered_cache_count
                        > 3000
                    ):

                        filtered_cache = (
                            filter_cloud(
                                combined
                            )
                        )


                        filtered_cache_count = (
                            total_points
                        )


                    display_points = (
                        filtered_cache
                    )

                else:

                    display_points = (
                        combined
                    )


                if (
                    len(
                        display_points
                    )
                    > DISPLAY_MAX_POINTS
                ):

                    step = (
                        len(
                            display_points
                        )
                        //
                        DISPLAY_MAX_POINTS
                        + 1
                    )


                    display_points = (
                        display_points[
                            ::step
                        ]
                    )


                cloud.points = (
                    o3d.utility.Vector3dVector(
                        display_points
                    )
                )


                vis.update_geometry(
                    cloud
                )


            # ==================================================
            # STATUS
            # ==================================================

            if tracking_ok:

                tracking_text = (
                    f"TRACKING OK "
                    f"({packet['charuco_count']}) "
                    f"{packet.get('pose_status', '')}"
                )

                tracking_color = (
                    0,
                    255,
                    0
                )

            else:

                pose_status = packet.get(
                    "pose_status",
                    ""
                )

                if pose_status == "WAIT_CONFIRM":

                    tracking_text = (
                        f"POSE PRUEFUNG "
                        f"({packet['charuco_count']})"
                    )

                    tracking_color = (
                        0,
                        165,
                        255
                    )

                else:

                    tracking_text = (
                        f"TRACKING NICHT OK "
                        f"({packet['charuco_count']}) "
                        f"{pose_status}"
                    )

                    tracking_color = (
                        0,
                        0,
                        255
                    )


            # ==================================================
            # OVERLAY
            # ==================================================

            cv2.putText(
                frame,
                f"FPS: {packet['fps']:.1f}",
                (
                    30,
                    45
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (
                    0,
                    255,
                    0
                ),
                2
            )


            cv2.putText(
                frame,
                tracking_text,
                (
                    30,
                    85
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                tracking_color,
                2
            )


            if (
                packet[
                    "pose_error"
                ]
                is not None
            ):

                cv2.putText(
                    frame,
                    (
                        f"Pose err: "
                        f"{packet['pose_error']:.3f} px"
                    ),
                    (
                        30,
                        125
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (
                        255,
                        255,
                        0
                    ),
                    2
                )


            if (
                packet[
                    "translation_jump"
                ]
                is not None
            ):

                cv2.putText(
                    frame,
                    (
                        f"dPos: "
                        f"{packet['translation_jump']:.2f} mm"
                    ),
                    (
                        30,
                        165
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (
                        255,
                        255,
                        0
                    ),
                    2
                )


            if (
                packet[
                    "rotation_jump"
                ]
                is not None
            ):

                cv2.putText(
                    frame,
                    (
                        f"dRot: "
                        f"{packet['rotation_jump']:.2f} deg"
                    ),
                    (
                        30,
                        205
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (
                        255,
                        255,
                        0
                    ),
                    2
                )


            if (
                packet.get(
                    "translation_speed"
                )
                is not None
            ):

                cv2.putText(
                    frame,
                    (
                        f"vPose: "
                        f"{packet['translation_speed']:.0f} mm/s | "
                        f"{packet['rotation_speed']:.1f} deg/s"
                    ),
                    (
                        30,
                        235
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (
                        255,
                        255,
                        0
                    ),
                    2
                )


            scanner_position = packet[
                "scanner_position"
            ]


            if (
                scanner_position
                is not None
            ):

                cv2.putText(
                    frame,
                    (
                        f"Cam XYZ: "
                        f"{scanner_position[0]:.1f} "
                        f"{scanner_position[1]:.1f} "
                        f"{scanner_position[2]:.1f}"
                    ),
                    (
                        30,
                        245
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (
                        255,
                        255,
                        0
                    ),
                    2
                )


            # ==================================================
            # NEU: OBJEKTENTFERNUNG
            # ==================================================

            if (
                last_object_distance_mm
                is not None
            ):

                cv2.putText(
                    frame,
                    (
                        f"Objektabstand: "
                        f"{last_object_distance_mm:.1f} mm"
                    ),
                    (
                        30,
                        290
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (
                        0,
                        255,
                        255
                    ),
                    2
                )


            if (
                last_object_z_mm
                is not None
            ):

                cv2.putText(
                    frame,
                    (
                        f"Kamera-Z: "
                        f"{last_object_z_mm:.1f} mm"
                    ),
                    (
                        30,
                        330
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (
                        0,
                        255,
                        255
                    ),
                    2
                )


            cv2.putText(
                frame,
                (
                    f"Laserpunkte: "
                    f"{len(valid_pixels)}"
                ),
                (
                    30,
                    370
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (
                    0,
                    255,
                    0
                )
                if laser_ok
                else
                (
                    0,
                    0,
                    255
                ),
                2
            )


            cv2.putText(
                frame,
                (
                    f"Cloud Punkte: "
                    f"{total_points}"
                ),
                (
                    30,
                    410
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (
                    255,
                    255,
                    0
                ),
                2
            )


            if recording:

                record_text = (
                    "AUFNAHME"
                )

                record_color = (
                    0,
                    0,
                    255
                )

            elif (
                tracking_ok
                and
                laser_ok
            ):

                record_text = (
                    "SPACE HALTEN"
                )

                record_color = (
                    0,
                    255,
                    255
                )

            else:

                record_text = (
                    "NICHT BEREIT"
                )

                record_color = (
                    0,
                    0,
                    255
                )


            cv2.putText(
                frame,
                record_text,
                (
                    30,
                    460
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.05,
                record_color,
                3
            )


            cv2.putText(
                frame,
                (
                    "SPACE Scan | "
                    "F Filter | "
                    "S Save | "
                    "R Reset | "
                    "Q Ende"
                ),
                (
                    30,
                    frame.shape[0] - 35
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (
                    255,
                    255,
                    255
                ),
                2
            )


            # ==================================================
            # GUI
            # ==================================================

            display = cv2.resize(
                frame,
                (
                    DISPLAY_WIDTH,
                    DISPLAY_HEIGHT
                )
            )


            cv2.imshow(
                "3D Scanner",
                display
            )


            cv2.waitKey(
                1
            )


            alive = (
                vis.poll_events()
            )


            vis.update_renderer()


            if not alive:

                break


    finally:

        stop_event.set()


        worker.join(
            timeout=2.0
        )


        cv2.destroyAllWindows()


        vis.destroy_window()


        if len(
            all_points
        ) > 0:

            combined = (
                np.vstack(
                    all_points
                )
            )


            save_cloud(
                combined,
                "_raw"
            )


            filtered = (
                filter_cloud(
                    combined
                )
            )


            save_cloud(
                filtered,
                "_filtered"
            )


if __name__ == "__main__":

    main()