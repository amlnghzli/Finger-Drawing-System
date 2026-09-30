
import cv2
import numpy as np
import mediapipe as mp
import time
import os
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# SETTINGS
# ============================================================

CAMERA_INDEX = 0

MODEL_PATH = "hand_landmarker.task"

BRUSH_THICKNESS = 5
ERASER_SIZE = 20

# Smoothing for finger position
SMOOTHING = 0.55

# Gesture confirmation
GESTURE_CONFIRM_FRAMES = 5

# Color change cooldown
COLOR_COOLDOWN = 1

# Clear all cooldown
CLEAR_COOLDOWN = 1.0


# ============================================================
# COLORS
# OpenCV uses BGR
# ============================================================

COLORS = [
    (255, 255, 255),   # White
    (0, 0, 0),         # Black
    (0, 0, 255),       # Red
    (0, 255, 0),       # Green
    (255, 0, 0)        # Blue
]

COLOR_NAMES = [
    "WHITE",
    "BLACK",
    "RED",
    "GREEN",
    "BLUE"
]

current_color_index = 0
current_color = COLORS[current_color_index]


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    print("ERROR: hand_landmarker.task not found.")
    print()
    print("Expected location:")
    print(os.path.abspath(MODEL_PATH))
    print()
    print("Put hand_landmarker.task in the same folder")
    print("as drawing_finger.py")

    exit()


# ============================================================
# MEDIAPIPE HAND LANDMARKER
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.8,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)

detector = vision.HandLandmarker.create_from_options(
    options
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():

    print("ERROR: Cannot open webcam.")

    detector.close()

    exit()


# ============================================================
# DRAWING DATA
# ============================================================

# Each stroke has:
#
# {
#     "points": [...],
#     "color": (...)
# }
#
# Existing strokes therefore keep their own colors.

strokes = []

current_stroke = []


# ============================================================
# POSITION VARIABLES
# ============================================================

previous_x = None
previous_y = None

smooth_x = None
smooth_y = None


# ============================================================
# TIMESTAMP
# ============================================================

last_timestamp = 0


# ============================================================
# GESTURE STABILITY
# ============================================================

candidate_gesture = "STOP"
candidate_count = 0
confirmed_gesture = "STOP"


# ============================================================
# COOLDOWNS
# ============================================================

last_color_change_time = 0
last_clear_time = 0


# ============================================================
# DISTANCE
# ============================================================

def distance(p1, p2):

    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


# ============================================================
# FINGER UP
# ============================================================

def is_finger_up(landmarks, tip, pip, mcp):

    """
    Check whether a finger is extended upward.

    Example:
        Index:
            tip = 8
            pip = 6
            mcp = 5
    """

    return (
        landmarks[tip].y < landmarks[pip].y
        and
        landmarks[pip].y < landmarks[mcp].y
    )


# ============================================================
# FINGER FOLDED
# ============================================================

def is_finger_folded(landmarks, tip, pip, mcp):

    """
    Check whether a finger is folded.

    Fingertip must be below both PIP and MCP.
    """

    return (
        landmarks[tip].y > landmarks[pip].y
        and
        landmarks[tip].y > landmarks[mcp].y
    )


# ============================================================
# COLOR GESTURE
# ============================================================

def is_color_gesture(landmarks, width, height):

    """
    COLOR:

        Thumb + Index = PINCH

        Middle = UP
        Ring   = UP
        Pinky  = UP

    Gesture:

        🤏 + 3 fingers open
    """

    thumb_tip = (
        int(landmarks[4].x * width),
        int(landmarks[4].y * height)
    )

    index_tip = (
        int(landmarks[8].x * width),
        int(landmarks[8].y * height)
    )

    pinch_distance = distance(
        thumb_tip,
        index_tip
    )

    # Hand-size normalization
    wrist = (
        int(landmarks[0].x * width),
        int(landmarks[0].y * height)
    )

    middle_mcp = (
        int(landmarks[9].x * width),
        int(landmarks[9].y * height)
    )

    hand_size = distance(
        wrist,
        middle_mcp
    )

    if hand_size == 0:

        return False

    normalized_distance = (
        pinch_distance / hand_size
    )

    # Other 3 fingers must be OPEN
    middle_up = is_finger_up(
        landmarks,
        12,
        10,
        9
    )

    ring_up = is_finger_up(
        landmarks,
        16,
        14,
        13
    )

    pinky_up = is_finger_up(
        landmarks,
        20,
        18,
        17
    )

    return (
        normalized_distance < 0.55
        and
        middle_up
        and
        ring_up
        and
        pinky_up
    )


# ============================================================
# DRAW GESTURE
# ============================================================

def is_draw_gesture(landmarks):

    """
    DRAW:

             ☝️

        Index  = UP
        Middle = FOLDED
        Ring   = FOLDED
        Pinky  = FOLDED

    This is intentionally strict.

    Drawing will NOT happen if:
        - Index is folded
        - Middle is open
        - Ring is open
        - Pinky is open
    """

    # --------------------------------------------------------
    # Index must be UP
    # --------------------------------------------------------

    index_up = is_finger_up(
        landmarks,
        8,      # index tip
        6,      # index PIP
        5       # index MCP
    )

    # --------------------------------------------------------
    # Middle must be FOLDED
    # --------------------------------------------------------

    middle_folded = is_finger_folded(
        landmarks,
        12,     # middle tip
        10,     # middle PIP
        9       # middle MCP
    )

    # --------------------------------------------------------
    # Ring must be FOLDED
    # --------------------------------------------------------

    ring_folded = is_finger_folded(
        landmarks,
        16,     # ring tip
        14,     # ring PIP
        13      # ring MCP
    )

    # --------------------------------------------------------
    # Pinky must be FOLDED
    # --------------------------------------------------------

    pinky_folded = is_finger_folded(
        landmarks,
        20,     # pinky tip
        18,     # pinky PIP
        17      # pinky MCP
    )

    # --------------------------------------------------------
    # ALL CONDITIONS MUST BE TRUE
    # --------------------------------------------------------

    return (
        index_up
        and
        middle_folded
        and
        ring_folded
        and
        pinky_folded
    )


# ============================================================
# ERASER GESTURE
# ============================================================

def is_eraser_gesture(landmarks):

    """
    ERASER:

        ✌️

        Index  = UP
        Middle = UP
        Ring   = FOLDED
        Pinky  = FOLDED
    """

    index_up = is_finger_up(
        landmarks,
        8,
        6,
        5
    )

    middle_up = is_finger_up(
        landmarks,
        12,
        10,
        9
    )

    ring_folded = is_finger_folded(
        landmarks,
        16,
        14,
        13
    )

    pinky_folded = is_finger_folded(
        landmarks,
        20,
        18,
        17
    )

    return (
        index_up
        and
        middle_up
        and
        ring_folded
        and
        pinky_folded
    )


# ============================================================
# FIST / PAUSE
# ============================================================

def is_fist_gesture(landmarks):

    """
    PAUSE:

        ✊

        Index  = FOLDED
        Middle = FOLDED
        Ring   = FOLDED
        Pinky  = FOLDED
    """

    index_folded = is_finger_folded(
        landmarks,
        8,
        6,
        5
    )

    middle_folded = is_finger_folded(
        landmarks,
        12,
        10,
        9
    )

    ring_folded = is_finger_folded(
        landmarks,
        16,
        14,
        13
    )

    pinky_folded = is_finger_folded(
        landmarks,
        20,
        18,
        17
    )

    return (
        index_folded
        and
        middle_folded
        and
        ring_folded
        and
        pinky_folded
    )


# ============================================================
# THUMBS DOWN
# ============================================================

def is_thumbs_down(landmarks):

    """
    CLEAR ALL:

        👎

        Thumb  = DOWN
        Index  = FOLDED
        Middle = FOLDED
        Ring   = FOLDED
        Pinky  = FOLDED
    """

    index_folded = is_finger_folded(
        landmarks,
        8,
        6,
        5
    )

    middle_folded = is_finger_folded(
        landmarks,
        12,
        10,
        9
    )

    ring_folded = is_finger_folded(
        landmarks,
        16,
        14,
        13
    )

    pinky_folded = is_finger_folded(
        landmarks,
        20,
        18,
        17
    )

    # Thumb points down
    thumb_down = (
        landmarks[4].y > landmarks[3].y
    )

    return (
        thumb_down
        and
        index_folded
        and
        middle_folded
        and
        ring_folded
        and
        pinky_folded
    )


# ============================================================
# DETECT GESTURE
# ============================================================

def detect_gesture(landmarks, width, height):

    """
    Detect gesture.

    Detection order is important.
    """

    # --------------------------------------------------------
    # COLOR
    # --------------------------------------------------------

    if is_color_gesture(
        landmarks,
        width,
        height
    ):

        return "COLOR"

    # --------------------------------------------------------
    # CLEAR ALL
    # --------------------------------------------------------

    if is_thumbs_down(landmarks):

        return "CLEAR ALL"

    # --------------------------------------------------------
    # ERASER
    # --------------------------------------------------------

    if is_eraser_gesture(landmarks):

        return "ERASER"

    # --------------------------------------------------------
    # DRAW
    # --------------------------------------------------------

    if is_draw_gesture(landmarks):

        return "DRAW"

    # --------------------------------------------------------
    # PAUSE
    # --------------------------------------------------------

    if is_fist_gesture(landmarks):

        return "PAUSE"

    # --------------------------------------------------------
    # NOTHING
    # --------------------------------------------------------

    return "STOP"


# ============================================================
# STABLE GESTURE
# ============================================================

def get_stable_gesture(new_gesture):

    """
    Confirm a gesture only after several consecutive frames.

    This prevents accidental switching between:
        COLOR
        ERASER
        PAUSE
        CLEAR ALL
    """

    global candidate_gesture
    global candidate_count
    global confirmed_gesture

    if new_gesture == candidate_gesture:

        candidate_count += 1

    else:

        candidate_gesture = new_gesture
        candidate_count = 1

    if candidate_count >= GESTURE_CONFIRM_FRAMES:

        confirmed_gesture = candidate_gesture

    return confirmed_gesture


# ============================================================
# DRAW HAND LANDMARKS
# ============================================================

def draw_hand(frame, landmarks):

    height, width, _ = frame.shape

    points = []

    # Convert normalized coordinates to pixels
    for landmark in landmarks:

        x = int(
            landmark.x * width
        )

        y = int(
            landmark.y * height
        )

        points.append(
            (x, y)
        )

    # Hand connections
    connections = [

        # Thumb
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),

        # Index
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),

        # Middle
        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),

        # Ring
        (9, 13),
        (13, 14),
        (14, 15),
        (15, 16),

        # Pinky
        (13, 17),
        (17, 18),
        (18, 19),
        (19, 20),

        # Palm
        (0, 17)
    ]

    # Draw connections
    for start, end in connections:

        cv2.line(
            frame,
            points[start],
            points[end],
            (0, 255, 0),
            2
        )

    # Draw landmarks
    for point in points:

        cv2.circle(
            frame,
            point,
            4,
            (0, 255, 0),
            -1
        )


# ============================================================
# DRAW COMPLETED STROKES
# ============================================================

def draw_strokes(frame):

    """
    Draw all completed strokes.

    Each stroke keeps its original color.
    """

    for stroke in strokes:

        points = stroke["points"]

        color = stroke["color"]

        if len(points) < 2:

            continue

        for i in range(1, len(points)):

            cv2.line(
                frame,
                points[i - 1],
                points[i],
                color,
                BRUSH_THICKNESS,
                cv2.LINE_AA
            )


# ============================================================
# FINISH CURRENT STROKE
# ============================================================

def finish_current_stroke():

    """
    Save current stroke.

    The current color is copied into the stroke.

    Changing color later will NOT change this stroke.
    """

    global current_stroke

    if len(current_stroke) >= 2:

        strokes.append({
            "points": current_stroke.copy(),
            "color": current_color
        })

    current_stroke = []


# ============================================================
# DRAW CURRENT STROKE
# ============================================================

def draw_current_stroke(frame):

    """
    Draw the stroke currently being created.
    """

    if len(current_stroke) < 2:

        return

    for i in range(1, len(current_stroke)):

        cv2.line(
            frame,
            current_stroke[i - 1],
            current_stroke[i],
            current_color,
            BRUSH_THICKNESS,
            cv2.LINE_AA
        )


# ============================================================
# ERASE AREA
# ============================================================

def erase_area(
    eraser_x,
    eraser_y
):

    """
    Erase the area touched by the eraser.

    Existing strokes are split into pieces.

    Their original colors remain unchanged.
    """

    global strokes

    new_strokes = []

    for stroke in strokes:

        points = stroke["points"]

        stroke_color = stroke["color"]

        if len(points) < 2:

            continue

        segments = []

        current_segment = []

        for point in points:

            d = distance(
                point,
                (
                    eraser_x,
                    eraser_y
                )
            )

            # Outside eraser
            if d > ERASER_SIZE:

                current_segment.append(
                    point
                )

            # Inside eraser
            else:

                if len(current_segment) >= 2:

                    segments.append(
                        current_segment
                    )

                current_segment = []

        # Remaining segment
        if len(current_segment) >= 2:

            segments.append(
                current_segment
            )

        # Store remaining segments
        for segment in segments:

            new_strokes.append({
                "points": segment,
                "color": stroke_color
            })

    strokes = new_strokes


# ============================================================
# CHANGE COLOR
# ============================================================

def change_color():

    """
    Cycle through:

        WHITE
        BLACK
        RED
        GREEN
        BLUE

    Only NEW strokes use the new color.
    """

    global current_color_index
    global current_color

    current_color_index += 1

    if current_color_index >= len(COLORS):

        current_color_index = 0

    current_color = COLORS[
        current_color_index
    ]

    print(
        "New stroke color:",
        COLOR_NAMES[current_color_index]
    )


# ============================================================
# CLEAR ALL
# ============================================================

def clear_all():

    global strokes
    global current_stroke

    strokes = []

    current_stroke = []

    print(
        "All drawings cleared."
    )


# ============================================================
# SAVE DRAWING
# ============================================================

def save_drawing(width, height):

    """
    Save drawing with a white background.
    """

    finish_current_stroke()

    # White background
    saved_image = np.ones(
        (
            height,
            width,
            3
        ),
        dtype=np.uint8
    ) * 255

    # Draw all strokes
    draw_strokes(
        saved_image
    )

    filename = (
        f"drawing_{int(time.time())}.png"
    )

    cv2.imwrite(
        filename,
        saved_image
    )

    print()
    print("Drawing saved:")
    print(os.path.abspath(filename))
    print()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # ========================================================
    # READ CAMERA
    # ========================================================

    success, frame = cap.read()

    if not success:

        print(
            "ERROR: Cannot read webcam."
        )

        break

    # Mirror webcam
    frame = cv2.flip(
        frame,
        1
    )

    frame_height, frame_width, _ = frame.shape

    # ========================================================
    # TIMESTAMP
    # ========================================================

    timestamp = int(
        time.time() * 1000
    )

    if timestamp <= last_timestamp:

        timestamp = (
            last_timestamp + 1
        )

    last_timestamp = timestamp

    # ========================================================
    # CONVERT BGR -> RGB
    # ========================================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # ========================================================
    # HAND DETECTION
    # ========================================================

    result = detector.detect_for_video(
        mp_image,
        timestamp
    )

    # Default
    raw_gesture = "NO HAND"
    gesture = "NO HAND"

    # ========================================================
    # HAND DETECTED
    # ========================================================

    if result.hand_landmarks:

        landmarks = result.hand_landmarks[0]

        # ----------------------------------------------------
        # Detect gesture
        # ----------------------------------------------------

        raw_gesture = detect_gesture(
            landmarks,
            frame_width,
            frame_height
        )

        # ----------------------------------------------------
        # Stable gesture
        # ----------------------------------------------------

        gesture = get_stable_gesture(
            raw_gesture
        )

        # ----------------------------------------------------
        # Index fingertip
        # ----------------------------------------------------

        index_tip = landmarks[8]

        finger_x = int(
            index_tip.x * frame_width
        )

        finger_y = int(
            index_tip.y * frame_height
        )

        # ====================================================
        # POSITION SMOOTHING
        # ====================================================

        if smooth_x is None:

            smooth_x = finger_x
            smooth_y = finger_y

        smooth_x = int(
            smooth_x * SMOOTHING
            +
            finger_x * (1 - SMOOTHING)
        )

        smooth_y = int(
            smooth_y * SMOOTHING
            +
            finger_y * (1 - SMOOTHING)
        )

        # ====================================================
        # DRAWING
        # ====================================================
        #
        # IMPORTANT:
        #
        # We use RAW gesture here.
        #
        # This means drawing stops immediately when the
        # index finger is no longer UP.
        # ====================================================

        if raw_gesture == "DRAW":

            # ------------------------------------------------
            # Start a new stroke
            # ------------------------------------------------

            if previous_x is None:

                current_stroke = [
                    (
                        smooth_x,
                        smooth_y
                    )
                ]

            else:

                # ------------------------------------------------
                # Add point when finger moves
                # ------------------------------------------------

                if distance(
                    (
                        smooth_x,
                        smooth_y
                    ),
                    (
                        previous_x,
                        previous_y
                    )
                ) > 1:

                    current_stroke.append(
                        (
                            smooth_x,
                            smooth_y
                        )
                    )

            previous_x = smooth_x
            previous_y = smooth_y

            # ------------------------------------------------
            # Drawing cursor
            # ------------------------------------------------

            cv2.circle(
                frame,
                (
                    smooth_x,
                    smooth_y
                ),
                10,
                current_color,
                2
            )

        else:

            # ------------------------------------------------
            # STOP DRAWING IMMEDIATELY
            # ------------------------------------------------

            finish_current_stroke()

            previous_x = None
            previous_y = None

        # ====================================================
        # COLOR
        # ====================================================

        if gesture == "COLOR":

            finish_current_stroke()

            previous_x = None
            previous_y = None

            current_time = time.time()

            if (
                current_time
                -
                last_color_change_time
                >
                COLOR_COOLDOWN
            ):

                change_color()

                last_color_change_time = (
                    current_time
                )

        # ====================================================
        # ERASER
        # ====================================================

        elif gesture == "ERASER":

            finish_current_stroke()

            previous_x = None
            previous_y = None

            erase_area(
                smooth_x,
                smooth_y
            )

            # ------------------------------------------------
            # Eraser circle
            # ------------------------------------------------

            cv2.circle(
                frame,
                (
                    smooth_x,
                    smooth_y
                ),
                ERASER_SIZE,
                (255, 255, 255),
                2
            )

            cv2.circle(
                frame,
                (
                    smooth_x,
                    smooth_y
                ),
                4,
                (255, 255, 255),
                -1
            )

        # ====================================================
        # PAUSE
        # ====================================================

        elif gesture == "PAUSE":

            finish_current_stroke()

            previous_x = None
            previous_y = None

        # ====================================================
        # CLEAR ALL
        # ====================================================

        elif gesture == "CLEAR ALL":

            finish_current_stroke()

            previous_x = None
            previous_y = None

            current_time = time.time()

            if (
                current_time
                -
                last_clear_time
                >
                CLEAR_COOLDOWN
            ):

                clear_all()

                last_clear_time = (
                    current_time
                )

    # ========================================================
    # NO HAND
    # ========================================================

    else:

        finish_current_stroke()

        previous_x = None
        previous_y = None

        smooth_x = None
        smooth_y = None

        raw_gesture = "NO HAND"

        candidate_gesture = "NO HAND"
        candidate_count = 0

        gesture = "NO HAND"

    # ========================================================
    # DRAW COMPLETED STROKES
    # ========================================================

    draw_strokes(
        frame
    )

    # ========================================================
    # DRAW CURRENT STROKE
    # ========================================================

    draw_current_stroke(
        frame
    )

    # ========================================================
    # DRAW HAND LANDMARKS
    # ========================================================

    if result.hand_landmarks:

        draw_hand(
            frame,
            result.hand_landmarks[0]
        )

    # ========================================================
    # CURRENT COLOR BOX
    # ========================================================

    cv2.rectangle(
        frame,
        (20, 15),
        (70, 65),
        current_color,
        -1
    )

    cv2.rectangle(
        frame,
        (20, 15),
        (70, 65),
        (0, 0, 0),
        2
    )

    cv2.putText(
        frame,
        f"COLOR: {COLOR_NAMES[current_color_index]}",
        (85, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
       (0, 0, 0),
        2
    )

    # ========================================================
    # GESTURE STATUS
    # ========================================================

    cv2.putText(
        frame,
        f"GESTURE: {gesture}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2
    )

    # ========================================================
    # STROKE COUNT
    # ========================================================

    cv2.putText(
        frame,
        f"STROKES: {len(strokes)}",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
       (0, 0, 0),
        2
    )

   # ========================================================
    # CONTROLS
    # ========================================================

    controls_x = frame_width - 200
    controls_y = 10
    
    cv2.putText( frame, "Index = Draw", (controls_x, controls_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2 )
    cv2.putText( frame, "Peace = Eraser", (controls_x, controls_y + 47),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2 )
    cv2.putText( frame, "OK = Color Change", (controls_x, controls_y + 69),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2 )
    cv2.putText( frame, "Fist = Pause", (controls_x, controls_y + 91),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2 )
    cv2.putText( frame, "Boo = Clear", (controls_x, controls_y + 113),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2 )
    cv2.putText( frame, "C = Clear | S = Save | Q = Quit", (controls_x, controls_y + 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2 )

    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "AI Finger Drawing System",
        frame
    )

    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    # --------------------------------------------------------
    # C = CLEAR
    # --------------------------------------------------------

    if key == ord("c"):

        clear_all()

        previous_x = None
        previous_y = None

    # --------------------------------------------------------
    # S = SAVE
    # --------------------------------------------------------

    elif key == ord("s"):

        save_drawing(
            frame_width,
            frame_height
        )

        previous_x = None
        previous_y = None

    # --------------------------------------------------------
    # Q = QUIT
    # --------------------------------------------------------

    elif key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

detector.close()

cv2.destroyAllWindows()
