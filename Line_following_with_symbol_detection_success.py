import time
import cv2
import numpy as np
import os
import RPi.GPIO as GPIO
from picamera2 import Picamera2

# ================= USER CONFIGURATION =================
on_color_line = False
entry_side = None # Stores where the color line was relative to black line
arrow_left, arrow_right, arrow_turn, enter_junction = False, False, False, False

# ================= DECISION LOGIC =================
ACTION = "FOLLOW"
ACTION_END_TIME = 0
ACTIONS = []   # list of (action, end_time)

def add_action(action, duration):
    end_time = time.time() + duration
    ACTIONS.append((action, end_time))

# ================= GPIO SETUP =================
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

IN1, IN2, IN3, IN4 = 23, 24, 17, 27
ENA, ENB = 12, 13
GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB], GPIO.OUT)

pwm_left = GPIO.PWM(ENA, 500)
pwm_right = GPIO.PWM(ENB, 500)
pwm_left.start(0)
pwm_right.start(0)

# ================= MOTOR FUNCTIONS =================
def forward(left, right):
    GPIO.output(IN1, GPIO.LOW); GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW); GPIO.output(IN4, GPIO.HIGH)
    pwm_left.ChangeDutyCycle(max(0, min(100, left)))
    pwm_right.ChangeDutyCycle(max(0, min(100, right)))

def turn_left_spin(speed):
    GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW);  GPIO.output(IN4, GPIO.HIGH)
    pwm_left.ChangeDutyCycle(speed); pwm_right.ChangeDutyCycle(speed)

def turn_right_spin(speed):
    GPIO.output(IN1, GPIO.LOW);  GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH); GPIO.output(IN4, GPIO.LOW)
    pwm_left.ChangeDutyCycle(speed); pwm_right.ChangeDutyCycle(speed)

def stop():
    pwm_left.ChangeDutyCycle(0); pwm_right.ChangeDutyCycle(0)

# ================= HSV =================
HSV_THRESHOLDS = {
    "black":  {"low": np.array([0, 0, 0]),     "high": np.array([180, 255, 60])},
    "yellow": {"low": np.array([20, 100, 100]), "high": np.array([35, 255, 255])},
    "red1":   {"low": np.array([0, 100, 100]),  "high": np.array([10, 255, 255])},
    "red2":   {"low": np.array([160, 100, 100]), "high": np.array([180, 255, 255])}
}

def get_selective_mask(hsv_roi):
    # 1. Detect Color
    m1 = cv2.inRange(hsv_roi, HSV_THRESHOLDS["red1"]["low"], HSV_THRESHOLDS["red1"]["high"])
    m2 = cv2.inRange(hsv_roi, HSV_THRESHOLDS["red2"]["low"], HSV_THRESHOLDS["red2"]["high"])
    red_mask = cv2.bitwise_or(m1, m2)
    yellow_mask = cv2.inRange(hsv_roi, HSV_THRESHOLDS["yellow"]["low"], HSV_THRESHOLDS["yellow"]["high"])
    color_mask = cv2.bitwise_or(red_mask, yellow_mask)

    # 2. Detect Black
    black_mask = cv2.inRange(hsv_roi, HSV_THRESHOLDS["black"]["low"], HSV_THRESHOLDS["black"]["high"])

    if cv2.countNonZero(color_mask) > 500:
        return color_mask, True, black_mask # Return black_mask too for side-checking
    
    return black_mask, False, None


# ================= ARROW DETECTION =================
def detect_arrow_skeleton_color(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ===== COLOR MASK =====
    r1 = cv2.inRange(hsv, np.array([0,100,100]), np.array([10,255,255]))
    r2 = cv2.inRange(hsv, np.array([160,100,100]), np.array([180,255,255]))
    red = cv2.bitwise_or(r1, r2)

    green = cv2.inRange(hsv, np.array([40, 80, 80]), np.array([80, 255, 255]))
    blue  = cv2.inRange(hsv, np.array([90, 80, 80]), np.array([130, 255, 255]))
    orange = cv2.inRange(hsv, np.array([10, 100, 100]), np.array([25, 255, 255]))

    mask = cv2.bitwise_or(red, green)
    mask = cv2.bitwise_or(mask, blue)
    mask = cv2.bitwise_or(mask, orange)

    # ===== MORPH CLEAN =====
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # ===== FIND CONTOURS =====
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)

    # ?? LOWER threshold to allow small arrows
    if cv2.contourArea(cnt) < 200:
        return None

    # ===== CROP ROI AROUND ARROW =====
    x, y, w, h = cv2.boundingRect(cnt)
    roi = mask[y:y+h, x:x+w]

    # ?? UPSCALE SMALL ARROWS (KEY FIX)
    scale = 3
    roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    # ===== FILL SHAPE =====
    arrow_mask = np.zeros_like(roi)
    cnt_shifted = cnt - [x, y]
    cnt_scaled = (cnt_shifted * scale).astype(np.int32)
    cv2.drawContours(arrow_mask, [cnt_scaled], -1, 255, -1)

    # ===== SKELETON =====
    skel = cv2.ximgproc.thinning(arrow_mask)

    # ===== FIND ENDPOINTS =====
    endpoints = []
    h, w = skel.shape

    for yy in range(1, h-1):
        for xx in range(1, w-1):
            if skel[yy, xx] == 255:
                neighbors = np.sum(skel[yy-1:yy+2, xx-1:xx+2] == 255) - 1
                if neighbors <= 2:  # ?? relaxed (was ==1)
                    endpoints.append((xx, yy))

    if len(endpoints) < 2:
        return None

    pts = np.array(endpoints)
    center = np.mean(pts, axis=0)

    dists = np.linalg.norm(pts - center, axis=1)
    tip = pts[np.argmax(dists)]

    dx = tip[0] - center[0]
    dy = tip[1] - center[1]

    if abs(dx) > abs(dy):
        return "Left" if dx > 0 else "Right"
    else:
        return "Forward" if dy < 0 else "Backward"

def is_square(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    return len(approx) == 4

# ================= ORB =================
ROOT_FOLDER = 'samples'
ORB = cv2.ORB_create(nfeatures=1000)
BF = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

class_data = []
for class_name in os.listdir(ROOT_FOLDER):
    path = os.path.join(ROOT_FOLDER, class_name)
    if os.path.isdir(path):
        imgs = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg'))]
        if imgs:
            img = cv2.imread(os.path.join(path, imgs[0]), 0)
            kp, des = ORB.detectAndCompute(img, None)
            if des is not None:
                class_data.append({'name': class_name.lower(), 'image': img, 'kp': kp, 'des': des})

# ================= CAMERA =================
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "XRGB8888", "size": (320, 240)})
picam2.configure(config)
picam2.start()
picam2.set_controls({
    "AeEnable": False,
    "ExposureTime": 5000, # ms exposure to kill motion blur
    "AnalogueGain": 20     # Increase gain so the image isn't too dark
})

KP, KD = 0.9, 0.7
previous_error = 0
BASE_SPEED, RECOVERY_SPEED = 35, 80
look_ahead_direction = 0
CENTER = 160

# ================= MAIN LOOP =================
try:
    while True:
        frame = picam2.capture_array()
        frame = cv2.flip(frame, -1)
        HEIGHT, WIDTH, CHANNELS = frame.shape


        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # ================= APPLY DECISION =================
        current_time = time.time()
        
        # Remove expired actions
        ACTIONS[:] = [a for a in ACTIONS if a[1] > current_time]
        
        if ACTIONS:
            current_action = ACTIONS[0][0]

            if current_action == "STOP":
                stop()
                continue
            elif current_action == "LEFT":
                turn_right_spin(90)
                continue
            elif current_action == "RIGHT":
                turn_left_spin(90)
                continue
            elif current_action == "TURN AROUND":
                turn_right_spin(90)
                continue
            elif current_action == "FORWARD":
                forward(50, 50)
                continue
        
        display = frame.copy()
        
        # --- TOP ROI (look-ahead memory) ---
        roi_top = hsv[80:140, :]
        # Added another , _ to handle the 3rd return value (black_mask)
        mask_top, _, _ = get_selective_mask(roi_top) 
        con_top, _ = cv2.findContours(mask_top, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if con_top:
            ct = max(con_top, key=cv2.contourArea)
            Mt = cv2.moments(ct)
            x, y, w, h = cv2.boundingRect(ct)
            if Mt["m00"] > 500:
                ctx = int(Mt["m10"] / Mt["m00"])
                if ctx < CENTER - 25:
                    look_ahead_direction = -1
                elif ctx > CENTER + 25:
                    look_ahead_direction = 1
                else:
                    look_ahead_direction = 0
            if w / h < 1.5 and arrow_turn == True:
                enter_junction = False
            if (w / h) > 3 and arrow_turn == True and enter_junction == False:
                print(arrow_right, arrow_left)
                if arrow_right == True:
                    add_action("FORWARD", 0.3)
                    add_action("RIGHT", 0.6)
                elif arrow_left == True:
                    add_action("FORWARD", 0.3)
                    add_action("LEFT", 0.6)
                arrow_right = arrow_left = arrow_turn = False
                enter_junction = True

        # --- BOTTOM ROI (control) ---
        roi_bot = hsv[160:240, :]
        # mask (1), is_priority (2), black_mask (3)
        mask, is_priority, black_mask = get_selective_mask(roi_bot)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



        if contours:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 500:
                cx = int(M["m10"] / M["m00"])
                cv2.circle(display, (cx, 200), 10, (255, 255, 0) if is_priority else (0, 255, 0), -1)

                # --- IMPROVED TRANSITION LOGIC ---
                if is_priority and not on_color_line:
                    # Find where the black line was to decide turn direction
                    con_b, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if con_b:
                        cb = max(con_b, key=cv2.contourArea)
                        Mb = cv2.moments(cb)
                        if Mb["m00"] > 0:
                            bx = int(Mb["m10"] / Mb["m00"])
                            # Turn away from black line toward color line
                            entry_side = "LEFT" if cx < bx else "RIGHT"
                    
                    # Fallback if black line is already gone
                    if entry_side is None: entry_side = "LEFT" if cx < CENTER else "RIGHT"
                    add_action("FORWARD", 0.1)
                    add_action(entry_side, 0.4) # Increased duration for sharper jump
                    on_color_line = True
                
                elif not is_priority and on_color_line:
                    add_action(entry_side, 0.4)
                    on_color_line = False
                # --------------------------------

                error = cx - CENTER
                derivative = error - previous_error
                correction = (KP * error + KD * derivative)
                previous_error = error
                forward(BASE_SPEED - correction, BASE_SPEED + correction)

        else:
            # ?? RESTORED recovery (THIS FIXES JUNCTION TURNING)
            if look_ahead_direction == -1:
                turn_right_spin(RECOVERY_SPEED)
            elif look_ahead_direction == 1:
                turn_left_spin(RECOVERY_SPEED)
            else:
                stop()

        # ================= ORB DETECTION =================
        kp_f, des_f = ORB.detectAndCompute(gray, None)
        best_match_obj, max_matches, best_matches_list = None, 0, []

        if des_f is not None:
            time.sleep(0.001)
            for item in class_data:
                matches = BF.match(item['des'], des_f)
                good = [m for m in matches if m.distance < 45.0]
                if len(good) > max_matches and len(good) > 22:
                    max_matches, best_match_obj, best_matches_list = len(good), item, good


        if best_match_obj:

            if "arrow" in best_match_obj['name']:
                direction = detect_arrow_skeleton_color(frame)

                if direction:
                    cv2.putText(display, f"Dir: {direction}", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    if direction == "Left":
                        print("left")
                        if arrow_right == False and arrow_turn == False:
                            add_action("FORWARD", 0.1)
                            add_action("LEFT", 0.6)
                            arrow_left = True
                        arrow_turn, enter_junction = True, True

                    elif direction == "Right":
                        print("right")
                        if arrow_left == False and arrow_turn == False:
                            add_action("FORWARD", 0.1)
                            add_action("RIGHT", 0.6)
                            arrow_right = True
                        arrow_turn, enter_junction = True, True

            cv2.putText(display, f"Match: {best_match_obj['name']}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if best_match_obj['name'] == "button" or best_match_obj['name'] == "danger":
                #print("stop")
                add_action("STOP", 2)
                add_action("FORWARD", 0.2)
            elif best_match_obj['name'] == "recycle":
                time.sleep(0.5)
                kp_f, des_f = ORB.detectAndCompute(gray, None)
                best_match_obj, max_matches, best_matches_list = None, 0, []

                if des_f is not None:
                    for item in class_data:
                        matches = BF.match(item['des'], des_f)
                        good = [m for m in matches if m.distance < 45.0]
                        if len(good) > max_matches and len(good) > 22:
                            max_matches, best_match_obj, best_matches_list = len(good), item, good
                if "arrow" in best_match_obj['name']:
                    direction = detect_arrow_skeleton_color(frame)

                    if direction:
                        cv2.putText(display, f"Dir: {direction}", (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                        if direction == "Left":
                            print("left")
                            if arrow_right == False and arrow_turn == False:
                                add_action("FORWARD", 0.1)
                                add_action("LEFT", 0.6)
                                arrow_left = True
                            arrow_turn, enter_junction = True, True

                        elif direction == "Right":
                            print("right")
                            if arrow_left == False and arrow_turn == False:
                                add_action("FORWARD", 0.1) 
                                add_action("RIGHT", 0.6)
                                arrow_right = True
                            arrow_turn, enter_junction = True, True
                else:

                    #print("recycle")
                    add_action("TURN AROUND", 3.5)
                    add_action("FORWARD", 0.2)
            elif best_match_obj['name'] == "qrcode" or best_match_obj['name'] == "fingerprint":
                print(f"{best_match_obj['name']}")

        cv2.imshow("Combined System", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) & 0xFF == ord('c'):
            arrow_left, arrow_right, arrow_turn = False, False, False
            print("cleared")

finally:
    stop()
    pwm_left.stop()
    pwm_right.stop()
    picam2.stop()
    GPIO.cleanup()
    cv2.destroyAllWindows()


Final code(week 3)
