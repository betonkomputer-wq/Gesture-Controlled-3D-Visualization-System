import cv2
import mediapipe as mp
import numpy as np
import time

# --- PALET WARNA MINIMALIS ---
C_BG = (10, 10, 10)           
C_WHITE = (245, 245, 245)     
C_GREY = (100, 100, 100)      
C_ACCENT = (255, 200, 50)     
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Init MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# --- GENERATOR BENTUK ---
def create_globe(radius=1.0, num_points=1200):
    vertices = []
    phi = np.pi * (3. - np.sqrt(5.)) 
    for i in range(num_points):
        y = 1 - (i / float(num_points - 1)) * 2
        r_y = np.sqrt(1 - y * y)
        theta = phi * i
        vertices.append([np.cos(theta)*r_y*radius, y*radius, np.sin(theta)*r_y*radius])
    return np.array(vertices, dtype=np.float32)

def create_saturn(radius=1.0, num_points=1200):
    vertices = []
    num_planet = int(num_points * 0.4)
    phi = np.pi * (3. - np.sqrt(5.))
    pr = radius * 0.6
    for i in range(num_planet):
        y = 1 - (i / float(num_planet - 1)) * 2
        r_y = np.sqrt(1 - y * y)
        theta = phi * i
        vertices.append([np.cos(theta)*r_y*pr, y*pr, np.sin(theta)*r_y*pr])
    num_ring = num_points - num_planet
    for i in range(num_ring):
        theta = np.random.uniform(0, 2*np.pi)
        r = np.sqrt(np.random.uniform(pr**2 * 2.5, radius**2 * 4))
        vertices.append([r * np.cos(theta), np.random.uniform(-0.02, 0.02), r * np.sin(theta)])
    return np.array(vertices, dtype=np.float32)

def create_dna(radius=1.0, num_points=1200):
    vertices = []
    turns = 4; height = 3.0
    for i in range(num_points):
        t = i / num_points
        y = (t - 0.5) * height
        angle = t * turns * 2 * np.pi
        if i % 2 == 0: vertices.append([radius*0.5*np.cos(angle), y, radius*0.5*np.sin(angle)])
        else: vertices.append([radius*0.5*np.cos(angle+np.pi), y, radius*0.5*np.sin(angle+np.pi)])
        if i % 15 == 0:
            for k in range(10):
                r = k/10
                x1, z1 = radius*0.5*np.cos(angle), radius*0.5*np.sin(angle)
                x2, z2 = radius*0.5*np.cos(angle+np.pi), radius*0.5*np.sin(angle+np.pi)
                vertices.append([x1+(x2-x1)*r, y, z1+(z2-z1)*r])
    return np.array(vertices, dtype=np.float32)

def create_galaxy(radius=1.0, num_points=1200):
    vertices = []
    arms = 3; arm_sep = 2 * np.pi / arms
    for i in range(num_points):
        off = (i % arms) * arm_sep
        dist = np.random.power(2) * radius * 1.5
        angle = dist * 3.0 + off
        vertices.append([dist * np.cos(angle) + np.random.normal(0, 0.05), np.random.normal(0, 0.1 * (2 - dist)), dist * np.sin(angle) + np.random.normal(0, 0.05)])
    return np.array(vertices, dtype=np.float32)

def create_starfield(num_stars=300):
    stars = []
    for _ in range(num_stars):
        stars.append([np.random.uniform(-15, 15), np.random.uniform(-10, 10), np.random.uniform(1, 20)])
    return np.array(stars, dtype=np.float32)

# --- UI DRAWING ---

def draw_sidebar_minimal(img, current_key, is_auto_rotate):
    h, w = img.shape[:2]
    start_y = 150
    x_offset = 40
    
    cv2.putText(img, "OBJECTS", (x_offset, start_y - 30), FONT, 0.5, C_GREY, 1, cv2.LINE_AA)
    
    options = [('1', 'GLOBE'), ('2', 'SATURN'), ('3', 'DNA'), ('4', 'GALAXY')]
    
    y_pos = start_y
    for key, name in options:
        is_active = (key == current_key)
        if is_active:
            cv2.line(img, (x_offset - 15, y_pos - 10), (x_offset - 15, y_pos + 10), C_ACCENT, 2)
            cv2.putText(img, name, (x_offset, y_pos + 5), FONT, 0.6, C_WHITE, 1, cv2.LINE_AA)
            cv2.putText(img, key, (x_offset + 120, y_pos + 5), FONT, 0.4, C_ACCENT, 1, cv2.LINE_AA)
        else:
            cv2.putText(img, name, (x_offset, y_pos + 5), FONT, 0.6, C_GREY, 1, cv2.LINE_AA)
        y_pos += 50
        
    y_rot = y_pos + 30
    cv2.putText(img, "AUTO-ROTATION", (x_offset, y_rot), FONT, 0.5, C_GREY, 1, cv2.LINE_AA)
    status_text = "ON" if is_auto_rotate else "OFF"
    col_status = C_WHITE if is_auto_rotate else C_GREY
    cv2.putText(img, status_text, (x_offset, y_rot + 30), FONT, 0.6, col_status, 1, cv2.LINE_AA)
    if is_auto_rotate:
        cv2.circle(img, (x_offset + 50, y_rot + 22), 3, C_ACCENT, -1)

def draw_header_footer_minimal(img, status, fps):
    h, w = img.shape[:2]
    cv2.putText(img, "HOLO // OS", (40, 50), FONT, 0.7, C_WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, "v5.5", (160, 50), FONT, 0.4, C_ACCENT, 1, cv2.LINE_AA)
    
    if status != "IDLE":
        text_w = cv2.getTextSize(status, FONT, 0.5, 1)[0][0]
        center_x = w // 2
        overlay = img.copy()
        cv2.rectangle(overlay, (center_x - text_w//2 - 20, 30), (center_x + text_w//2 + 20, 60), (30,30,30), -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
        cv2.putText(img, status, (center_x - text_w//2, 50), FONT, 0.5, C_WHITE, 1, cv2.LINE_AA)
        cv2.circle(img, (center_x - text_w//2 - 10, 46), 2, C_ACCENT, -1)

    cv2.putText(img, f"{int(fps)} FPS", (40, h - 40), FONT, 0.5, C_GREY, 1, cv2.LINE_AA)
    cv2.putText(img, "RENDER: VULKAN", (120, h - 40), FONT, 0.4, (80,80,80), 1, cv2.LINE_AA)

def draw_pip_minimal(img, x, y, w, h):
    cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 50), 1)
    cv2.putText(img, "REALITY FEED", (x, y - 10), FONT, 0.4, C_GREY, 1, cv2.LINE_AA)
    cv2.circle(img, (x + w - 10, y + 15), 3, (0, 0, 200), -1)

# --- SETUP SYSTEM ---
shapes = {
    '1': ('GLOBE', create_globe()),
    '2': ('SATURN', create_saturn()),
    '3': ('DNA HELIX', create_dna()),
    '4': ('GALAXY', create_galaxy())
}

curr_key = '2'
main_vertices = shapes[curr_key][1]
bg_stars = create_starfield(400)

auto_rotate = True
rx, ry, rz = 0.5, 0, 0
rvx, rvy, rvz = 0, 0, 0
scale = 1.0; scale_v = 0
px, py = 0, 0; pvx, pvy = 0, 0

# --- PENGATURAN FISIKA BARU ---
# Damping lebih rendah agar momentum bertahan sedikit lebih lama
DAMP = 0.94      
# Sensitivity disesuaikan karena kita menggunakan delta position
SENS_ROT = 6.0   
SENS_MOVE = 3.0
# Faktor smoothing untuk input tangan (0.0 - 1.0)
# Nilai kecil = sangat smooth tapi agak delay
# Nilai besar = responsif tapi agak jittery
SMOOTH_FACTOR = 0.3 

cap = cv2.VideoCapture(0)
cap.set(3, 1280); cap.set(4, 720)
prev_time = 0

# Helper Math
def rotate_coords(verts, rx, ry, rz):
    c, s = np.cos(rx), np.sin(rx); Rx = np.array([[1,0,0],[0,c,-s],[0,s,c]])
    c, s = np.cos(ry), np.sin(ry); Ry = np.array([[c,0,s],[0,1,0],[-s,0,c]])
    c, s = np.cos(rz), np.sin(rz); Rz = np.array([[c,-s,0],[s,c,0],[0,0,1]])
    return np.dot(verts, (Rz @ Ry @ Rx).T)

def update_stars(img, stars, w, h, speed=0.08):
    stars[:, 2] -= speed
    reset = stars[:, 2] < 0.5
    if np.sum(reset) > 0:
        stars[reset, 2] = np.random.uniform(15, 20, np.sum(reset))
        stars[reset, 0] = np.random.uniform(-15, 15, np.sum(reset))
        stars[reset, 1] = np.random.uniform(-10, 10, np.sum(reset))
    f = 400
    for s in stars:
        x, y, z = s
        fac = f/z
        sx, sy = int(x*fac + w/2), int(y*fac + h/2)
        if 0<=sx<w and 0<=sy<h:
            b = max(20, min(100, int(255*(1-z/20.0))))
            cv2.circle(img, (sx, sy), 1, (b,b,b), -1)

def draw_obj(img, verts, w, h, s, ox, oy):
    proj = []; depths = []
    z_off = 4.0
    for v in verts:
        z = v[2] + z_off
        if z <= 0.1: z=0.1
        fac = (250*s)/z
        proj.append([int(v[0]*fac + w/2 + ox), int(v[1]*fac + h/2 + oy)])
        depths.append(z)
    
    inds = np.argsort(depths)[::-1]
    min_d, max_d = min(depths), max(depths)
    dr = max_d - min_d if max_d != min_d else 1
    
    for i in inds:
        pt = tuple(proj[i])
        if not (0<=pt[0]<w and 0<=pt[1]<h): continue
        rel = 1.0 - (depths[i]-min_d)/dr
        val = int(50 + rel*205)
        rad = max(1, int(1 + rel*3*s))
        col = (val, val, val) 
        cv2.circle(img, pt, rad, col, -1)
        if rel > 0.95: cv2.circle(img, pt, rad+1, (255,255,255), 1)

# --- MAIN LOOP ---
prev_rh, prev_lh = None, None
prev_z_angle = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h_scr, w_scr = frame.shape[:2]
    
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    
    canvas = np.zeros((h_scr, w_scr, 3), dtype=np.uint8)
    update_stars(canvas, bg_stars, w_scr, h_scr, speed=0.05)
    
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    status = "IDLE"
    
    is_interacting_right = False
    
    if results.multi_hand_landmarks:
        for lm, hnd in zip(results.multi_hand_landmarks, results.multi_handedness):
            lbl = hnd.classification[0].label
            
            # Helper Coords
            palm = [0,1,5,9,13,17]
            px = sum([lm.landmark[i].x for i in palm])/6
            py = sum([lm.landmark[i].y for i in palm])/6
            
            if lbl == "Right":
                is_interacting_right = True
                tips=[8,12,16,20]; mcps=[5,9,13,17]
                fist = sum([1 for t,m in zip(tips,mcps) if lm.landmark[t].y > lm.landmark[m].y])>=3
                
                if fist:
                    # ROTASI Z (Roll) - Menggunakan sudut vektor pergelangan tangan
                    vec = np.array([lm.landmark[9].x - lm.landmark[0].x, lm.landmark[9].y - lm.landmark[0].y])
                    curr_angle = np.arctan2(vec[0], -vec[1])
                    
                    # Hitung delta angle (perubahan sudut)
                    # Ini mencegah snapping liar
                    if prev_rh is not None: 
                        angle_diff = curr_angle - prev_z_angle
                        # Handle wrap around (misal lompat dari PI ke -PI)
                        if angle_diff > np.pi: angle_diff -= 2*np.pi
                        if angle_diff < -np.pi: angle_diff += 2*np.pi
                        
                        # Apply smoothing
                        rvz = rvz * (1-SMOOTH_FACTOR) + (angle_diff * 0.8) * SMOOTH_FACTOR
                        
                    prev_z_angle = curr_angle
                    status = "ROLL"
                    
                else:
                    # ROTASI X/Y (Pitch/Yaw) - Direct Manipulation
                    if prev_rh is not None:
                        # Hitung PERUBAHAN posisi (Delta)
                        dx = px - prev_rh[0]
                        dy = py - prev_rh[1]
                        
                        # Mapping:
                        # Gerak Tangan Kanan-Kiri (dx) -> Rotasi Y (Yaw)
                        # Gerak Tangan Atas-Bawah (dy) -> Rotasi X (Pitch)
                        
                        target_rvy = dx * SENS_ROT
                        target_rvx = dy * SENS_ROT
                        
                        # Apply Smoothing ke Velocity
                        # Kita blend velocity lama dengan input baru agar halus
                        rvx = rvx * (1 - SMOOTH_FACTOR) + target_rvx * SMOOTH_FACTOR
                        rvy = rvy * (1 - SMOOTH_FACTOR) + target_rvy * SMOOTH_FACTOR
                        
                        status = "TUMBLE"
                
                prev_rh = (px, py)
                
            elif lbl == "Left":
                # Logic Kiri Tetap Sama
                dist = np.hypot(lm.landmark[8].x-lm.landmark[4].x, lm.landmark[8].y-lm.landmark[4].y)
                if dist < 0.05:
                    if prev_lh:
                        # Direct translation logic
                        dx = px - prev_lh[0]
                        dy = py - prev_lh[1]
                        
                        pvx = pvx * (1 - SMOOTH_FACTOR) + (dx * w_scr * SENS_MOVE) * SMOOTH_FACTOR
                        pvy = pvy * (1 - SMOOTH_FACTOR) + (dy * h_scr * SENS_MOVE) * SMOOTH_FACTOR
                        status = "MOVE"
                else:
                    scale_v += (max(0.3, min(3.5, dist * 8)) - scale) * 0.1
                    status = "SCALE"
                prev_lh = (px, py)
    else:
        # Reset tracker jika tangan hilang agar tidak 'jump' saat masuk lagi
        prev_rh = None
        prev_lh = None

    # --- PHYSICS UPDATE ---
    
    # Auto Rotate logic: Hanya berputar jika tidak sedang dipegang tangan kanan
    if auto_rotate and not is_interacting_right: 
        rvy += 0.005 # Putar pelan saat idle
    
    # Apply Velocity to Position
    rx += rvx; ry += rvy; rz += rvz
    scale += scale_v; px += pvx; py += pvy
    
    # Apply Damping (Gesekan Udara)
    rvx*=DAMP; rvy*=DAMP; rvz*=DAMP
    scale_v*=DAMP; pvx*=DAMP; pvy*=DAMP
    
    # Clamp Scale
    scale = max(0.2, min(4.0, scale))
    
    # Render Object
    curr_v = rotate_coords(main_vertices, rx, ry, rz) * scale
    draw_obj(canvas, curr_v, w_scr, h_scr, scale, px, py)
    
    # UI
    draw_header_footer_minimal(canvas, status, fps)
    draw_sidebar_minimal(canvas, curr_key, auto_rotate)
    
    # PiP
    pip_w, pip_h = 240, 135
    pip = cv2.resize(frame, (pip_w, pip_h))
    pip_gray = cv2.cvtColor(pip, cv2.COLOR_BGR2GRAY)
    pip_gray = cv2.cvtColor(pip_gray, cv2.COLOR_GRAY2BGR)
    pip_final = cv2.addWeighted(pip, 0.4, pip_gray, 0.6, 0)
    
    pip_x, pip_y = w_scr - pip_w - 40, h_scr - pip_h - 40
    canvas[pip_y:pip_y+pip_h, pip_x:pip_x+pip_w] = pip_final
    draw_pip_minimal(canvas, pip_x, pip_y, pip_w, pip_h)
    
    cv2.imshow('Holo-UI Smooth Control', canvas)
    
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'): break
    elif k == ord('a'): auto_rotate = not auto_rotate
    elif k in [ord('1'), ord('2'), ord('3'), ord('4')]:
        curr_key = chr(k)
        main_vertices = shapes[curr_key][1]

cap.release()
cv2.destroyAllWindows()
hands.close()