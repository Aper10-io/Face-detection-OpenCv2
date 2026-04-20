import cv2
import mediapipe as mp
import numpy as np
import math
import time

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# --- KONFIGURASI KOTAK 3D ---
# Titik-titik (vertices) dari sebuah kubus 3D
cube_vertices = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
])

# Garis-garis yang menghubungkan titik-titik tersebut
cube_edges = [
    (0, 1), (1, 2), (2, 3), (3, 0), # Sisi belakang
    (4, 5), (5, 6), (6, 7), (7, 4), # Sisi depan
    (0, 4), (1, 5), (2, 6), (3, 7)  # Garis penghubung
]

# Variabel Status Kotak
cube_scale = 100
cube_pos = [320, 240] # Posisi tengah awal (asumsi resolusi 640x480)
angle_x, angle_y, angle_z = 0.0, 0.0, 0.0

# --- FUNGSI MATEMATIKA & EFEK ---

def project_3d_to_2d(vertex, pos, scale, angle_x, angle_y, angle_z):
    """Fungsi untuk memutar titik 3D dan memproyeksikannya ke layar 2D"""
    # Matriks Rotasi X
    rot_x = np.array([
        [1, 0, 0],
        [0, math.cos(angle_x), -math.sin(angle_x)],
        [0, math.sin(angle_x), math.cos(angle_x)]
    ])
    # Matriks Rotasi Y
    rot_y = np.array([
        [math.cos(angle_y), 0, math.sin(angle_y)],
        [0, 1, 0],
        [-math.sin(angle_y), 0, math.cos(angle_y)]
    ])
    # Matriks Rotasi Z
    rot_z = np.array([
        [math.cos(angle_z), -math.sin(angle_z), 0],
        [math.sin(angle_z), math.cos(angle_z), 0],
        [0, 0, 1]
    ])
    
    # Menerapkan rotasi ke titik (vertex)
    rotated_2d = np.dot(rot_z, np.dot(rot_y, np.dot(rot_x, vertex)))
    
    # Proyeksi Orthogonal Sederhana dengan Scaling dan Translasi
    x_2d = int(rotated_2d[0] * scale) + pos[0]
    y_2d = int(rotated_2d[1] * scale) + pos[1]
    return (x_2d, y_2d)

def draw_magic_circle(frame, center, radius, time_val, color=(0, 165, 255)):
    """Menggambar efek lingkaran sihir ala Doctor Strange di sekitar tangan"""
    # Lingkaran luar (berputar)
    cv2.circle(frame, center, radius, color, 2)
    cv2.circle(frame, center, radius - 10, color, 1)
    
    # Menggambar rune/garis di dalam yang berputar
    points = []
    for i in range(6):
        angle = time_val * 2 + (i * math.pi / 3) # Berputar seiring waktu
        x = int(center[0] + (radius - 5) * math.cos(angle))
        y = int(center[1] + (radius - 5) * math.sin(angle))
        points.append([x, y])
    
    points = np.array(points, np.int32)
    points = points.reshape((-1, 1, 2))
    cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)
    
    # Bintang di tengah
    for i in range(3):
        pt1 = tuple(points[i][0])
        pt2 = tuple(points[i+3][0])
        cv2.line(frame, pt1, pt2, color, 1)

def get_distance(p1, p2):
    """Menghitung jarak antara dua titik (x, y)"""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# --- PROGRAM UTAMA ---
def main():
    global cube_pos, cube_scale, angle_x, angle_y, angle_z
    
    cap = cv2.VideoCapture(0)
    # Set resolusi ke 1280x720 untuk tampilan yang lebih luas
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Warna "Sihir" Doctor Strange (Orange/Kuning) dalam BGR
    magic_color = (0, 165, 255) 
    # Warna Kotak Hologram (Cyan/Biru Muda)
    box_color = (255, 255, 0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Balik frame seperti cermin (Anti-Mirror)
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # 2. Proses MediaPipe (butuh format RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Variabel untuk interaksi
        grab_points = [] # Menyimpan titik genggaman dari tangan-tangan
        
        # Waktu untuk animasi putaran
        current_time = time.time()
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Ambil koordinat ujung jempol (4) dan ujung telunjuk (8)
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                # Konversi dari rasio MediaPipe ke pixel layar
                t_x, t_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                i_x, i_y = int(index_tip.x * w), int(index_tip.y * h)
                
                # Pusat dari "capitan" jari (tengah-tengah jempol dan telunjuk)
                pinch_center = ((t_x + i_x) // 2, (t_y + i_y) // 2)
                
                # Gambar lingkaran sihir di telapak tangan (menggunakan tengah-tengah tangan)
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                middle = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                palm_center = (int((wrist.x + middle.x)*w/2), int((wrist.y + middle.y)*h/2))
                
                draw_magic_circle(frame, palm_center, 60, current_time, magic_color)
                
                # Hitung jarak antara jempol dan telunjuk
                distance = get_distance((t_x, t_y), (i_x, i_y))
                
                # Jika jaraknya dekat (mencubit/menggenggam)
                if distance < 40:
                    grab_points.append(pinch_center)
                    # Efek cahaya saat menggenggam
                    cv2.circle(frame, pinch_center, 15, (255, 255, 255), -1)
                    cv2.circle(frame, pinch_center, 25, magic_color, 2)
                
                # Gambar Landmark (Garis-garis tangan opsional, saya matikan agar fokus ke sihir)
                # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # --- LOGIKA INTERAKSI KOTAK 3D ---
        # Rotasi otomatis kotak perlahan-lahan
        angle_x += 0.02
        angle_y += 0.03
        angle_z += 0.01
        
        if len(grab_points) == 1:
            # Jika 1 tangan menggenggam -> Pindahkan kotak (Telekinesis)
            cube_pos = list(grab_points[0])
            # Gambar garis energi dari tangan ke kotak
            cv2.line(frame, grab_points[0], tuple(cube_pos), magic_color, 3)
            
        elif len(grab_points) == 2:
            # Jika 2 tangan menggenggam -> Posisikan di tengah tangan & Ubah ukuran (Scale)
            p1, p2 = grab_points[0], grab_points[1]
            
            # Posisikan kotak tepat di tengah-tengah kedua tangan
            cube_pos[0] = (p1[0] + p2[0]) // 2
            cube_pos[1] = (p1[1] + p2[1]) // 2
            
            # Hitung jarak kedua tangan untuk mengatur besarnya kotak
            hands_dist = get_distance(p1, p2)
            cube_scale = int(max(30, hands_dist / 2)) # Skala disesuaikan
            
            # Gambar garis sihir melintang antar dua tangan
            cv2.line(frame, p1, p2, magic_color, 2)
            cv2.circle(frame, tuple(cube_pos), 10, box_color, -1)
            
            # Kotak berputar cepat saat ditarik dua tangan (Efek charging)
            angle_x += 0.1
            angle_y += 0.1

        # --- MENGGAMBAR KOTAK 3D HOLOGRAM ---
        # Kumpulkan titik 2D
        projected_points = []
        for vertex in cube_vertices:
            pt2d = project_3d_to_2d(vertex, cube_pos, cube_scale, angle_x, angle_y, angle_z)
            projected_points.append(pt2d)
            # Titik sudut bersinar
            cv2.circle(frame, pt2d, 5, (255, 255, 255), -1)

        # Gambar garis yang menghubungkan titik-titik (Rangka Kotak)
        for edge in cube_edges:
            pt1 = projected_points[edge[0]]
            pt2 = projected_points[edge[1]]
            # Menggambar dengan dua layer agar terlihat bersinar (Hologram efek)
            cv2.line(frame, pt1, pt2, (255, 100, 0), 4) # Bayangan tebal luar
            cv2.line(frame, pt1, pt2, box_color, 2)     # Inti dalam terang

        # --- UI TAMBAHAN ---
        cv2.putText(frame, "Cubit dengan 1 Tangan: Pindahkan Kotak", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Cubit dengan 2 Tangan: Tarik/Perbesar Kotak", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Tekan 'Q' untuk keluar", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Tampilkan hasil
        cv2.imshow('Sorcerer Supreme - 3D Box Manipulation', frame)

        # Tombol keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()