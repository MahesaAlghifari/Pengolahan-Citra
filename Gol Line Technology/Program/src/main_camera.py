import sys
import os
import cv2
import numpy as np

# Menambahkan folder src ke dalam PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mengimpor fungsi untuk deteksi bola dan garis menggunakan metode threshold warna dan Hough Transform
from detection.color_threshold import apply_color_threshold
from detection.hough_transform import detect_circles
from utils.config import HSV_RANGE, HOUGH_PARAMS

def main():
    # Menggunakan kamera perangkat (0 untuk kamera default)
    video_path = 0  # Ganti path video dengan 0 untuk menggunakan kamera perangkat

    # Membuka kamera untuk pemrosesan
    cap = cv2.VideoCapture(video_path)

    # Mendefinisikan koordinat x dari garis gawang (bagian kiri dari garis gawang)
    goal_line_x = 266  # Menyesuaikan ini dengan posisi garis gawang yang sebenarnya pada video

    # Variabel untuk melacak posisi bola pada frame sebelumnya
    last_ball_position = None
    stationary_frames_count = 0  # Hitung berapa banyak frame bola tidak bergerak

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Jika frame video tidak ada lagi, keluar dari loop

        # Menerapkan threshold warna untuk deteksi bola
        mask, result = apply_color_threshold(frame, HSV_RANGE['lower'], HSV_RANGE['upper'])

        # Mengonversi gambar ke grayscale dan menerapkan GaussianBlur untuk meningkatkan deteksi lingkaran
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)  # Menyesuaikan ukuran kernel sesuai kebutuhan

        # Mendeteksi lingkaran menggunakan HoughCircles
        circles = detect_circles(blurred)

        # Jika lingkaran terdeteksi, menggambar lingkaran pada frame
        if circles is not None:
            circles = np.uint16(np.around(circles))  # Membulatkan nilai koordinat lingkaran
            for circle in circles[0, :]:
                # Menyaring informasi center (x, y) dan radius dari lingkaran yang terdeteksi
                center_x, center_y, radius = circle[0], circle[1], circle[2]

                # Menggambar lingkaran yang lebih besar di sekitar bola (100% dari bola)
                cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), 3)  # Lingkaran hijau

                # Menggambar pusat lingkaran (pusat bola)
                cv2.circle(frame, (center_x, center_y), 2, (0, 0, 255), 3)  # Titik pusat berwarna merah

                # Menghitung posisi horizontal bola
                ball_position_x = center_x  # Koordinat x dari pusat bola

                # Menghitung persentase bola yang telah melewati garis gawang
                ball_cross_percentage = max(0, min(100, (goal_line_x - (ball_position_x - radius)) / (2 * radius) * 100))
                goal_percentage = ball_cross_percentage

                # Menampilkan persentase pada frame
                goal_text = f"Goal: {goal_percentage:.2f}%"
                cv2.putText(frame, goal_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Mengecek apakah bola sudah berhenti bergerak
                if last_ball_position is not None:
                    # Jika perubahan posisi bola kecil, anggap bola berhenti
                    if abs(last_ball_position - ball_position_x) < 5:  # Threshold pergerakan kecil
                        stationary_frames_count += 1
                    else:
                        stationary_frames_count = 0  # Reset jika bola bergerak

                # Jika bola sudah berhenti bergerak selama 10 frame berturut-turut, tampilkan pesan
                if stationary_frames_count >= 10:
                    if goal_percentage == 100:
                        cv2.putText(frame, "Var Checking Possible Goal", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, "Var Checking No Goal", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Simpan posisi bola untuk frame berikutnya
                last_ball_position = ball_position_x

        # Menggambar garis gawang di sisi kiri frame
        cv2.line(frame, (goal_line_x, 0), (goal_line_x, frame.shape[0]), (0, 0, 255), 3)  # Garis gawang berwarna merah

        # Menampilkan frame yang telah diproses
        cv2.imshow('Detected Ball and Goal Line', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break  # Jika tombol 'q' ditekan, keluar dari loop

    # Melepaskan capture video dan menutup semua jendela OpenCV
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
