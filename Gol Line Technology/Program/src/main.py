import sys  # Library untuk berinteraksi dengan interpreter Python
import os  # Library untuk operasi file dan sistem
import cv2  # OpenCV digunakan untuk pemrosesan gambar dan video
import numpy as np  # Library untuk manipulasi array dan perhitungan numerik

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mengimpor fungsi dan konfigurasi yang diperlukan
from detection.color_threshold import apply_color_threshold  # Deteksi bola berdasarkan warna
from detection.hough_transform import detect_circles  # Deteksi lingkaran menggunakan Hough Transform
from utils.config import HSV_RANGE, HOUGH_PARAMS  # Parameter konfigurasi HSV dan Hough Transform

def main():
    video_path = "data/videos/Comp 1_3.mp4"  
    if not os.path.exists(video_path):  
        print(f"File video tidak ditemukan: {video_path}")
        return

    # Membuka video untuk diproses frame by frame
    cap = cv2.VideoCapture(video_path)

    # Posisi garis gawang pada frame (dalam piksel horizontal)
    goal_line_x = 266  # Sesuaikan dengan posisi garis gawang pada video

    # Variabel untuk melacak posisi bola dan status pergerakannya
    last_ball_position = None  # Posisi bola pada frame sebelumnya
    stationary_frames_count = 0  # Counter untuk mendeteksi bola yang berhenti bergerak

    # Loop utama untuk membaca dan memproses setiap frame video
    while cap.isOpened():
        ret, frame = cap.read()  # Membaca satu frame dari video
        if not ret:  # Jika frame tidak ada lagi, keluar dari loop
            break

        # Menerapkan threshold warna untuk mendeteksi bola
        mask, result = apply_color_threshold(frame, HSV_RANGE['lower'], HSV_RANGE['upper'])

        # Mengonversi hasil ke grayscale dan menerapkan GaussianBlur untuk memperbaiki deteksi
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)  # Mengurangi noise dengan blur

        # Mendeteksi lingkaran pada gambar yang sudah di-blur
        circles = detect_circles(blurred)

        # Jika lingkaran terdeteksi, proses informasi lingkaran
        if circles is not None:
            circles = np.uint16(np.around(circles))  # Membulatkan koordinat lingkaran
            for circle in circles[0, :]:
                # Mendapatkan koordinat pusat dan radius lingkaran
                center_x, center_y, radius = circle[0], circle[1], circle[2]

                # Menggambar lingkaran hijau pada bola
                cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), 3)

                # Menggambar titik merah di pusat lingkaran
                cv2.circle(frame, (center_x, center_y), 2, (0, 0, 255), 3)

                # Menghitung posisi horizontal bola
                ball_position_x = center_x  # Koordinat x dari bola

                # Menghitung persentase bola melewati garis gawang
                ball_cross_percentage = max(0, min(100, (goal_line_x - (ball_position_x - radius)) / (2 * radius) * 100))
                goal_percentage = ball_cross_percentage

                # Menampilkan persentase gol pada frame
                goal_text = f"Goal: {goal_percentage:.2f}%"
                cv2.putText(frame, goal_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Mengecek apakah bola berhenti bergerak
                if last_ball_position is not None:
                    # Membandingkan posisi bola saat ini dengan frame sebelumnya
                    if abs(last_ball_position - ball_position_x) < 5:  # Threshold untuk mendeteksi pergerakan kecil
                        stationary_frames_count += 1
                    else:
                        stationary_frames_count = 0  # Reset jika bola bergerak

                # Jika bola berhenti selama 10 frame berturut-turut, tampilkan pesan
                if stationary_frames_count >= 10:
                    if goal_percentage == 100:  # Bola sepenuhnya melewati garis
                        cv2.putText(frame, "Var Checking Possible Goal", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:  # Bola tidak sepenuhnya melewati garis
                        cv2.putText(frame, "Var Checking No Goal", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Menyimpan posisi bola untuk frame berikutnya
                last_ball_position = ball_position_x

        # Menggambar garis gawang merah pada frame
        cv2.line(frame, (goal_line_x, 0), (goal_line_x, frame.shape[0]), (0, 0, 255), 3)

        # Menampilkan frame yang telah diproses
        cv2.imshow('Detected Ball and Goal Line', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):  # Tekan 'q' untuk keluar dari loop
            break

    # Melepaskan video capture dan menutup semua jendela OpenCV
    cap.release()
    cv2.destroyAllWindows()

# Memanggil fungsi main jika file dijalankan sebagai program utama
if __name__ == "__main__":
    main()
