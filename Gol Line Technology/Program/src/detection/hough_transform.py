import cv2
import numpy as np

def detect_circles(image):
    """
    Detect circles using the Hough Circle Transform.
    """
    circles = cv2.HoughCircles(
        image,                  # Gambar input yang akan diproses (biasanya gambar abu-abu)
        cv2.HOUGH_GRADIENT,     # Metode deteksi, menggunakan deteksi gradien untuk menemukan lingkaran
        dp=1,                   # Resolusi gambar output. 1 berarti resolusi output sama dengan gambar input
        minDist=50,             # Jarak minimum antara pusat dua lingkaran yang terdeteksi (untuk menghindari deteksi ganda)
        param1=50,              # Parameter untuk deteksi tepi (terkait dengan deteksi tepi menggunakan Canny Edge Detector)
        param2=30,              # Parameter akumulasi (threshold untuk akumulasi deteksi lingkaran di ruang parameter)
        minRadius=20,          # Radius minimum lingkaran yang akan terdeteksi
        maxRadius=150           # Radius maksimum lingkaran yang akan terdeteksi
    )

    return circles  
