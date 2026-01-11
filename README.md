# Gesture-Controlled-3D-Visualization-System
Project ini membangun sistem visualisasi objek 3D interaktif berbasis kamera. Sistem membaca gestur tangan kamu secara real time untuk mengontrol rotasi, skala, dan pergerakan objek virtual. Project ini fokus pada computer vision, interaksi manusia dan komputer, serta logika kontrol yang halus dan stabil.

FITUR UTAMA

Kontrol objek 3D menggunakan gestur tangan kanan dan kiri

Rotasi pitch, yaw, dan roll berbasis pergerakan tangan

Scaling dan translasi objek secara real time

Pilihan objek 3D. Globe, Saturn, DNA Helix, Galaxy

Auto rotation saat tidak ada interaksi

Tampilan UI minimalis dengan status sistem dan FPS

Background starfield untuk efek visual dinamis

TEKNOLOGI

Python

OpenCV

MediaPipe Hands

NumPy

CARA KERJA SISTEM

Kamera menangkap citra tangan kamu secara real time

MediaPipe mendeteksi landmark tangan dan klasifikasi kiri atau kanan

Tangan kanan mengontrol rotasi objek

Tangan kiri mengontrol posisi dan skala objek

Sistem menerapkan smoothing dan damping untuk gerakan halus

Objek 3D dirender sebagai point cloud dengan depth shading

KONTROL

Tangan kanan terbuka. Rotasi pitch dan yaw

Tangan kanan mengepal. Rotasi roll

Tangan kiri pinch. Translasi objek

Tangan kiri terbuka. Scaling objek

Tombol 1 sampai 4. Ganti objek 3D

Tombol A. Aktifkan atau matikan auto rotation

Tombol Q. Keluar dari program

KEGUNAAN

Simulasi antarmuka berbasis gestur

Eksplorasi konsep Human Computer Interaction

Dasar pengembangan AR dan sistem kontrol natural

Portofolio computer vision dan interactive system

CATATAN
Gunakan pencahayaan yang cukup agar deteksi tangan stabil. Kamera laptop atau webcam eksternal sudah cukup untuk menjalankan sistem ini.
