import streamlit as st
import cv2
import numpy as np
import time

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="UAS Pengolahan Citra", layout="wide")

# =========================
# CSS CLEAN UI
# =========================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #eaf2ff, #ffffff);
    font-family: Arial;
}

/* HEADER */
.header {
    text-align: center;
    padding: 25px;
    background: linear-gradient(90deg, #0d47a1, #42a5f5);
    color: white;
    border-radius: 18px;
    margin-bottom: 20px;
    box-shadow: 0px 6px 15px rgba(0,0,0,0.2);
}

/* KPI */
.kpi {
    background: white;
    padding: 18px;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0px 6px 15px rgba(0,0,0,0.1);
}

.kpi-number {
    font-size: 34px;
    font-weight: bold;
    color: #0d47a1;
}

.kpi-text {
    font-size: 13px;
    color: #444;
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<div class="header">
    <h1>Deteksi Tepi dan Segmentasi Citra Biji Kopi</h1>
    <h3>Menggunakan OpenCV</h3>
    <p>UAS Pengolahan Citra Digital</p>
</div>
""", unsafe_allow_html=True)

# =========================
# IDENTITAS
# =========================
st.markdown("""
<div style="
text-align:center;
background:white;
padding:20px;
border-radius:15px;
box-shadow:0px 4px 12px rgba(0,0,0,0.1);
margin-bottom:20px;
">

<h3>ALVANIA AISYAH NUR FADHLILAH</h3>
<h4>2313020072</h4>

<p>Program Studi Teknik Informatika</p>
<p>Fakultas Teknik dan Ilmu Komputer</p>

</div>
""", unsafe_allow_html=True)
# =========================
# UPLOAD + START BUTTON
# =========================
uploaded_file = st.file_uploader("Upload Gambar Biji Kopi", type=["jpg","png","jpeg"])
start = st.button("Start Analysis")

if uploaded_file is not None and start:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # =========================
    # LOADING
    # =========================
    progress = st.progress(0)
    status = st.empty()

    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)
        status.text(f"Processing... {i+1}%")

    status.text("Selesai")

    # =========================
    # PROCESSING
    # =========================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # SOBEL
    sx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, 3)
    sy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, 3)
    sobel = cv2.convertScaleAbs(cv2.magnitude(sx, sy))

    # PREWITT FIX AMAN
    kx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=np.float32)
    ky = np.array([[1,1,1],[0,0,0],[-1,-1,-1]], dtype=np.float32)

    px = cv2.filter2D(blur, cv2.CV_32F, kx)
    py = cv2.filter2D(blur, cv2.CV_32F, ky)

    prewitt = np.sqrt(px**2 + py**2)
    prewitt = cv2.convertScaleAbs(prewitt)

    # HSV SEGMENTATION
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([0,35,40])
    upper = np.array([30,255,230])

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    seg = cv2.bitwise_and(img, img, mask=mask)
    seg_rgb = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)

    # CONTOUR
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = img.copy()
    count = 0

    for c in contours:
        if cv2.contourArea(c) > 700:
            count += 1
            cv2.drawContours(result, [c], -1, (0,255,0), 2)

    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    # =========================
    # KPI
    # =========================
    st.markdown("## Hasil Analisis")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-number">{count}</div>
            <div class="kpi-text">Biji Kopi Terdeteksi</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-number">{img.shape[0]}x{img.shape[1]}</div>
            <div class="kpi-text">Resolusi</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="kpi">
            <div class="kpi-number">OpenCV</div>
            <div class="kpi-text">AI Vision</div>
        </div>
        """, unsafe_allow_html=True)

    # =========================
    # GRID HASIL PROSES (SAMPING)
    # =========================
    st.markdown("## Hasil Proses")

    row1 = st.columns(4)

    with row1[0]:
        st.image(img_rgb, caption="Original")

    with row1[1]:
        st.image(gray, caption="Grayscale")

    with row1[2]:
        st.image(sobel, caption="Sobel")

    with row1[3]:
        st.image(prewitt, caption="Prewitt")

    row2 = st.columns(2)

    with row2[0]:
        st.image(seg_rgb, caption="Segmentasi HSV")

    with row2[1]:
        st.image(result_rgb, caption=f"Contour Result ({count})")

    st.success(f"Total biji kopi terdeteksi: {count}")