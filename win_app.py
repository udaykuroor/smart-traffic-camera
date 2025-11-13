import streamlit as st
import os
import pandas as pd
import subprocess
from pathlib import Path
import sys
import time

st.set_page_config(page_title="Smart Traffic Camera", layout="wide")


def about_page():
    st.markdown("""
    ### About

    The **Smart Traffic Camera** is an intelligent video analytics tool that uses **computer vision**
    to detect, classify, and track vehicles in real-time.  
    It leverages **YOLOv8** for detection and **CLIP** for semantic understanding.

    #### Features
    - Vehicle detection, tracking, and classification  
    - Automatic speed estimation  
    - Calibration using pixel-to-meter ratio  
    - Processed video and CSV summary outputs  

    ---
    **Developed by:**  
    Uday, Anupam, Abdullah, Vaishak, Praveena, and Akhilesh
    """)


def home_page():
    st.markdown("""
    <div style="background-color:#1E88E5; padding: 1.5rem; border-radius: 10px; text-align:center;">
        <h1 style="color: white; margin-bottom: 0;">Smart Traffic Camera</h1>
        <p style="color: #E3F2FD; font-size: 1.1rem;">Vehicle Detection, Tracking & Analytics powered by YOLOv8 + CLIP</p>
    </div>
    <style>
    div.stSpinner {
        text-align: center;
        align-items: center;
        justify-content: center;
    }
    </style>

    """, unsafe_allow_html=True)
    st.write("")

    
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown("### Upload Traffic Feed")
        uploaded_video = st.file_uploader(
            "Upload your video file",
            type=["mp4", "mov", "avi"],
            label_visibility="collapsed"
        )

    with col2:
        st.markdown("### Calibration")
        pixels_per_meter = st.number_input(
            "Pixels per Meter",
            min_value=1.0,
            max_value=500.0,
            value=15.0,
            step=1.0,
            help="How many meters one pixel represents (for speed estimation)"
        )

    process_button = st.button("Process", use_container_width=True)

    
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

   
    if process_button:
        if uploaded_video is None:
            st.warning("Please upload a video first.")
        else:
            video_path = Path("uploads") / uploaded_video.name
            with open(video_path, "wb") as f:
                f.write(uploaded_video.read())

            
           
                
            with st.spinner("Processing video... This may take a few minutes ⏳"):
                cmd = [
                    sys.executable,
                    "detect_and_track_v2.py",
                 str(video_path),
                 str(pixels_per_meter)
                ]
            try:
                subprocess.run(cmd, check=True)
                time.sleep(1)
                st.success("✅ Processing complete!")
                st.balloons()     
            except subprocess.CalledProcessError:
                st.error("❌ Error occurred while running the detection script.")
            
            
          

            
            time.sleep(1)
            output_videos = sorted(Path(".").rglob("*.mp4"), key=os.path.getmtime, reverse=True)
            output_csvs = sorted(Path(".").rglob("*.csv"), key=os.path.getmtime, reverse=True)

            result_video_path = output_videos[0] if output_videos else None
            result_csv_path = output_csvs[0] if output_csvs else None

            if result_video_path and Path(result_video_path).exists():
                st.subheader("🎥 Processed Video")
                st.video(str(result_video_path))
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    pass
                with col2:
                    if result_video_path:
                        st.download_button(
                            "⬇️ Download Processed Video",
                            data=open(result_video_path, "rb"),
                            file_name=result_video_path.name
                        )
                with col3:
                    pass
            else:
                st.warning("⚠️ No output video found.")
        

            if result_csv_path:
                st.subheader("📊 Vehicle Summary")
                df = pd.read_csv(result_csv_path)
                st.dataframe(df, use_container_width=True)

                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    pass
                with col2:
                    st.download_button(
                        "⬇️ Download CSV",
                        data=open(result_csv_path, "rb"),
                        file_name=result_csv_path.name
                    )
                with col3:
                    pass
            else:
                st.warning("⚠️ No CSV results found.")


pages = {
    "Smart Traffic Camera": [
        st.Page(home_page, title="Home"),
        st.Page(about_page, title="About"),
    ]
}

pg = st.navigation(pages, position="top")
pg.run()
