import streamlit as st
import os
import pandas as pd
import subprocess
from pathlib import Path
import sys
import time
from PIL import Image

# Custom page config with icon
try:
    icon = Image.open("assets/traffic_icon.png")
    st.set_page_config(page_title="Smart Traffic Camera", layout="wide")
except:
    st.set_page_config(page_title="Smart Traffic Camera", layout="wide")

# Configuration
DETECTION_SCRIPT = "detect_and_track_Sep11.py"
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
ASSETS_DIR = Path("assets")

def about_page():
    st.markdown("""
    ### About

    The **Smart Traffic Camera** is an intelligent video analytics tool that uses **computer vision**
    to detect, classify, and track vehicles in real-time.  
    It leverages **YOLOv8** for detection, **ByteTrack** for tracking, and **CLIP** for classification.

    #### Features
    - **Vehicle detection and tracking** with IDs
    - **Color detection** 
    - **Automatic speed estimation** 
    - **Vehicle classification** (Car, SUV, Van, Pickup)
    - **Automatic calibration** using lane width detection
    - **Comprehensive CSV analytics** export
    - **Video output**

    #### Technology Stack
    - **YOLOv8** - Object detection
    - **ByteTrack** - Multi-object tracking
    - **CLIP** - Zero-shot vehicle classification
    - **OpenCV** - Video processing & lane detection
    - **Streamlit** - Web interface

    #### How Calibration Works
    Instead of manually calculating pixels-per-meter, our system:
    1. Automatically detects lane markings in your video
    2. Measures lane width in pixels
    3. Compares to your specified lane width (default: 3.65m)
    4. Calculates accurate speed measurements

    ---
    **Developed by:**  
    Uday, Anupam, Abdullah, Vaishak, Praveena, and Akhilesh
    
    **Project:** Computer Vision
    """)


def home_page():
    # Add logo if it exists
    if (ASSETS_DIR / "logo.png").exists():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(str(ASSETS_DIR / "logo.png"), width=250)
        st.write("")
    
    st.markdown("""
    <div style="background-color:#1E88E5; padding: 1.5rem; border-radius: 10px; text-align:center;">
        <h1 style="color: white; margin-bottom: 0;">Smart Traffic Camera</h1>
        <p style="color: #E3F2FD; font-size: 1.1rem;">Vehicle Detection, Tracking & Analytics powered by AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    st.write("")

    # Check if detection script exists
    if not Path(DETECTION_SCRIPT).exists():
        st.error(f"Detection script '{DETECTION_SCRIPT}' not found.")
        st.info("Please ensure detect_and_track_v2.py is in the same directory as this Streamlit app.")
        return

    # Check if lane detector exists
    if not Path("lane_detector.py").exists():
        st.error(f"Lane detector script 'lane_detector.py' not found.")
        st.info("Please ensure lane_detector.py is in the same directory.")
        return

    # Input section
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown("### 📹 Upload Traffic Feed")
        uploaded_video = st.file_uploader(
            "Choose a video file (MP4, MOV, or AVI)",
            type=["mp4", "mov", "avi"],
            help="Upload a traffic camera video for analysis"
        )
        
        if uploaded_video:
            st.info(f"📁 **File:** {uploaded_video.name}")
            st.info(f"**Size:** {uploaded_video.size / (1024*1024):.2f} MB")

    with col2:
        st.markdown("### ⚙️ Calibration Settings")
        
        lane_width_meters = st.number_input(
            "Lane Width (meters)",
            min_value=2.0,
            max_value=6.0,
            value=3.65,
            step=0.05,
            help="Standard lane width for accurate speed calculation. System will automatically detect lane width in pixels."
        )
        
        st.info("💡 **Standard widths:**\n- Highway: 3.65m (12 ft)\n- Urban: 3.0-3.5m\n- Narrow: 2.75m")
        
        with st.expander("ℹ️ How does calibration work?"):
            st.markdown("""
            Our system automatically:
            1. **Detects lane markings** in your video using edge detection
            2. **Measures lane width** in pixels
            3. **Calculates conversion factor** (pixels ÷ meters)
            4. **Estimates vehicle speeds** accurately
            
            No manual measurement needed! Just specify your lane width.
            """)

    st.write("")
    process_button = st.button("Process Video", use_container_width=True, type="primary")

    # Create directories
    UPLOAD_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Processing logic
    if process_button:
        if uploaded_video is None:
            st.warning("⚠️ No video has been uploaded.")
            return

        # Save uploaded video
        video_path = UPLOAD_DIR / uploaded_video.name
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())
        
        st.success(f"✅ Video uploaded: {video_path.name}")

        # Define output paths
        output_video_path = OUTPUT_DIR / "output_tracking.mp4"
        output_csv_path = OUTPUT_DIR / "vehicle_tracking_results.csv"

        # Remove old outputs if they exist
        if output_video_path.exists():
            output_video_path.unlink()
        if output_csv_path.exists():
            output_csv_path.unlink()

        # Prepare command
        cmd = [
            sys.executable,
            DETECTION_SCRIPT,
            str(video_path),
            str(lane_width_meters)
        ]

        # Run processing
        st.markdown("---")
        st.subheader("🔄 Processing Video...")
        
        with st.spinner("Running automatic calibration and vehicle tracking. This may take several minutes."):
            try:
                # Run the detection script
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace'
                )
                
                if result.returncode == 0:
                    st.success("✅ Processing complete!")
                    
                    # Wait for file to be fully written
                    time.sleep(2)
                    
                    # Show calibration info
                    with st.expander("📏 Calibration & Processing Details"):
                        if result.stdout:
                            # Extract and display calibration info
                            lines = result.stdout.split('\n')
                            for line in lines:
                                if any(keyword in line.lower() for keyword in ['calibration', 'lane width', 'pixels per meter', 'detected']):
                                    st.text(line)
                else:
                    st.error(f"Processing failed with exit code {result.returncode}")
                    with st.expander("Show error details"):
                        st.code(result.stderr if result.stderr else result.stdout)
                    return
                    
            except FileNotFoundError:
                st.error(f"Could not find Python or detection script")
                return
            except Exception as e:
                st.error(f"Error occurred: {str(e)}")
                return

        # Display results
        st.markdown("---")
        
        # Check if outputs exist
        if not output_video_path.exists():
            st.error("Output video not found. Check the detection script output.")
            return
        
        if not output_csv_path.exists():
            st.error("Results CSV not found. Check the detection script output.")
            return

        # Display video
        st.subheader("🎥 Processed Video")
        st.video(str(output_video_path))
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with open(output_video_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Processed Video",
                    data=f.read(),
                    file_name=f"tracked_{uploaded_video.name}",
                    mime="video/mp4",
                    use_container_width=True
                )

        st.write("")
        
        # Display CSV results
        st.subheader("📊 Vehicle Analytics Summary")
        
        try:
            df = pd.read_csv(output_csv_path)
            
            if len(df) > 0:
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Vehicles", len(df))
                with col2:
                    st.metric("Avg Speed", f"{df['Speed (km/h)'].mean():.1f} km/h")
                with col3:
                    st.metric("Max Speed", f"{df['Speed (km/h)'].max():.1f} km/h")
                with col4:
                    unique_types = df['Type'].nunique()
                    st.metric("Vehicle Types", unique_types)
                
                st.write("")
                
                # Show dataframe
                st.dataframe(
                    df.style.highlight_max(subset=['Speed (km/h)'], color='lightcoral'),
                    use_container_width=True
                )
                
                # Charts
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Vehicle Type Distribution")
                    type_counts = df['Type'].value_counts()
                    st.bar_chart(type_counts)
                
                with col2:
                    st.markdown("#### Color Distribution")
                    color_counts = df['Color'].value_counts()
                    st.bar_chart(color_counts)
                
                # Download CSV
                st.write("")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    csv_data = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "⬇️ Download Analytics CSV",
                        data=csv_data,
                        file_name=f"analytics_{uploaded_video.name}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            else:
                st.warning("No vehicles detected in the video.")
                
        except Exception as e:
            st.error(f"Error reading results: {str(e)}")


# Page navigation
pages = {
    "Smart Traffic Camera": [
        st.Page(home_page, title="Home", icon="🏠"),
        st.Page(about_page, title="About", icon="ℹ️"),
    ]
}

pg = st.navigation(pages)
pg.run()