import streamlit as st
import os
import pandas as pd
import subprocess
from pathlib import Path
import sys
import time
from PIL import Image
import base64

st.set_page_config(page_title="DriveSense AI", layout="wide", page_icon="icon.png")
# Configuration
DETECTION_SCRIPT = "detect_track_final-4.py"
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
ASSETS_DIR = Path("assets")

if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'output_video_path' not in st.session_state:
    st.session_state.output_video_path = None
if 'output_csv_path' not in st.session_state:
    st.session_state.output_csv_path = None
if 'uploaded_video_name' not in st.session_state:
    st.session_state.uploaded_video_name = None

def about_page():
    st.markdown("""
    ### About

    **DriveSense AI** is an intelligent video analytics tool that uses **computer vision**
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
    - **Advanced filtering** by speed, type, and color

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
    Uday, Anupam, Abdullah, and Vaishak 
    
    **Project:** Computer Vision
    """)


def home_page():
   
    logo_path = Path("logo.png")
    logo_img_tag = ""
    if logo_path.exists():
        try:
            b64 = base64.b64encode(logo_path.read_bytes()).decode("utf-8")
            logo_img_tag = f'<img src="data:image/png;base64,{b64}" alt="DriveSense AI" style="height:72px; display:block; margin:0 auto;" />'
            b64 = base64.b64encode(logo_path.read_bytes()).decode("utf-8")
            logo_height = 72 
            logo_max_width = 1000 
            logo_border_radius = 12  
            logo_box_shadow = "0 4px 10px rgba(0,0,0,0.18)"
            logo_style = (
                f"height:{logo_height}px; max-width:{logo_max_width}px; display:block; "
                f"margin:0 auto; border-radius:{logo_border_radius}px; "
                f"box-shadow:{logo_box_shadow}; object-fit:contain;")

            logo_img_tag = f'<img src="data:image/png;base64,{b64}" alt="DriveSense AI" style="{logo_style}" />'
        except Exception:
            logo_img_tag = ""

    st.markdown(
        f"""
        <div style="background-color:#106CB6; padding: 1.0rem 1.5rem; border-radius: 10px; text-align:center;">
            {logo_img_tag or '<h1 style="color: white; margin:0;">DriveSense AI</h1>'}
            <p style="color: #E3F2FD; font-size: 1.05rem; margin-top:8px;">Vehicle Detection, Tracking & Analytics powered by AI</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
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

    # Show results if already processed
    if st.session_state.processed:
        display_results()
        
        # Add button to process new video
        st.write("")
        st.write("")
        if st.button("🔄 Process New Video", use_container_width=True):
            st.session_state.processed = False
            st.session_state.output_video_path = None
            st.session_state.output_csv_path = None
            st.session_state.uploaded_video_name = None
            st.rerun()
        return

    # Input section (only show if not processed)
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
        st.session_state.uploaded_video_name = uploaded_video.name

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
                    st.balloons()
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
                    
                    # Store in session state
                    st.session_state.processed = True
                    st.session_state.output_video_path = output_video_path
                    st.session_state.output_csv_path = output_csv_path
                    
                    # Rerun to show results
                    st.rerun()
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


def display_results():
    """Display the processed results with filtering options"""
    st.markdown("---")
    
    output_video_path = st.session_state.output_video_path
    output_csv_path = st.session_state.output_csv_path
    
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
                file_name=f"tracked_{st.session_state.uploaded_video_name}",
                mime="video/mp4",
                use_container_width=True
            )

    st.write("")
    
    # Display CSV results with filtering
    st.subheader("📊 Vehicle Analytics Summary")
    
    try:
        df = pd.read_csv(output_csv_path)
        
        if len(df) > 0:
            # Filter Section
            st.markdown("### 🔍 Filter Vehicles")
            
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            
            with filter_col1:
                filter_type = st.selectbox(
                    "Filter by Category",
                    ["None", "Speed Limit", "Vehicle Type", "Vehicle Color"],
                    key="filter_type_select"
                )
            
            # Initialize filtered dataframe
            filtered_df = df.copy()
            highlight_mask = pd.Series([False] * len(df))
            
            if filter_type == "Speed Limit":
                with filter_col2:
                    speed_limit = st.number_input(
                        "Speed Limit (km/h)",
                        min_value=0,
                        max_value=200,
                        value=60,
                        step=5,
                        key="speed_limit_input"
                    )
                with filter_col3:
                    show_violations = st.checkbox("Show only violations", value=True, key="speed_violations_check")
                
                if show_violations:
                    highlight_mask = df['Speed (km/h)'] > speed_limit
                    filtered_df = df[highlight_mask].copy()
                    st.info(f"Showing {len(filtered_df)} vehicles exceeding {speed_limit} km/h")
                else:
                    highlight_mask = df['Speed (km/h)'] > speed_limit
            
            elif filter_type == "Vehicle Type":
                with filter_col2:
                    available_types = df['Type'].unique().tolist()
                    selected_type = st.selectbox(
                        "Select Vehicle Type",
                        available_types,
                        key="vehicle_type_select"
                    )
                with filter_col3:
                    show_only = st.checkbox("Show only selected type", value=True, key="type_only_check")
                
                if show_only:
                    highlight_mask = df['Type'] == selected_type
                    filtered_df = df[highlight_mask].copy()
                    st.info(f"Showing {len(filtered_df)} {selected_type} vehicles")
                else:
                    highlight_mask = df['Type'] == selected_type
            
            elif filter_type == "Vehicle Color":
                with filter_col2:
                    available_colors = df['Color'].unique().tolist()
                    selected_color = st.selectbox(
                        "Select Vehicle Color",
                        available_colors,
                        key="vehicle_color_select"
                    )
                with filter_col3:
                    show_only = st.checkbox("Show only selected color", value=True, key="color_only_check")
                
                if show_only:
                    highlight_mask = df['Color'] == selected_color
                    filtered_df = df[highlight_mask].copy()
                    st.info(f"Showing {len(filtered_df)} {selected_color} vehicles")
                else:
                    highlight_mask = df['Color'] == selected_color
            
            st.write("")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Vehicles", len(filtered_df))
            with col2:
                st.metric("Avg Speed", f"{filtered_df['Speed (km/h)'].mean():.1f} km/h")
            with col3:
                st.metric("Max Speed", f"{filtered_df['Speed (km/h)'].max():.1f} km/h")
            with col4:
                unique_types = filtered_df['Type'].nunique()
                st.metric("Vehicle Types", unique_types)
            
            st.write("")
            
            # Show dataframe with conditional formatting
            if filter_type != "None" and not (filter_type == "Speed Limit" and show_violations) and not (filter_type == "Vehicle Type" and show_only) and not (filter_type == "Vehicle Color" and show_only):
                # Show all data but highlight matches
                def highlight_rows(row):
                    if highlight_mask[row.name]:
                        return ['background-color: #ffeb3b33'] * len(row)
                    return [''] * len(row)
                
                styled_df = df.style.apply(highlight_rows, axis=1)
                st.dataframe(styled_df, use_container_width=True)
            else:
                # Show filtered data
                st.dataframe(
                    filtered_df.style.highlight_max(subset=['Speed (km/h)'], color='lightcoral'),
                    use_container_width=True
                )
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Vehicle Type Distribution")
                type_counts = filtered_df['Type'].value_counts()
                st.bar_chart(type_counts)
            
            with col2:
                st.markdown("#### Color Distribution")
                color_counts = filtered_df['Color'].value_counts()
                st.bar_chart(color_counts)
            
            # Download CSV
            st.write("")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                csv_data = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Download Filtered Analytics CSV",
                    data=csv_data,
                    file_name=f"filtered_analytics_{st.session_state.uploaded_video_name}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_csv_button"
                )
        else:
            st.warning("No vehicles detected in the video.")
            
    except Exception as e:
        st.error(f"Error reading results: {str(e)}")

logo_path = Path("icon.png")

if logo_path.exists():
    try:
        b64 = base64.b64encode(logo_path.read_bytes()).decode("utf-8")
        nav_label = f'<img src="data:image/png;base64,{b64}" alt="DriveSense AI" style="height:48px; vertical-align:middle;" />'
    except Exception:
        nav_label = ""
pages = {
    "Home": home_page,
    "About": about_page,
}
if 'selected_page' not in st.session_state:
    st.session_state.selected_page = "Home"
col_left, col_title = st.columns([2,6])
st.markdown(
    """
    <style>
    /* Style all Streamlit buttons */
    div.stButton > button {
        background-color: #106CB6 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        font-weight: 600 !important;
        box-shadow: none !important;
        border: 1px solid rgba(0,0,0,0.05) !important;
    }
    div.stButton > button:hover {
        background-color: #0d57a0 !important;
        color: #ffffff !important;
    }
    div.stButton > button:focus {
        outline: 2px solid rgba(16,108,182,0.25) !important;
        box-shadow: 0 4px 10px rgba(16,108,182,0.2) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
with col_left:
    icon_col, home_col, about_col = st.columns([5, 6, 6])
    with icon_col:
        st.markdown(nav_label, unsafe_allow_html=True)
    with home_col:
        if st.button("Home", key="nav_home" ):
            st.session_state.selected_page = "Home"
    with about_col:
        if st.button("About", key="nav_about"):
            st.session_state.selected_page = "About"
st.markdown("---")
pages[st.session_state.selected_page]()
