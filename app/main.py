import streamlit as st
from ultralytics import YOLO
from sqlalchemy import create_engine, text
from datetime import datetime
import pandas as pd
import boto3
import uuid
from pathlib import Path

# ==========================================================
# CONFIGURATION
# ==========================================================
DB_URL = "postgresql+psycopg2://postgres:Admin@localhost:5432/jio_advision"
BUCKET_NAME = "jioadvision-uploads"
AWS_REGION = "ap-south-1"
MODEL_PATH = Path(r"C:\Users\infan\OneDrive\Desktop\Final Project- Jio_AdVision_Analytics\Jio_AdVision_Analytics\model\best1.pt")

# AWS & DB setup
s3 = boto3.client("s3", region_name=AWS_REGION)
engine = create_engine(DB_URL)

# Load YOLO model once
model = YOLO(MODEL_PATH)

# ==========================================================
# STREAMLIT UI
# ==========================================================
st.set_page_config(page_title="Jio AdVision Auto Tracker", layout="wide")
st.title("🏏 Jio Hotstar AdVision – Automatic Brand Tracking System")

st.markdown("### Upload Raw Match Video and Enter Match Details")

# --- Form ---
with st.form("match_form"):
    match_id = st.text_input("Match ID (e.g., JIO-MATCH-2025-001)")
    home_team = st.text_input("Home Team")
    away_team = st.text_input("Away Team")
    match_type = st.selectbox("Match Type", ["T20", "ODI", "Test"])
    location = st.text_input("Venue / Location")
    start_time = st.time_input("Match Start Time")
    end_time = st.time_input("Match End Time")
    winner = st.text_input("Winner")
    raw_video = st.file_uploader("🎥 Upload Raw Match Video", type=["mp4", "mov", "avi"])
    submitted = st.form_submit_button("🚀 Process & Save Match")

# ==========================================================
# LOGIC
# ==========================================================
if submitted:
    if not all([match_id, home_team, away_team, match_type, location, winner, raw_video]):
        st.error("⚠️ Please fill all fields and upload a video.")
    else:
        with st.spinner("⏳ Running YOLO tracking and uploading videos..."):
            match_uuid = str(uuid.uuid4())

            # Save uploaded raw video temporarily
            temp_video_path = Path(f"temp_{match_id}_raw.mp4")
            temp_video_path.write_bytes(raw_video.read())

            # === Step 1: Run YOLO tracking automatically ===
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"{match_id}_{timestamp}"
            output_dir = Path(f"runs/track/{folder_name}")

            model.track(
                source=str(temp_video_path),
                show=False,
                save=True,
                imgsz=480,
                conf=0.4,
                vid_stride=3,
                project="runs/track",
                name=folder_name,
                exist_ok=True
            )

            tracked_video_path = output_dir / f"{temp_video_path.stem}.mp4"
            if not tracked_video_path.exists():
                # YOLO sometimes names output as "source.mp4"
                tracked_video_path = list(output_dir.glob("*.mp4"))[0]

            # === Step 2: Upload both videos to S3 ===
            raw_s3_key = f"raw_videos/{match_id}_raw.mp4"
            s3.upload_file(str(temp_video_path), BUCKET_NAME, raw_s3_key)
            raw_video_s3_url = f"s3://{BUCKET_NAME}/{raw_s3_key}"

            tracked_s3_key = f"tracked_videos/{match_id}_tracked.mp4"
            s3.upload_file(str(tracked_video_path), BUCKET_NAME, tracked_s3_key)
            tracked_video_s3_url = f"s3://{BUCKET_NAME}/{tracked_s3_key}"

            # Clean temp files
            temp_video_path.unlink(missing_ok=True)

            # === Step 3: Save to PostgreSQL ===
            query = text("""
                INSERT INTO matches (
                    id, match_id, home_team, away_team, match_type, location,
                    start_time, end_time, winner,
                    video_s3_key, created_at, updated_at
                )
                VALUES (
                    :id, :match_id, :home_team, :away_team, :match_type, :location,
                    :start_time, :end_time, :winner,
                    :video_s3_key, :created_at, :updated_at
                )
            """)

            with engine.begin() as conn:
                conn.execute(query, {
                    "id": match_uuid,
                    "match_id": match_id,
                    "home_team": home_team,
                    "away_team": away_team,
                    "match_type": match_type,
                    "location": location,
                    "start_time": datetime.combine(datetime.today(), start_time),
                    "end_time": datetime.combine(datetime.today(), end_time),
                    "winner": winner,
                    "video_s3_key": f"{raw_video_s3_url}, {tracked_video_s3_url}",
                    "created_at": datetime.now(),
                    "updated_at": datetime.now()
                })

        st.success("✅ Match processed, tracked, uploaded, and saved successfully!")

# ==========================================================
# DISPLAY MATCHES TABLE
# ==========================================================
st.markdown("### 📋 All Stored Matches")

try:
    with engine.begin() as conn:
        df = pd.read_sql("SELECT * FROM matches ORDER BY created_at DESC", conn)
    st.dataframe(df, use_container_width=True)
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
