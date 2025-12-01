import streamlit as st
from sqlalchemy import create_engine, text
import pandas as pd
from datetime import datetime, timedelta
import json
import uuid
from pathlib import Path
import time
import boto3
from ultralytics import YOLO
import subprocess
from collections import defaultdict, Counter
import numpy as np
import cv2
import os
import math
# Plots
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

import tempfile, os
tmp = tempfile.gettempdir()



# Chat Bot
import google.generativeai as genai
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])


# ==========================================================
# CONFIG - update these values for your environment
# ==========================================================
DB_URL = st.secrets["DB_URL"]
BUCKET_NAME = "jioadvision-uploads"
AWS_REGION = "ap-south-1"
MODEL_PATH = Path(r"C:\Users\Infant Joshva\Desktop\Final Project - AdSpotter-AI\AdSpotter-AI\model\best1.pt")
FFMPEG_BIN = "ffmpeg"

MERGE_GAP_THRESHOLD = 1.0
PADDING_BEFORE = 0.5
PADDING_AFTER = 0.5
MAX_SAMPLE_FPS = 30.0


# ==========================================================
# INIT clients and model
# ==========================================================
engine = create_engine(DB_URL)
s3 = boto3.client("s3", region_name=AWS_REGION)
model = YOLO(MODEL_PATH)

st.set_page_config(page_title="AdSpotter AI", page_icon="🎬", layout="wide")
st.sidebar.title("📌 Navigation")
menu = st.sidebar.radio("Go to:", ["📄 About", "🧭 Insights & Metrics"])


# ==========================================================
# Ensure matches table exists
# ==========================================================
create_matches_table_sql = """
CREATE TABLE IF NOT EXISTS matches (
    id UUID PRIMARY KEY,
    match_id VARCHAR(50),
    home_team VARCHAR(100),
    away_team VARCHAR(100),
    match_type VARCHAR(20),
    location VARCHAR(100),
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    winner VARCHAR(100),
    raw_video_s3_key VARCHAR(255),
    tracked_video_s3_key VARCHAR(255),
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
"""
with engine.begin() as conn:
    conn.execute(text(create_matches_table_sql))


# ==========================================================
# Ensure brand_detections table exists
# ==========================================================
create_table_sql = """
CREATE TABLE IF NOT EXISTS brand_detections (
    id UUID PRIMARY KEY,
    match_id VARCHAR(50),
    brand_name VARCHAR(100),
    start_time_sec FLOAT,
    end_time_sec FLOAT,
    duration_sec FLOAT,
    placement VARCHAR(50),
    chunk_s3key VARCHAR(255),
    confidence FLOAT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
"""
with engine.begin() as conn:
    conn.execute(text(create_table_sql))

# ==========================================================
# Ensure brand_aggregates table exists
# ==========================================================
create_agg_table_sql = """
CREATE TABLE IF NOT EXISTS brand_aggregates (
    id UUID PRIMARY KEY,
    match_id VARCHAR(50),
    brand_name VARCHAR(100),
    total_duration_seconds FLOAT,
    visibility_ratio FLOAT,
    placement_distribution JSON,
    first_seen TIMESTAMP,
    last_seen TIMESTAMP,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
"""
with engine.begin() as conn:
    conn.execute(text(create_agg_table_sql))


# ==========================================================
# Utilities
# ==========================================================

def generate_brand_aggregates(match_id, match_start_dt, match_duration):
    """
    Build and insert aggregated rows into brand_aggregates for the given match.
    - match_start_dt: datetime object for the match start (so we can convert seconds -> timestamp)
    - match_duration: total video duration in seconds (float)
    """
    sql = text("""
        SELECT brand_name, duration_sec, placement, start_time_sec, end_time_sec
        FROM brand_detections
        WHERE match_id = :m
    """)
    with engine.begin() as conn:
        rows = conn.execute(sql, {"m": match_id}).fetchall()

    if not rows:
        return

    df = pd.DataFrame(rows, columns=[
        "brand_name", "duration_sec", "placement", "start_time_sec", "end_time_sec"
    ])

    brands = df["brand_name"].unique().tolist()

    for brand in brands:
        sub = df[df["brand_name"] == brand]

        total_duration = float(sub["duration_sec"].sum())
        visibility_ratio = float(total_duration / match_duration) if match_duration > 0 else 0.0

        # placement distribution as normalized counts (dict)
        placement_counts = sub["placement"].value_counts(normalize=True).to_dict()

        # first/last seen as timestamps (match_start + seconds)
        first_sec = float(sub["start_time_sec"].min())
        last_sec = float(sub["end_time_sec"].max())
        first_ts = match_start_dt + timedelta(seconds=first_sec)
        last_ts = match_start_dt + timedelta(seconds=last_sec)

        agg_id = str(uuid.uuid4())

        insert_sql = text("""
            INSERT INTO brand_aggregates (
                id, match_id, brand_name, total_duration_seconds,
                visibility_ratio, placement_distribution,
                first_seen, last_seen, created_at, updated_at
            ) VALUES (
                :id, :m, :bn, :td, :vr, :pd, :fs, :ls, :c, :u
            )
        """)
        with engine.begin() as conn:
            conn.execute(insert_sql, {
                "id": agg_id,
                "m": match_id,
                "bn": brand,
                "td": total_duration,
                "vr": visibility_ratio,
                # ensure JSON serializable: use json.dumps
                "pd": json.dumps(placement_counts),
                "fs": first_ts,
                "ls": last_ts,
                "c": datetime.now(),
                "u": datetime.now()
            })


def generate_match_id():
    query = "SELECT match_id FROM matches ORDER BY created_at DESC LIMIT 1"
    with engine.begin() as conn:
        row = conn.execute(text(query)).fetchone()

    # If table is empty -> start from 0001
    if not row:
        return "JIO-MATCH-2025-0001"

    last = str(row.match_id).strip()

    # Extract last number safely
    try:
        last_num = int(last.split("-")[-1])
    except Exception:
        last_num = 0

    new_id = f"JIO-MATCH-2025-{last_num + 1:04}"
    return new_id


def get_video_props(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return MAX_SAMPLE_FPS, 0, 0.0, None, None
    fps = cap.get(cv2.CAP_PROP_FPS) or MAX_SAMPLE_FPS
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration = frame_count / fps if fps > 0 else 0.0
    return float(fps), frame_count, float(duration), width, height


def safe_extract_coords(box, frame_w, frame_h):
    try:
        xy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(float, xy)
    except Exception:
        try:
            vals = list(box.xyxy[0])
            x1, y1, x2, y2 = float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])
        except Exception:
            return None, None, None
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    area = w * h
    rel_area = area / (frame_w * frame_h) if frame_w and frame_h else 0.0
    cx = x1 + w / 2
    cy = y1 + h / 2
    return cx / frame_w, cy / frame_h, rel_area


def placement_heuristic(rel_cx, rel_cy, rel_area):
    if rel_cy is None:
        return "other"
    if rel_cy < 0.18:
        return "overlay"
    if rel_area > 0.12:
        return "boundary"
    if rel_area < 0.02 and rel_cy > 0.35:
        return "jersey"
    if rel_cy > 0.6:
        return "ground"
    return "other"


def merge_detections(detections, gap=MERGE_GAP_THRESHOLD):
    byb = defaultdict(list)
    for d in detections:
        byb[d["brand"]].append((d["t"], d["conf"], d["placement"]))
    out = {}
    for b, items in byb.items():
        items.sort()
        intervals = []
        cur_s = None
        cur_e = None
        confs = []
        places = []
        for t, c, p in items:
            if cur_s is None:
                cur_s = cur_e = t
                confs = [c]
                places = [p]
            else:
                if t - cur_e <= gap:
                    cur_e = t
                    confs.append(c)
                    places.append(p)
                else:
                    intervals.append({"start": cur_s, "end": cur_e, "confs": confs[:], "places": places[:]})
                    cur_s = cur_e = t
                    confs = [c]
                    places = [p]
        if cur_s is not None:
            intervals.append({"start": cur_s, "end": cur_e, "confs": confs[:], "places": places[:]})
        out[b] = intervals
    return out

def ffmpeg_trim_and_upload(video, start, end, match_id, detid):
    out = Path(f"tmp_{detid}.mp4")
    dur = max(0.01, end - start)

    cmd = [
        FFMPEG_BIN, "-y",
        "-ss", f"{start:.3f}",
        "-i", str(video),
        "-t", f"{dur:.3f}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-c:a", "aac",
        str(out)
    ]

    try:
        # SAFER: ensure ffmpeg fully releases file before unlink
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        process.wait()
    except Exception:
        if out.exists():
            out.unlink()
        return None

    key = f"{match_id}/chunks/{detid}.mp4"

    try:
        s3.upload_file(str(out), BUCKET_NAME, key)
    except Exception:
        if out.exists():
            out.unlink()
        return None

    # Give Windows a moment to release FFmpeg file lock
    import time, gc
    gc.collect()
    time.sleep(0.1)

    try:
        out.unlink()
    except PermissionError:
        # Retry once after slight delay
        time.sleep(0.2)
        gc.collect()
        try:
            out.unlink()
        except Exception:
            pass  # still locked, ignore

    return key



def insert_detection_row(match_id, brand, start, end, placement, key, conf):
    did = str(uuid.uuid4())
    dur = end - start
    sql = text("""
    INSERT INTO brand_detections 
    (id,match_id,brand_name,start_time_sec,end_time_sec,duration_sec,placement,chunk_s3key,confidence,created_at,updated_at)
    VALUES 
    (:id,:m,:b,:s,:e,:d,:p,:k,:c,:cr,:u)
    """)
    with engine.begin() as conn:
        conn.execute(sql, {
            "id": did, "m": match_id, "b": brand,
            "s": float(start), "e": float(end), "d": float(dur),
            "p": placement, "k": key, "c": float(conf),
            "cr": datetime.now(), "u": datetime.now()
        })
    return did



def generate_charts_pdf(match_id):
    import tempfile, os

    pdf_path = f"Brand_Report_{match_id}.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    title = Paragraph(f"<b>Brand Exposure Report – Match {match_id}</b>", styles["Title"])
    elements.append(title)
    elements.append(Spacer(1, 20))

    # Temp directory
    tmp = tempfile.gettempdir()

    # EXACT MATCH (based on your saved images)
    charts = [
        ("Brand Exposure Duration",        os.path.join(tmp, "fig1_exposure.png")),
        ("Visibility Ratio",               os.path.join(tmp, "fig2_ratio.png")),
        ("Placement Distribution",         os.path.join(tmp, "fig3_pie.png")),
        ("Detections Count",               os.path.join(tmp, "fig4_detect_count.png")),
        ("Average Duration",               os.path.join(tmp, "fig5_avg_duration.png")),
        ("Confidence Distribution",        os.path.join(tmp, "fig6_confidence.png")),
        ("Placement Heatmap",              os.path.join(tmp, "fig7_heatmap.png")),
    ]

    for label, img_path in charts:
        elements.append(Paragraph(f"<b>{label}</b>", styles["Heading2"]))

        if os.path.exists(img_path):
            elements.append(Image(img_path, width=400, height=250))
        else:
            elements.append(Paragraph(f"<i>Missing: {os.path.basename(img_path)}</i>", styles["BodyText"]))

        elements.append(Spacer(1, 18))

    doc.build(elements)
    return pdf_path




# ==========================================================
# Stable match_id (session)
# ==========================================================
if "current_match_id" not in st.session_state:
    st.session_state.current_match_id = generate_match_id()

# holds the last match that finished processing (used for showing table/charts)
if "last_completed_match_id" not in st.session_state:
    st.session_state.last_completed_match_id = None


# ==========================================================
# PAGE 1 — ABOUT
# ==========================================================
if menu == "📄 About":

    st.markdown("""
    <style>
    /* Premium White Heading */
    .premium-title {
        color: #FFFFFF; 
        font-size: 32px;
        font-weight: 700;
        padding-bottom: 8px;
    }

    /* Subheadings – clean white */
    .premium-subtitle {
        color: #FFFFFF;
        font-size: 24px;
        font-weight: 600;
        padding-top: 18px;
    }

    /* Body text – soft white */
    .premium-body {
        font-size: 16px;
        line-height: 1.6;
        color: #F2F2F2;  /* softer white for smooth readability */
    }
    </style>
    """, unsafe_allow_html=True)

    # -------------------- MAIN ABOUT CONTENT --------------------
    st.markdown("""
    <div class="premium-title">⚡AdSpotter AI – Sports Sponsorship Intelligence</div>

    <div class="premium-body">
    A next-generation system that automatically detects, tracks, and analyzes brand advertisements in cricket match broadcasts — delivering accurate, fast, and audit-ready insights for sponsors and broadcasters.
    </div>

    <br>
    <div class="premium-subtitle">Core Features 💡</div>

    <div class="premium-body">
                
    🔍 <b>Automated Brand Detection</b> &nbsp;&nbsp; YOLOv8 identifies sponsor logos across frames.<br>

    🕒 <b>Timestamp & Duration Metrics</b> &nbsp;&nbsp; Calculates how long each brand stays visible.<br>

    🎯 <b>Placement Classification</b> &nbsp;&nbsp; Jersey • Boundary • Overlay • Ground.<br>

    ✂️ <b>Video Chunk Extraction</b> &nbsp;&nbsp; Creates brand-wise clips and uploads to S3.<br>

    📀 <b>Structured Storage</b> &nbsp;&nbsp; All detections & aggregates saved in PostgreSQL.<br>

    📊 <b>Interactive Dashboard</b> &nbsp;&nbsp; View match data, detections, and brand exposure summaries.
    </div>

    <div class="premium-subtitle">Business Value 💼</div>

    <div class="premium-body">
    📈 Precise sponsor ROI measurement<br>
    ⚡ Fast & automated reporting<br>
    🏟 Optimized ad placement decisions<br>
    🔍 Easy audit with brand-specific clips<br>
    🔄 Scalable for full tournaments<br>
    </div>

    <div class="premium-subtitle">Output You Get 🧩</div>

    <div class="premium-body">
    • Brand exposure duration<br>
    • Visibility ratio<br>
    • Placement distribution<br>
    • Brand-level summary tables<br>
    • Match metadata<br>
    • Processed video chunks in S3<br>
    </div>

    <div class="premium-subtitle">🚀 Impact</div>

    <div class="premium-body">
    Reliable, real-time advertising analytics that reduces manual effort and gives brands data-backed insights.
    </div>
    """, unsafe_allow_html=True)

    # -------------------- FOOTER --------------------
    st.markdown("""
        <hr>
        <div style="text-align: center;">
            <p style="font-size: 13px;">⚡Jio Hotstar AdVision & Analytics | Built by <strong>Infant Joshva</strong></p>
            <a href="https://github.com/Infant-Joshva" target="_blank" style="text-decoration: none; margin: 0 10px;">🐙 GitHub</a>
            <a href="https://www.linkedin.com/in/infant-joshva" target="_blank" style="text-decoration: none; margin: 0 10px;">🔗 LinkedIn</a>
            <a href="mailto:infantjoshva2024@gmail.com" style="text-decoration: none; margin: 0 10px;">📩 Contact</a>
        </div>
    """, unsafe_allow_html=True)



# ==========================================================
# PAGE 2 — DASHBOARD
# ==========================================================
elif menu == "🧭 Insights & Metrics":
    st.title("🏏 Exposure Intelligence & Metrics")
    tab_up, tab_ch, tab_tb, tab_bot, tab_ad = st.tabs(["Ingestion & Tracking", "Visual Analytics", "Brand Exposure Insights", "AI Chat Bot","System Controls"]) 

    # =============== Upload & Track Tab ===============
    with tab_up:
        st.header("Upload & Track")
        st.caption(f"📍Current Match ID: **{st.session_state.current_match_id}**")

        with st.form("upform"):
            home = st.text_input("Home Team")
            away = st.text_input("Away Team")
            mtype = st.selectbox("Match Type", ["T20","ODI","Test"]) 
            loc = st.text_input("Location")
            stt = st.time_input("Start Time")
            ett = st.time_input("End Time")
            win = st.text_input("Winner")
            raw = st.file_uploader("Upload Video",type=["mp4","mov","avi","mkv"]) 
            btn = st.form_submit_button("🚀 Process Video")

        if btn:
            if not all([home,away,mtype,loc,win,raw]):
                st.error("Fill all fields")
            else:
                match_id = st.session_state.current_match_id
                with st.spinner("Processing video...⏳"):
                    mid = str(uuid.uuid4())

                    # Save raw locally
                    temp = Path(f"temp_{match_id}.mp4")
                    temp.write_bytes(raw.read())

                    fps, fc, dur, W, H = get_video_props(temp)

                    # TRACK MODE
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    folder = f"{match_id}_{timestamp}"
                    outdir = Path(f"runs/track/{folder}")
                    outdir.mkdir(parents=True, exist_ok=True)

                    model.track(
                        source=str(temp),
                        show=False,
                        save=True,
                        imgsz=480,
                        vid_stride=10,
                        project="runs/track",
                        name=folder,
                        exist_ok=True
                    )

                    # find tracked file
                    time.sleep(1)
                    files = list(outdir.rglob("*.mp4")) + list(outdir.rglob("*.avi")) + list(outdir.rglob("*.mov"))
                    trk = files[0] if files else None

                    conv = None
                    if trk:
                        conv = outdir / f"{match_id}_tracked.mp4"
                        subprocess.run([FFMPEG_BIN, "-y", "-i", str(trk), "-vcodec", "libx264", "-acodec", "aac", str(conv)])

                    raw_key = f"{match_id}/raw/{match_id}.mp4"
                    trk_key = f"{match_id}/track/{match_id}_tracked.mp4" if conv else None

                    s3.upload_file(str(temp), BUCKET_NAME, raw_key)
                    if conv:
                        s3.upload_file(str(conv), BUCKET_NAME, trk_key)

                    trk_url = None
                    if trk_key:
                        trk_url = s3.generate_presigned_url("get_object", Params={"Bucket": BUCKET_NAME, "Key": trk_key}, ExpiresIn=3600)

                    # Insert matches row

                    # create proper datetimes for match start/end (reuse later)
                    match_start_dt = datetime.combine(datetime.today(), stt)
                    match_end_dt = datetime.combine(datetime.today(), ett)

                    with engine.begin() as conn:
                        conn.execute(text("""
                            INSERT INTO matches 
                            (id,match_id,home_team,away_team,match_type,location,
                             start_time,end_time,winner,raw_video_s3_key,tracked_video_s3_key,
                             created_at,updated_at)
                            VALUES 
                            (:i,:m,:h,:a,:t,:l,:st,:et,:w,:rk,:tk,:c,:u)
                        """), {
                            "i": mid, "m": match_id, "h": home, "a": away, "t": mtype,
                            "l": loc,
                            "st": match_start_dt,
                            "et": match_end_dt,
                            "w": win,
                            "rk": raw_key, "tk": trk_key,
                            "c": datetime.now(), "u": datetime.now()
                        })

                    st.caption("Tracking done ✅. Finalizing video chunks…⏱️")
                    if trk_url:
                        st.video(trk_url)

                    # DETECTION STREAM MODE
                    dets = []
                    idx = 0
                    fps = fps if fps > 0 else MAX_SAMPLE_FPS

                    for res in model(source=str(temp), stream=True):
                        t = idx / fps
                        boxes = getattr(res, "boxes", None)
                        img = getattr(res, "orig_img", None)
                        h = img.shape[0] if img is not None else H
                        w = img.shape[1] if img is not None else W

                        if boxes:
                            for b in boxes:
                                try:
                                    cid = int(b.cls[0])
                                    brand = model.names.get(cid, str(cid))
                                except Exception:
                                    brand = "unknown"
                                try:
                                    conf = float(b.conf[0])
                                except Exception:
                                    conf = 0.0

                                cx, cy, ar = safe_extract_coords(b, w, h)
                                place = placement_heuristic(cx, cy, ar)

                                dets.append({"brand": brand, "t": t, "conf": conf, "placement": place})
                        idx += 1

                    merged = merge_detections(dets)
                    count = 0
                    _, _, td, _, _ = get_video_props(temp)

                    for brand, ints in merged.items():
                        for itv in ints:
                            s = float(itv["start"])
                            e = float(itv["end"])
                            s_pad = max(0.0, s - PADDING_BEFORE)
                            e_pad = min(td, e + PADDING_AFTER)

                            confv = max(itv["confs"]) if itv.get("confs") else 0.0
                            plc = Counter(itv.get("places", [])).most_common(1)[0][0] if itv.get("places") else "other"
                            detid = str(uuid.uuid4())

                            key = ffmpeg_trim_and_upload(temp, s_pad, e_pad, match_id, detid)
                            insert_detection_row(match_id, brand, s, e, plc, key, confv)
                            count += 1

                    # ==========================================================
                    # AUTO-GENERATE NEXT MATCH ID (DB-based)
                    # ==========================================================

                    st.success(f"{count} Brands were detected and video chunks uploaded to S3... 🎉")

                    # Generate brand aggregates after detection
                    try:
                        generate_brand_aggregates(match_id, match_start_dt, td)
                    except Exception as e:
                        st.error(f"Error generating brand aggregates: {e}")

                    # 1) Save the match we just completed — use this for table/chart view
                    st.session_state.last_completed_match_id = match_id

                    # 2) Auto-generate the next match id (DB-based)
                    next_id = generate_match_id()
                    st.session_state.current_match_id = next_id
                    st.caption(f"Next Match ID Ready 🔑: {next_id}")

                    try:
                        temp.unlink()
                    except Exception:
                        pass

    # =============== CHARTS (placeholder) ===============
    with tab_ch:
        st.header("📈 Brand Analytic")
        st.caption(f"📍Reports for: **{st.session_state.last_completed_match_id}**")

        mid = st.session_state.last_completed_match_id

        if not mid:
            st.warning("Upload a video to view charts.")
        else:

            # Save each chart as png
            fig1_path = os.path.join(tmp, "fig1_exposure.png")
            fig2_path = os.path.join(tmp, "fig2_ratio.png")
            fig_pie_path = os.path.join(tmp, "fig3_pie.png")
            fig4_path = os.path.join(tmp, "fig4_detect_count.png")
            fig5_path = os.path.join(tmp, "fig5_avg_duration.png")
            fig6_path = os.path.join(tmp, "fig6_confidence.png")
            fig7_path = os.path.join(tmp, "fig7_heatmap.png")



            pdf = generate_charts_pdf(mid)
            with open(pdf, "rb") as f:
                st.download_button(
                    label="⬇️ Download as PDF",
                    data=f,
                    file_name=f"Brand_Analytics_{mid}.pdf",
                    mime="application/pdf"
                )


            # Load aggregate + detection tables
            with engine.begin() as conn:
                df_agg = pd.read_sql(text("""
                    SELECT brand_name, total_duration_seconds, visibility_ratio, placement_distribution
                    FROM brand_aggregates
                    WHERE match_id = :m
                """), conn, params={"m": mid})

                df_det = pd.read_sql(text("""
                    SELECT brand_name, start_time_sec, end_time_sec, duration_sec, placement, confidence
                    FROM brand_detections
                    WHERE match_id = :m
                """), conn, params={"m": mid})

            # ----------------------------------------------------------
            # ROW 0 — METRIC CARDS (PLOTLY STYLE)
            # ----------------------------------------------------------
            st.subheader("📌 Match Summary Metrics")

            total_brands = df_agg["brand_name"].nunique()
            total_exposure = df_agg["total_duration_seconds"].sum()

            max_brand = df_agg.loc[df_agg["total_duration_seconds"].idxmax(), "brand_name"]
            max_brand_dur = df_agg["total_duration_seconds"].max()

            c1, c2, c3 = st.columns(3)
            c1.metric("🏷️ Brands Detected", total_brands)
            c2.metric("🔥 Top Brand", f"{max_brand} ({round(max_brand_dur,2)}s)")
            c3.metric("⏱️ Total Exposure (Sec)", round(total_exposure, 2))

            # ----------------------------------------------------------
            # ROW 1 — BRAND EXPOSURE + VISIBILITY RATIO
            # ----------------------------------------------------------
            st.subheader("📌 Brand Exposure Overview")

            col1, col2 = st.columns(2)

            with col1:
                fig1 = px.bar(
                    df_agg,
                    x="brand_name",
                    y="total_duration_seconds",
                    title="Total Visibility Duration (Seconds)",
                    color="brand_name",
                    text_auto=True
                )
                fig1.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig1, use_container_width=True)
                fig1.write_image(fig1_path)

            with col2:
                df_ratio = df_agg.copy()
                df_ratio["visibility_ratio"] *= 100

                fig2 = px.bar(
                    df_ratio,
                    x="brand_name",
                    y="visibility_ratio",
                    title="Visibility Ratio (%)",
                    color="brand_name",
                    text_auto=True
                )
                fig2.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig2, use_container_width=True)
                fig2.write_image(fig2_path)

            # ======================================================
            # 🥧 PIE CHART — Placement Visibility (Overall Summary)
            # ======================================================
            st.subheader("🥧 Overall Placement Visibility Summary")

            # Sum total duration per placement
            df_place_sum = df_det.groupby("placement")["duration_sec"].sum().reset_index()

            # Convert to percentage
            total_dur = df_place_sum["duration_sec"].sum()
            df_place_sum["percentage"] = (df_place_sum["duration_sec"] / total_dur) * 100

            fig3 = px.pie(
                df_place_sum,
                names="placement",
                values="percentage",
                title="Placement Visibility Contribution (%)",
                color="placement",
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.45  # donut style 
            )

            fig3.update_traces(textinfo="label+percent", pull=[0.03] * len(df_place_sum))

            fig3.update_layout(height=420)

            st.plotly_chart(fig3, use_container_width=True)
            fig3.write_image(fig_pie_path)


            # ----------------------------------------------------------
            # ROW 3 — DETECTION COUNT + AVG DURATION
            # ----------------------------------------------------------
            st.subheader("🔍 Detection Insights")

            col3, col4 = st.columns(2)

            with col3:
                det_count = df_det.groupby("brand_name").size().reset_index(name="count")
                fig4 = px.bar(
                    det_count,
                    x="brand_name",
                    y="count",
                    color="brand_name",
                    text_auto=True,
                    title="Number of Detections per Brand"
                )
                fig4.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig4, use_container_width=True)
                fig4.write_image(fig4_path)

            with col4:
                avg_dur = df_det.groupby("brand_name")["duration_sec"].mean().reset_index()
                fig5 = px.bar(
                    avg_dur,
                    x="brand_name",
                    y="duration_sec",
                    color="brand_name",
                    text_auto=True,
                    title="Average Clip Duration per Brand"
                )
                fig5.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig5, use_container_width=True)
                fig5.write_image(fig5_path)

            # ----------------------------------------------------------
            # ROW 4 — CONFIDENCE HISTOGRAM + HEATMAP
            # ----------------------------------------------------------
            st.subheader("🎛️ Model Performance & Placement Stats")

            col5, col6 = st.columns(2)

            with col5:
                fig6 = px.histogram(
                    df_det,
                    x="confidence",
                    nbins=20,
                    title="Confidence Score Distribution",
                    color="brand_name"
                )
                fig6.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig6, use_container_width=True)
                fig6.write_image(fig6_path)

            with col6:
                df_heat = df_det.groupby(["placement", "brand_name"])["duration_sec"].sum().reset_index()
                fig7 = px.density_heatmap(
                    df_heat,
                    x="brand_name",
                    y="placement",
                    z="duration_sec",
                    color_continuous_scale="Blues",
                    title="Placement vs Duration Heatmap"
                )
                fig7.update_layout(height=350)
                st.plotly_chart(fig7, use_container_width=True)
                fig7.write_image(fig7_path)    
            

        
    # =============== DETECTION TABLE ===============
    with tab_tb:
        # -----------------------------------------------
        # 📌 Brand Detection Table
        # -----------------------------------------------
        st.caption(f"Data's of last completed match 📍: **{st.session_state.last_completed_match_id}**")
        mid = st.session_state.last_completed_match_id

        # --------------------------------------
        # 🟩 MATCHES TABLE
        # --------------------------------------
        
        if mid:
            st.subheader("🟩 Match Details (Meta Information)")
        with engine.begin() as conn:
            df_match = pd.read_sql(text("""
                SELECT 
                    match_id,
                    home_team,
                    away_team,
                    match_type,
                    location,
                    start_time,
                    end_time,
                    winner,
                    raw_video_s3_key,
                    tracked_video_s3_key,
                    created_at
                FROM matches
                WHERE match_id = :m
                LIMIT 1
            """), conn, params={"m": mid})

        if df_match.empty:
            st.warning("No match details found for this Match ID.")
        else:
            st.dataframe(df_match, use_container_width=True)

            # --------------------------------------
            # 🟩 Dectection TABLE
            # --------------------------------------
            if mid:
                st.subheader("🟧 Detection Records")
                with engine.begin() as conn:
                    df = pd.read_sql(text("""
                        SELECT brand_name, start_time_sec, end_time_sec, duration_sec,
                            placement, chunk_s3key, confidence, created_at
                        FROM brand_detections
                        WHERE match_id = :m
                        ORDER BY created_at DESC
                    """), conn, params={"m": mid})

                if df.empty:
                    st.warning("No detections for this match yet.")
                else:
                    st.dataframe(df, use_container_width=True)

                # -----------------------------------------------
                # 📌 Brand Aggregates Table
                # -----------------------------------------------
                st.subheader("🟪 Brand Exposure Summary")

                with engine.begin() as conn:
                    df_agg = pd.read_sql(text("""
                        SELECT 
                            brand_name,
                            total_duration_seconds,
                            visibility_ratio,
                            placement_distribution,
                            first_seen,
                            last_seen,
                            created_at
                        FROM brand_aggregates
                        WHERE match_id = :m
                        ORDER BY brand_name ASC
                    """), conn, params={"m": mid})

                if df_agg.empty:
                    st.warning("Brand aggregates not yet generated for this match.")
                else:
                    st.dataframe(df_agg, use_container_width=True)

    # =============== DETECTION TAB ===============
    with tab_bot:
        st.caption(f"Curren Match ID is: **📍 {st.session_state.last_completed_match_id}**")
        st.header("📺 AI Chat Bot")
        
        
        st.caption("Suggestions questions:")
        st.caption("• Which brand has the highest exposure time?")
        st.caption("• Which brand has the highest visibility ratio?")
        st.caption("• Which brand appeared most frequently?")

        # Gemini model
        model = genai.GenerativeModel("gemini-2.5-flash")

        # SQL cleaner
        import re
        def clean_sql(raw):
            if not raw:
                return ""
            q = raw.strip()
            q = re.sub(r"```sql", "", q, flags=re.IGNORECASE)
            q = re.sub(r"```", "", q)
            q = re.sub(r"^sql\s+", "", q, flags=re.IGNORECASE)
            return q.strip()

        question = st.text_input("Ask anything about this match:")

        if st.button("Ask"):
            mid = st.session_state.last_completed_match_id
            if not mid:
                st.error("Please process a match first.")
                st.stop()

            # Step 1: Convert user question → SQL
            prompt = f"""
            Convert this into a SQL SELECT query.
            Use ONLY these two tables:

            brand_detections(
                match_id, brand_name, start_time_sec, end_time_sec,
                duration_sec, placement, confidence
            )

            brand_aggregates(
                match_id, brand_name, total_duration_seconds,
                visibility_ratio, placement_distribution
            )

            Always use: match_id = '{mid}'
            Return only SQL. No explanation.

            User question: {question}
            """

            sql_raw = model.generate_content(prompt).text
            sql = clean_sql(sql_raw)

            # Step 2: Run SQL silently
            try:
                df = pd.read_sql(text(sql), engine)
            except Exception as e:
                st.error(f"Unable to fetch data: {e}")
                st.stop()

            # Step 3: Ask Gemini to summarize only the final answer
            summary_prompt = f"""
            User question: {question}

            SQL result (table content):
            {df.to_string(index=False)}

            Provide a clear, short answer.
            Do NOT mention SQL, query, database, or table.
            """

            answer = model.generate_content(summary_prompt).text

            st.subheader("💬 Here’s your answer!")
            
            st.markdown(
                f"""
                <div style="
                    background-color:#1e1e1e;
                    padding:10px 14px;
                    border-radius:8px;
                    border-left:3px solid #4B7BE5;
                    margin-top:12px;
                    font-size:14px;
                    line-height:1.45;
                    color:#f0f0f0;
                ">
                    {answer}
                </div>
                """,
                unsafe_allow_html=True
            )

    # =============== ADMIN TAB ===============
    with tab_ad:
        st.header("Admin Tools")

        current_mid = st.session_state.last_completed_match_id
        st.caption(f"Match ID: **📍 {st.session_state.last_completed_match_id}**")

        if current_mid is None:
            st.warning("Nothing to delete yet. 🗑️")
        else:
            st.warning(f"⚠️ This will delete ALL the data's and videos for Match ID: {current_mid}")
            confirm = st.text_input("Type DELETE MATCH to confirm:")

            if st.button("🗑 Delete Entire Current Match"):
                if confirm.strip() == "DELETE MATCH":

                    # -----------------------------
                    # 1️⃣ DELETE ALL S3 FOLDER FILES
                    # -----------------------------
                    prefix = f"{current_mid}/"
                    try:
                        paginator = s3.get_paginator("list_objects_v2")
                        delete_list = []

                        for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
                            for obj in page.get("Contents", []):
                                delete_list.append({"Key": obj["Key"]})

                        # delete 1000 items at a time
                        for i in range(0, len(delete_list), 1000):
                            s3.delete_objects(
                                Bucket=BUCKET_NAME,
                                Delete={"Objects": delete_list[i: i + 1000]}
                            )

                        st.caption("🗂️ Cloud Storage Cleaned")

                    except Exception as e:
                        st.error(f"Error deleting from S3: {e}")

                    # -----------------------------
                    # 2️⃣ DELETE DB ROWS
                    # -----------------------------
                    try:
                        with engine.begin() as conn:
                            conn.execute(text("DELETE FROM matches WHERE match_id = :m"), {"m": current_mid})
                            conn.execute(text("DELETE FROM brand_detections WHERE match_id = :m"), {"m": current_mid})
                            conn.execute(text("DELETE FROM brand_aggregates WHERE match_id = :m"), {"m": current_mid})
                            

                        st.caption("🧹 Database Cleared")
                    except Exception as e:
                        st.error(f"DB delete error: {e}")

                    # -----------------------------
                    # 3️⃣ RESET MATCH ID (Option B)
                    # -----------------------------
                    new_id = generate_match_id()
                    st.session_state.current_match_id = new_id
                    st.session_state.last_completed_match_id = None

                    st.caption(f"⚡ Fresh Match ID loaded: {new_id} 👍🏻")

                else:
                    st.error("Type DELETE MATCH exactly to confirm.")
