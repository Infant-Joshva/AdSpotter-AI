# streamlit_app_full.py
import streamlit as st
from sqlalchemy import create_engine, text
import pandas as pd
from datetime import datetime
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

# ==========================================================
# CONFIG - update these values for your environment
# ==========================================================
DB_URL = "postgresql+psycopg2://postgres:Admin@localhost:5432/jio_advision"
BUCKET_NAME = "jioadvision-uploads"
AWS_REGION = "ap-south-1"
MODEL_PATH = Path(r"C:\Users\infan\OneDrive\Desktop\Final Project- Jio_AdVision_Analytics\Jio_AdVision_Analytics\model\best1.pt")
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

st.set_page_config(layout="wide")
st.sidebar.title("📌 Navigation")
menu = st.sidebar.radio("Go to:", ["📄 About Project", "🧭 Dashboard (Track / Charts / DB / Admin)"])

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
# Utilities
# ==========================================================
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
    cx = x1 + w/2
    cy = y1 + h/2
    return cx/frame_w, cy/frame_h, rel_area

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
        intervals=[]
        cur_s=None
        cur_e=None
        confs=[]
        places=[]
        for t,c,p in items:
            if cur_s is None:
                cur_s=cur_e=t
                confs=[c]
                places=[p]
            else:
                if t-cur_e<=gap:
                    cur_e=t
                    confs.append(c)
                    places.append(p)
                else:
                    intervals.append({"start":cur_s,"end":cur_e,"confs":confs[:],"places":places[:]})
                    cur_s=cur_e=t
                    confs=[c]
                    places=[p]
        intervals.append({"start":cur_s,"end":cur_e,"confs":confs[:],"places":places[:]})
        out[b]=intervals
    return out

def ffmpeg_trim_and_upload(video, start, end, match_id, detid):
    out = Path(f"tmp_{detid}.mp4")
    dur = max(0.01, end-start)
    cmd=[FFMPEG_BIN,"-y","-ss",f"{start:.3f}","-i",str(video),"-t",f"{dur:.3f}","-c:v","libx264","-preset","fast","-c:a","aac",str(out)]
    try:
        subprocess.run(cmd,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    except:
        if out.exists(): out.unlink()
        return None
    key=f"{match_id}/chunks/{detid}.mp4"
    try:
        s3.upload_file(str(out),BUCKET_NAME,key)
    except:
        out.unlink()
        return None
    out.unlink()
    return key

def insert_detection_row(match_id, brand, start, end, placement, key, conf):
    did=str(uuid.uuid4())
    dur=end-start
    sql=text("""
    INSERT INTO brand_detections 
    (id,match_id,brand_name,start_time_sec,end_time_sec,duration_sec,placement,chunk_s3key,confidence,created_at,updated_at)
    VALUES 
    (:id,:m,:b,:s,:e,:d,:p,:k,:c,:cr,:u)
    """)
    with engine.begin() as conn:
        conn.execute(sql,{
            "id":did,"m":match_id,"b":brand,
            "s":float(start),"e":float(end),"d":float(dur),
            "p":placement,"k":key,"c":float(conf),
            "cr":datetime.now(),"u":datetime.now()
        })
    return did

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
if menu == "📄 About Project":
    st.title("📄 About — Jio Hotstar AdVision & Analytics")
    st.markdown("Automated brand detection & analytics.")

# ==========================================================
# PAGE 2 — DASHBOARD
# ==========================================================
elif menu == "🧭 Dashboard (Track / Charts / DB / Admin)":
    st.title("🧭 Dashboard")
    tab_up, tab_ch, tab_tb, tab_ad = st.tabs(["Upload & Track", "Charts (Coming Soon)", "Detection Table", "Admin Tools"])

    # =============== Upload & Track Tab ===============
    with tab_up:
        st.header("Upload & Track")
        st.caption(f"Current Match ID: **{st.session_state.current_match_id}**")

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
                    mid=str(uuid.uuid4())

                    # Save raw locally
                    temp=Path(f"temp_{match_id}.mp4")
                    temp.write_bytes(raw.read())

                    fps,fc,dur,W,H = get_video_props(temp)

                    # TRACK MODE
                    timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
                    folder=f"{match_id}_{timestamp}"
                    outdir=Path(f"runs/track/{folder}")
                    outdir.mkdir(parents=True,exist_ok=True)

                    model.track(
                        source=str(temp),
                        show=False,
                        save=True,
                        imgsz=480,
                        vid_stride=20,
                        project="runs/track",
                        name=folder,
                        exist_ok=True
                    )

                    # find tracked file
                    time.sleep(1)
                    files=list(outdir.rglob("*.mp4"))+list(outdir.rglob("*.avi"))+list(outdir.rglob("*.mov"))
                    trk=files[0] if files else None

                    conv=None
                    if trk:
                        conv=outdir/f"{match_id}_tracked.mp4"
                        subprocess.run([FFMPEG_BIN,"-y","-i",str(trk),"-vcodec","libx264","-acodec","aac",str(conv)])

                    raw_key=f"{match_id}/raw/{match_id}.mp4"
                    trk_key=f"{match_id}/track/{match_id}_tracked.mp4" if conv else None

                    s3.upload_file(str(temp),BUCKET_NAME,raw_key)
                    if conv:
                        s3.upload_file(str(conv),BUCKET_NAME,trk_key)

                    trk_url=None
                    if trk_key:
                        trk_url=s3.generate_presigned_url("get_object",Params={"Bucket":BUCKET_NAME,"Key":trk_key},ExpiresIn=3600)

                    # Insert matches row
                    with engine.begin() as conn:
                        conn.execute(text("""
                            INSERT INTO matches 
                            (id,match_id,home_team,away_team,match_type,location,
                             start_time,end_time,winner,raw_video_s3_key,tracked_video_s3_key,
                             created_at,updated_at)
                            VALUES 
                            (:i,:m,:h,:a,:t,:l,:st,:et,:w,:rk,:tk,:c,:u)
                        """),{
                            "i":mid,"m":match_id,"h":home,"a":away,"t":mtype,
                            "l":loc,
                            "st":datetime.combine(datetime.today(),stt),
                            "et":datetime.combine(datetime.today(),ett),
                            "w":win,
                            "rk":raw_key,"tk":trk_key,
                            "c":datetime.now(),"u":datetime.now()
                        })

                    st.caption("Tracking Completed ✅")
                    if trk_url:
                        st.video(trk_url)

                    # DETECTION STREAM MODE
                    # st.caption("✂️ Chunk extraction completed…")
                    dets=[]
                    idx=0
                    fps=fps if fps>0 else MAX_SAMPLE_FPS

                    for res in model(source=str(temp),stream=True):
                        t=idx/fps
                        boxes=getattr(res,"boxes",None)
                        img=getattr(res,"orig_img",None)
                        h=img.shape[0] if img is not None else H
                        w=img.shape[1] if img is not None else W

                        if boxes:
                            for b in boxes:
                                try:
                                    cid=int(b.cls[0])
                                    brand=model.names.get(cid,str(cid))
                                except: brand="unknown"
                                try: conf=float(b.conf[0])
                                except: conf=0.0

                                cx,cy,ar = safe_extract_coords(b,w,h)
                                place=placement_heuristic(cx,cy,ar)

                                dets.append({"brand":brand,"t":t,"conf":conf,"placement":place})
                        idx+=1

                    merged=merge_detections(dets)
                    count=0
                    _,_,td,_,_=get_video_props(temp)

                    for brand,ints in merged.items():
                        for itv in ints:
                            s=float(itv["start"])
                            e=float(itv["end"])
                            s_pad=max(0.0,s-PADDING_BEFORE)
                            e_pad=min(td,e+PADDING_AFTER)

                            confv=max(itv["confs"])
                            plc = Counter(itv["places"]).most_common(1)[0][0]
                            detid=str(uuid.uuid4())

                            key=ffmpeg_trim_and_upload(temp,s_pad,e_pad,match_id,detid)
                            insert_detection_row(match_id,brand,s,e,plc,key,confv)
                            count+=1
                    # ==========================================================
                    # AUTO-GENERATE NEXT MATCH ID (DB-based)
                    # ==========================================================

                    st.success(f"{count} Brands were detected and video chunks uploaded to S3... 🎉")

                    # 1) Save the match we just completed — use this for table/chart view
                    st.session_state.last_completed_match_id = match_id

                    # 2) Auto-generate the next match id (DB-based)
                    next_id = generate_match_id()
                    st.session_state.current_match_id = next_id
                    st.caption(f"Next Match ID Ready 🔑: {next_id}")

                    try: temp.unlink()
                    except: pass

    # =============== CHARTS (placeholder) ===============
    with tab_ch:
        st.header("Charts (Coming Soon)")
        st.info("Charts will be generated only for the current match_id.")

    # =============== DETECTION TABLE ===============
    with tab_tb:
        st.header("Detection Table (Current Match)")
        mid = st.session_state.last_completed_match_id

        if mid is None:
            st.info("No completed match yet — upload a match to see detections here.")
        else:
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
                st.info(f"Showing detections for completed Match ID: **{mid}**")



    # =============== ADMIN TAB ===============
    with tab_ad:
        st.header("Admin Tools")

        current_mid = st.session_state.last_completed_match_id

        if current_mid is None:
            st.info("No completed match to delete yet.")
        else:
            st.warning(f"⚠️ This will delete ALL data for Match ID: {current_mid}")
            st.warning("Rows from MATCHES table, BRAND_DETECTIONS table and ALL S3 videos will be deleted!")

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

                        st.success("S3 folder deleted successfully.")

                    except Exception as e:
                        st.error(f"Error deleting from S3: {e}")

                    # -----------------------------
                    # 2️⃣ DELETE DB ROWS
                    # -----------------------------
                    try:
                        with engine.begin() as conn:
                            conn.execute(text("DELETE FROM brand_detections WHERE match_id = :m"), {"m": current_mid})
                            conn.execute(text("DELETE FROM matches WHERE match_id = :m"), {"m": current_mid})

                        st.success("Database rows deleted successfully.")
                    except Exception as e:
                        st.error(f"DB delete error: {e}")

                    # -----------------------------
                    # 3️⃣ RESET MATCH ID (Option B)
                    # -----------------------------
                    new_id = generate_match_id()
                    st.session_state.current_match_id = new_id
                    st.session_state.last_completed_match_id = None

                    st.success(f"Reset completed! Next Match ID Ready: {new_id}")

                else:
                    st.error("Type DELETE MATCH exactly to confirm.")

