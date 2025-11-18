# Jio Hotstar AdVision & Analytics

AI-powered system for automated advertisement detection, brand visibility analytics, video chunking, PDF report generation, and RAG-based conversational insights — built for analysing cricket match broadcasts and producing sponsor-ready analytics.

## 🚀 Project Overview
Jio AdVision & Analytics detects brand advertisements in match videos, calculates visibility metrics, classifies placements, extracts brand chunks, stores analytics in an RDS, and exposes dashboards + a RAG-powered chatbot for natural language queries.

## 🎯 Key Features
- Brand Detection (YOLOv8 – Ultralytics)
- Placement Classification (boundary, jersey, overlay, scoreboard)
- Match Moment Tagging (six, wicket, batting, bowling, fielding)
- Timestamp & Duration Extraction
- Video Chunking (FFmpeg + S3 Upload)
- Automated PDF Report Generation
- Interactive Streamlit Dashboard
- RAG Chatbot using Google Generative AI

## 📁 Folder Structure
```

Jio_AdVision_Analytics/
├── app/ # Streamlit dashboard UI + backend processing functions
├── docs/ # source video links
├── model/ # YOLO models
├── notebooks/ # Jupyter notebooks for experimentation & testing
├── testing_video/ # Small test videos used for demo/testing purpose
├── requirements.txt # Python dependency list
├── README.md 
└── .gitignore 

```

## 📸 Dashboard Screenshot
(Add your dashboard screenshot here)
```
![Dashboard Screenshot](docs/images/dashboard.png)
```

## 🛠️ Tech Stack
- YOLOv8 (Ultralytics)
- OpenCV
- Streamlit
- Plotly
- SQLAlchemy
- PostgreSQL (RDS)
- boto3 (AWS S3)
- FFmpeg
- ReportLab
- Google Generative AI (RAG)

## ⚙️ Setup Instructions
### 1. Clone Repo
```
git clone https://github.com/Infant-Joshva/Jio_AdVision_Analytics.git
cd Jio_AdVision_Analytics
```

### 2. Install Dependencies
```
pip install -r requirements.txt
```

### 3. Add Secrets
Create `app/.streamlit/secrets.toml` with:
```
aws_access_key="YOUR_AWS_KEY"
aws_secret_key="YOUR_AWS_SECRET"
bucket_name="YOUR_BUCKET_NAME"
genai_api_key="YOUR_GENAI_KEY"
database_url="postgresql://user:pass@host:port/db"
```

### 4. Run Dashboard
```
streamlit run app/main.py
```

## 🧪 Testing
Place your test video inside:
```
testing_video/Short_test_video.mp4
```

Run detection:
```
python scripts/detect_brands.py --video testing_video/Short_test_video.mp4 --match-id JIO-MATCH-TEST
```

## 📦 AWS S3 Structure
```

s3://jioadvision-uploads/
  └── MatchID/
      └── chunks/
          └── chunk.mp4
      └── raw/
          └── raw.mp4
      └── track/
          └── track.mp4

```

## 📄 PDF Report Output
- Visibility duration
- Visibility ratio
- Placement distribution
- Event-based visibility
- S3 chunk links

## 🧩 API Endpoints
```
POST /api/upload
GET  /api/status/<id>
GET  /api/report/<match>
GET  /api/aggregate
```

## 📈 Evaluation Metrics
- Detection precision / recall / F1
- Timestamp accuracy
- Video chunk quality
- Dashboard-RDS sync accuracy
- RAG answer correctness

## ✅ Deliverables
- Full pipeline code
- YOLO model + weights
- Test video
- Extracted clips
- Streamlit dashboard
- PDF reports
- RAG chatbot
- Documentation
