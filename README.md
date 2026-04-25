# Road-Surface-Damage-Detection
# Road Damage Detection App

AI-powered road damage detection using YOLOv8 with cost estimation and reporting.

## Features

- Upload images or videos for road damage detection
- AI-powered damage classification (Potholes, Cracks, Manholes)
- Automatic cost estimation based on severity
- Generate PDF, CSV, and Excel reports
- Download annotated results

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py
```

## Deploy to Railway

### Step 1: Install Railway CLI (optional)
```bash
npm install -g @railway/cli
```

### Step 2: Create Railway Project
1. Go to [Railway Dashboard](https://railway.app/dashboard)
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Connect your GitHub account and select this repository

### Step 3: Deploy
Railway will automatically detect the `Procfile` and deploy your app.

The app will be available at: `https://your-project-name.up.railway.app`

### Environment Variables (Optional)
Set these in Railway Dashboard → Variables:
- `DEBUG`: Set to `False` for production

## File Structure

```
.
├── app.py                 # Main Flask application
├── best (3).pt           # YOLOv8 model weights
├── requirements.txt      # Python dependencies
├── Procfile             # Railway deployment config
├── railway.json         # Railway settings
├── templates/           # HTML templates
│   └── index.html
└── static/              # Static files
    ├── uploads/         # Uploaded images/videos
    ├── output_video/    # Processed videos
    └── reports/         # Generated reports
```

## Supported File Formats

- Images: JPG, JPEG, PNG
- Videos: MP4, AVI, MOV

## Notes

- The YOLO model file `best (3).pt` is included in the repo (~52MB)
- Uploaded files and generated reports are stored in `static/` folder
- Railway has ephemeral filesystem - files may not persist between restarts
