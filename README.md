# Face Meme Detector

Real-time facial expression detection that reacts to your face with meme images. Make a face — get a meme.

## Features

- Live webcam face tracking using MediaPipe Face Mesh
- Detects three expressions in real time: happy, neutral, and angry
- Displays a matching meme image in a separate window when an expression is detected
- Expression detection is based on facial landmark geometry (no ML model downloads required)
- Meme images are fully customizable — just swap the files in the `memes/` folder

## How It Works

MediaPipe maps 468 facial landmarks onto your face each frame. The app measures:
- **Smile score** — position of mouth corners relative to the upper lip
- **Brow score** — distance between inner eyebrows and eyes

These scores are compared against thresholds to classify your expression as happy, angry, or neutral.

## Tech Stack

- Python 3.11
- OpenCV
- MediaPipe

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Adding Memes

Place images in the `memes/` folder named after the expression:

| File | Triggered by |
|------|-------------|
| `happy.jpg` | Smiling |
| `angry.jpg` | Furrowed brows |
| `neutral.jpg` | Relaxed face |

Supported formats: `jpg`, `jpeg`, `png`

## Run

```bash
python main.py
```

Press `Q` to quit.
