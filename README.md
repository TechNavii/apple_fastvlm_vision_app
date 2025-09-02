# 🍎 FastVLM GUI

A real-time vision-language model GUI using Apple's official FastVLM implementation. Features webcam support with automatic analysis at customizable intervals.

## Features

✅ **Real-time webcam analysis** with 20 FPS smooth video feed  
✅ **Auto-analysis mode** - Automatically analyze frames at customizable intervals (5-60 seconds)  
✅ **Multiple model sizes** - 0.5B, 1.5B, and 7B models  
✅ **Actual camera names** - Shows real device names on macOS  
✅ **Image upload support** - Analyze static images  
✅ **Runs locally** on Apple Silicon (MPS)  

## Prerequisites

1. **macOS** with Apple Silicon (M1/M2/M3) or Intel Mac
2. **Python 3.8+**
3. **Webcam** (built-in or external)

## Installation

### 1. Clone Apple's FastVLM Repository

```bash
cd ~/Desktop/coding_test
git clone https://github.com/apple/ml-fastvlm.git
cd ml-fastvlm
pip install -e .
```

### 2. Download FastVLM Models

```bash
# Download all models (or choose specific ones)
./get_models.sh

# The models will be in ml-fastvlm/checkpoints/
# - llava-fastvithd_0.5b_stage3.zip (1.2GB)
# - llava-fastvithd_1.5b_stage3.zip (3.0GB)  
# - llava-fastvithd_7b_stage3.zip (11GB)
```

### 3. Install GUI Dependencies

```bash
cd ~/Desktop/coding_test
git clone https://github.com/TechNavii/apple_fastvlm_vision_app.git
cd fastvlm
pip install -r requirements.txt
```

## Usage

### Start the GUI

```bash
python official_gui.py
```

Open your browser at: **http://127.0.0.1:7860**

### How to Use

1. **Load Model**
   - Select model size (0.5B for speed, 7B for accuracy)
   - Click "📦 Load Model"
   - Wait for success message

2. **Setup Webcam**
   - Click "🔍 Search Cameras" to find cameras
   - Select your camera from dropdown (shows actual device names)
   - Click "▶️ Start Camera"

3. **Enable Auto-Analysis** (Optional)
   - Check "Enable Auto-Analysis"
   - Set interval (5-60 seconds)
   - Enter your prompt
   - Model will automatically analyze at set intervals

4. **Manual Analysis**
   - Click "🔍 Analyze Webcam Frame" anytime
   - Or upload an image in the "Image Upload" tab

## Model Information

| Model | Size | Speed | Use Case |
|-------|------|-------|----------|
| FastVLM-0.5B | 1.2GB | Fastest | Real-time applications, mobile |
| FastVLM-1.5B | 3.0GB | Balanced | General use |
| FastVLM-7B | 11GB | Slower | Best accuracy |

## Troubleshooting

**Camera not found?**
- Grant camera permissions in System Settings → Privacy & Security → Camera

**Model won't load?**
- Ensure models are downloaded and unzipped in `ml-fastvlm/checkpoints/`
- Check you have enough RAM (7B model needs ~16GB)

**Low FPS?**
- Use smaller model (0.5B)
- Increase auto-analysis interval
- Close other applications

## Acknowledgments

- Apple ML Research for FastVLM models
- Based on official Apple FastVLM implementation
