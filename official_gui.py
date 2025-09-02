#!/usr/bin/env python3
"""
GUI for Official Apple FastVLM with webcam support
"""
import sys
import os
import cv2
import torch
import gradio as gr
import numpy as np
from PIL import Image
import time
import threading
from queue import Queue

# Add the ml-fastvlm directory to path
sys.path.insert(0, '/Users/tooru/Desktop/coding_test/ml-fastvlm')

from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

class FastVLMApp:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.context_len = None
        self.model_loaded = False
        self.camera = None
        self.camera_active = False
        self.current_frame = None
        self.last_inference_time = 0
        self.last_response = ""
        self.processing = False
        self.auto_analyze = False
        self.auto_analyze_interval = 10  # seconds
        self.last_auto_analyze_time = 0
        self.analyze_thread = None
        self.auto_prompt = "Describe what you see in detail."  # Default prompt for auto-analysis
        
        # Available models - check which ones exist
        self.model_paths = {}
        base_path = "/Users/tooru/Desktop/coding_test/ml-fastvlm/checkpoints"
        
        # Check for each model
        for size, name in [("0.5b", "FastVLM-0.5B"), ("1.5b", "FastVLM-1.5B"), ("7b", "FastVLM-7B")]:
            stage3_path = f"{base_path}/llava-fastvithd_{size}_stage3"
            stage3_zip = f"{base_path}/llava-fastvithd_{size}_stage3.zip"
            
            if os.path.exists(stage3_path) or os.path.exists(stage3_zip):
                self.model_paths[name] = stage3_path
        
    def load_model(self, model_name):
        """Load the selected FastVLM model"""
        if model_name not in self.model_paths:
            return f"❌ Model {model_name} not found"
        
        model_path = self.model_paths[model_name]
        
        # Check if model exists
        if not os.path.exists(model_path):
            # Try to unzip it
            zip_path = model_path.replace("stage3", "stage3.zip")
            if os.path.exists(zip_path):
                import subprocess
                print(f"Unzipping {model_name}...")
                subprocess.run(["unzip", "-q", zip_path, "-d", os.path.dirname(model_path)], 
                             cwd=os.path.dirname(model_path))
            else:
                return f"❌ Model files not found for {model_name}"
        
        try:
            print(f"Loading {model_name}...")
            
            # Remove generation config if exists
            generation_config = None
            if os.path.exists(os.path.join(model_path, 'generation_config.json')):
                generation_config = os.path.join(model_path, '.generation_config.json')
                if os.path.exists(generation_config):
                    os.remove(generation_config)
                os.rename(os.path.join(model_path, 'generation_config.json'), generation_config)
            
            # Load model
            disable_torch_init()
            model_name_internal = get_model_name_from_path(model_path)
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path, None, model_name_internal, device="mps"
            )
            
            # Set pad token
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
            
            # Restore generation config
            if generation_config and os.path.exists(generation_config):
                os.rename(generation_config, os.path.join(model_path, 'generation_config.json'))
            
            self.model_loaded = True
            return f"✅ {model_name} loaded successfully!"
            
        except Exception as e:
            return f"❌ Failed to load model: {str(e)}"
    
    def get_camera_names(self):
        """Get actual camera device names on macOS"""
        import subprocess
        import json
        
        camera_names = {}
        try:
            # Use system_profiler to get camera info on macOS
            result = subprocess.run(
                ["system_profiler", "SPCameraDataType", "-json"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                if "SPCameraDataType" in data:
                    cameras = data["SPCameraDataType"]
                    for idx, camera in enumerate(cameras):
                        name = camera.get("_name", f"Camera {idx}")
                        camera_names[idx] = name
        except:
            pass
        
        return camera_names
    
    def search_cameras(self):
        """Search for available cameras with their actual names"""
        available_cameras = []
        camera_map = {}
        
        # Get actual camera names
        camera_names = self.get_camera_names()
        
        # Check which indices work with OpenCV
        for i in range(5):  # Check first 5 camera indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Get the camera name or use generic name
                if i in camera_names:
                    name = f"{camera_names[i]} (Camera {i})"
                else:
                    name = f"Camera {i}"
                
                available_cameras.append(name)
                camera_map[name] = i
                cap.release()
        
        # Store the mapping for later use
        self.camera_map = camera_map
        
        if available_cameras:
            return f"✅ Found {len(available_cameras)} camera(s)", gr.update(choices=available_cameras, value=available_cameras[0])
        else:
            return "❌ No cameras found", gr.update(choices=[], value=None)
    
    def start_camera(self, camera_id):
        """Start the selected camera"""
        if not camera_id:
            return "❌ No camera selected"
        
        try:
            # Get camera index from the mapping
            if hasattr(self, 'camera_map') and camera_id in self.camera_map:
                cam_idx = self.camera_map[camera_id]
            else:
                # Fallback: try to extract from string
                cam_idx = int(camera_id.split()[-1].rstrip(')'))
            
            if self.camera is not None:
                self.camera.release()
            
            self.camera = cv2.VideoCapture(cam_idx)
            if self.camera.isOpened():
                self.camera_active = True
                return f"✅ {camera_id} started"
            else:
                return f"❌ Failed to open {camera_id}"
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def stop_camera(self):
        """Stop the camera"""
        self.camera_active = False
        self.auto_analyze = False  # Stop auto analysis
        if self.camera:
            self.camera.release()
            self.camera = None
        return "✅ Camera stopped"
    
    def toggle_auto_analysis(self, enabled, interval, prompt=None):
        """Toggle automatic analysis"""
        self.auto_analyze = enabled
        self.auto_analyze_interval = interval
        if prompt:
            self.auto_prompt = prompt
        
        if enabled:
            # Start the auto-analysis thread if not running
            if self.analyze_thread is None or not self.analyze_thread.is_alive():
                self.analyze_thread = threading.Thread(target=self._auto_analyze_worker)
                self.analyze_thread.daemon = True
                self.analyze_thread.start()
            return f"✅ Auto-analysis enabled (every {interval}s)"
        else:
            return "⏸️ Auto-analysis disabled"
    
    def _auto_analyze_worker(self):
        """Background worker for automatic analysis"""
        while self.auto_analyze and self.camera_active:
            current_time = time.time()
            if current_time - self.last_auto_analyze_time >= self.auto_analyze_interval:
                if self.current_frame is not None and not self.processing:
                    # Trigger analysis with the stored prompt
                    frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                    response = self.process_image(
                        frame_rgb, 
                        self.auto_prompt, 
                        temperature=0.7, 
                        max_tokens=150
                    )
                    self.last_auto_analyze_time = current_time
            time.sleep(0.5)  # Check every 0.5 seconds
    
    def get_frame(self):
        """Get current frame from camera"""
        if self.camera and self.camera_active:
            ret, frame = self.camera.read()
            if ret:
                self.current_frame = frame
                # Convert BGR to RGB for display
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    
    def process_image(self, image, prompt, temperature=0.7, max_tokens=150):
        """Process an image with the model"""
        if not self.model_loaded:
            return "❌ Please load a model first"
        
        if self.processing:
            return "⏳ Processing in progress..."
        
        self.processing = True
        start_time = time.time()
        
        try:
            # Handle different image inputs
            if image is None:
                return "❌ No image provided"
            
            # Convert to PIL Image if needed
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            # Construct prompt
            qs = prompt
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            
            conv = conv_templates["qwen_2"].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            full_prompt = conv.get_prompt()
            
            # Tokenize
            input_ids = tokenizer_image_token(
                full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
            ).unsqueeze(0).to(torch.device("mps"))
            
            # Process image
            image_tensor = process_images([pil_image], self.image_processor, self.model.config)[0]
            
            # Generate
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half(),
                    image_sizes=[pil_image.size],
                    do_sample=temperature > 0,
                    temperature=temperature,
                    max_new_tokens=max_tokens,
                    use_cache=True
                )
                
                response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            
            inference_time = time.time() - start_time
            self.last_inference_time = inference_time
            self.last_response = response
            
            return f"{response}\n\n⏱️ Inference time: {inference_time:.2f}s"
            
        except Exception as e:
            return f"❌ Error: {str(e)}"
        finally:
            self.processing = False
    
    def analyze_webcam(self, prompt, temperature, max_tokens):
        """Analyze current webcam frame"""
        if self.current_frame is None:
            return "❌ No webcam frame available"
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        return self.process_image(frame_rgb, prompt, temperature, max_tokens)

# Create app instance
app = FastVLMApp()

# Create Gradio interface
with gr.Blocks(title="FastVLM GUI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🍎 Apple FastVLM - Vision Language Model")
    gr.Markdown("Official Apple FastVLM implementation with webcam support")
    
    with gr.Tab("📹 Webcam"):
        with gr.Row():
            with gr.Column(scale=1):
                # Model selection
                model_dropdown = gr.Dropdown(
                    choices=list(app.model_paths.keys()),
                    value=list(app.model_paths.keys())[0] if app.model_paths else None,
                    label="Select Model"
                )
                load_model_btn = gr.Button("📦 Load Model", variant="primary")
                model_status = gr.Textbox(label="Model Status", value="No model loaded")
                
                gr.Markdown("### Camera Controls")
                search_btn = gr.Button("🔍 Search Cameras")
                camera_dropdown = gr.Dropdown(label="Select Camera", choices=[])
                start_camera_btn = gr.Button("▶️ Start Camera", variant="primary")
                stop_camera_btn = gr.Button("⏹️ Stop Camera")
                camera_status = gr.Textbox(label="Camera Status", value="Camera not started")
                
                gr.Markdown("### Auto-Analysis Settings")
                auto_analyze_checkbox = gr.Checkbox(
                    label="Enable Auto-Analysis",
                    value=False
                )
                auto_interval_slider = gr.Slider(
                    minimum=5,
                    maximum=60,
                    value=10,
                    step=5,
                    label="Analysis Interval (seconds)"
                )
                auto_status = gr.Textbox(label="Auto-Analysis Status", value="Disabled")
                
            with gr.Column(scale=2):
                webcam_image = gr.Image(label="Webcam Feed", height=480)
                
                prompt_input = gr.Textbox(
                    label="Prompt",
                    value="Describe what you see in detail.",
                    lines=2
                )
                
                with gr.Row():
                    temperature_slider = gr.Slider(
                        minimum=0, maximum=1, value=0.7, step=0.1,
                        label="Temperature"
                    )
                    max_tokens_slider = gr.Slider(
                        minimum=50, maximum=500, value=150, step=50,
                        label="Max Tokens"
                    )
                
                analyze_btn = gr.Button("🔍 Analyze Webcam Frame", variant="primary")
                
                response_output = gr.Textbox(
                    label="Model Response",
                    lines=10,
                    value="Response will appear here..."
                )
    
    with gr.Tab("🖼️ Image Upload"):
        with gr.Row():
            with gr.Column():
                upload_image = gr.Image(label="Upload Image", type="pil")
                upload_prompt = gr.Textbox(
                    label="Prompt",
                    value="What is in this image?",
                    lines=2
                )
                
                with gr.Row():
                    upload_temperature = gr.Slider(
                        minimum=0, maximum=1, value=0.7, step=0.1,
                        label="Temperature"
                    )
                    upload_max_tokens = gr.Slider(
                        minimum=50, maximum=500, value=150, step=50,
                        label="Max Tokens"
                    )
                
                upload_analyze_btn = gr.Button("🔍 Analyze Image", variant="primary")
            
            with gr.Column():
                upload_response = gr.Textbox(
                    label="Model Response",
                    lines=15,
                    value="Upload an image and click Analyze..."
                )
    
    with gr.Tab("ℹ️ About"):
        gr.Markdown("""
        ## Apple FastVLM
        
        FastVLM is a family of vision-language models designed by Apple for efficient on-device inference.
        
        ### Available Models:
        - **FastVLM-0.5B**: Smallest and fastest, ideal for mobile devices
        - **FastVLM-1.5B**: Balanced performance and accuracy
        
        ### Features:
        - Real-time webcam analysis
        - Image upload and analysis
        - Adjustable generation parameters
        - Runs locally on Apple Silicon (MPS)
        
        ### Usage:
        1. Select and load a model
        2. For webcam: Search cameras → Select camera → Start camera → Analyze
        3. For images: Upload image → Enter prompt → Analyze
        """)
    
    # Event handlers
    def load_model_handler(model_name):
        status = app.load_model(model_name)
        return status
    
    def search_cameras_handler():
        status, dropdown = app.search_cameras()
        return status, dropdown
    
    def start_camera_handler(camera_id):
        status = app.start_camera(camera_id)
        # Auto-start analysis if checkbox is enabled
        if app.auto_analyze:
            app.toggle_auto_analysis(True, app.auto_analyze_interval)
        return status
    
    def stop_camera_handler():
        status = app.stop_camera()
        return status
    
    def toggle_auto_analysis_handler(enabled, interval, prompt):
        if not app.camera_active and enabled:
            return "⚠️ Start camera first"
        status = app.toggle_auto_analysis(enabled, interval, prompt)
        return status
    
    def update_webcam():
        frame = app.get_frame()
        return frame
    
    def update_response():
        # Return the last response from auto-analysis or manual analysis
        return app.last_response if app.last_response else "Waiting for analysis..."
    
    def analyze_webcam_handler(prompt, temp, tokens):
        response = app.analyze_webcam(prompt, temp, tokens)
        return response
    
    def analyze_upload_handler(image, prompt, temp, tokens):
        response = app.process_image(image, prompt, temp, tokens)
        return response
    
    # Connect events
    load_model_btn.click(
        load_model_handler,
        inputs=[model_dropdown],
        outputs=[model_status]
    )
    
    search_btn.click(
        search_cameras_handler,
        outputs=[camera_status, camera_dropdown]
    )
    
    start_camera_btn.click(
        start_camera_handler,
        inputs=[camera_dropdown],
        outputs=[camera_status]
    )
    
    stop_camera_btn.click(
        stop_camera_handler,
        outputs=[camera_status]
    )
    
    # Auto-analysis controls
    auto_analyze_checkbox.change(
        toggle_auto_analysis_handler,
        inputs=[auto_analyze_checkbox, auto_interval_slider, prompt_input],
        outputs=[auto_status]
    )
    
    auto_interval_slider.change(
        lambda interval, prompt: app.toggle_auto_analysis(app.auto_analyze, interval, prompt) if app.auto_analyze else "Disabled",
        inputs=[auto_interval_slider, prompt_input],
        outputs=[auto_status]
    )
    
    analyze_btn.click(
        analyze_webcam_handler,
        inputs=[prompt_input, temperature_slider, max_tokens_slider],
        outputs=[response_output]
    )
    
    upload_analyze_btn.click(
        analyze_upload_handler,
        inputs=[upload_image, upload_prompt, upload_temperature, upload_max_tokens],
        outputs=[upload_response]
    )
    
    # Timer for webcam feed update (high frame rate)
    webcam_timer = gr.Timer(0.05, active=True)  # 20 FPS for smooth video
    webcam_timer.tick(update_webcam, outputs=[webcam_image])
    
    # Timer for response update (for auto-analysis results)
    response_timer = gr.Timer(0.5, active=True)
    response_timer.tick(update_response, outputs=[response_output])

# Launch the app
if __name__ == "__main__":
    print("Starting FastVLM GUI...")
    print("Opening browser at http://127.0.0.1:7860")
    demo.queue()
    demo.launch(share=False)