from flask import Flask, render_template, request, send_file, jsonify
import torch
from diffusers import QwenImageLayeredPipeline
from PIL import Image
from psd_tools import PSDImage
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

pipeline = None
def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = QwenImageLayeredPipeline.from_pretrained("Qwen/Qwen-Image-Layered", torch_dtype=torch.bfloat16)
        # pipeline = pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipeline

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            result = process_image(filepath)
            return jsonify(result)
    return render_template('index.html')

def process_image(image_path):
    image = Image.open(image_path).convert("RGBA")
    pipe = get_pipeline()
    with torch.inference_mode():
        output = pipe(...)  # your params
        layers = output.images[0]
    return {"status": "success", "psd_url": "/download/psd"}