from flask import Flask, render_template, request, jsonify
from PIL import Image
import os
import torch
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration

app = Flask(__name__)

# In the container this points at /opt/model, the checkpoint baked in at build
# time -- HF_HUB_OFFLINE=1 there, so nothing is downloadable at runtime.
MODEL_ID = os.environ.get("MODEL_ID", "Salesforce/blip-image-captioning-large")

processor = BlipProcessor.from_pretrained(MODEL_ID)
# .float() upcasts the float16 on-disk checkpoint: half the image size, but
# float32 math, since CPUs have no usable float16 kernels. No-op on an fp32 one.
model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).float().eval()

max_new_tokens = 100

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process-image', methods=['POST'])
def process_image():
    try:
        image_file = request.files['image']
        image_bytes = image_file.read()
        raw_image = Image.open(BytesIO(image_bytes)).convert('RGB')

        # unconditional image captioning
        inputs = processor(raw_image, return_tensors="pt")
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens)
        caption = processor.decode(out[0], skip_special_tokens=True)

        return jsonify({'caption': caption})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
