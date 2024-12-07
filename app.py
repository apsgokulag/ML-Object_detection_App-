from flask import Flask, render_template, request, jsonify
from PIL import Image
import os
import io
import sys
import numpy as np 
import cv2
import base64
from yolo_detection_images import runModel

app = Flask(__name__)

@app.route('/detectObject', methods=['POST'])
def mask_image():
    try:
        # Check if image file exists in the request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file uploaded'}), 400
        
        # Read image file
        file = request.files['image'].read()  # byte file
        
        # Convert to numpy array and decode 
        npimg = np.frombuffer(file, np.uint8)
        
        # Decode image 
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        # Check if image was successfully decoded
        if img is None:
            return jsonify({'error': 'Invalid image file'}), 400

        # Run object detection model
        try:
            detected_img, detected_objects = runModel(img)
        except Exception as e:
            print(f"Object detection error: {e}", file=sys.stderr)
            return jsonify({'error': 'Object detection failed'}), 500

        # Convert detected image to PIL Image
        img = Image.fromarray(detected_img.astype("uint8"))
        
        # Save to bytes buffer
        rawBytes = io.BytesIO()
        img.save(rawBytes, "JPEG")
        rawBytes.seek(0)
        
        # Encode to base64 and decode to string
        img_base64 = base64.b64encode(rawBytes.read()).decode('utf-8')
        
        return jsonify({
            'status': img_base64,
            'objects': detected_objects
        })
    
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return jsonify({'error': 'An unexpected error occurred'}), 500

# ... rest of the code remains the same

@app.route('/test', methods=['GET','POST'])
def test():
    print("log: got at test", file=sys.stderr)
    return jsonify({'status':'success'})

@app.route('/')
def home():
    return render_template('index.html')

@app.after_request
def after_request(response):
    print("log: setting cors", file=sys.stderr)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

if __name__ == '__main__':
    # Check for required YOLO files
    required_files = [
        'yolov3.weights', 
        'coco.names', 
        'yolov3.cfg'
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"[WARNING] Required file {file} is missing!")
    
    app.run(debug=True)