from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io

# TensorFlow import that works on Windows
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Load TFLite model - correct method for TF 2.13
print("Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path="plant_disease_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"✅ Model loaded!")
print(f"   Input shape: {input_details[0]['shape']}")
print(f"   Output shape: {output_details[0]['shape']}")

# Disease labels
LABELS = ['Aphids', 'Healthy', 'Leaf Miner']



@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'running',
        'message': 'CitrusCare API is running!',
        'model': 'plant_disease_model.tflite',
        'classes': LABELS
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if image was sent
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        # Get image from request
        file = request.files['image']
        print(f"📸 Received image: {file.filename}")
        
        # Load and preprocess image
        img = Image.open(io.BytesIO(file.read()))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to 224x224 (MobileNetV2 input size)
        img = img.resize((224, 224))
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)
        
        # MobileNetV2 preprocessing: scale to [-1, 1]
        img_array = (img_array / 127.5) - 1.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        print(f"🔧 Preprocessed shape: {img_array.shape}")
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        
        # Get predictions
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        
        print(f"🔍 Raw predictions: {predictions}")
        
        # Find best prediction
        class_index = int(np.argmax(predictions))
        confidence = float(predictions[class_index])
        disease = LABELS[class_index]
        
        # Format all results
        results = []
        for i, prob in enumerate(predictions):
            results.append({
                'label': LABELS[i],
                'confidence': float(prob),
                'index': i
            })
        
        # Sort by confidence
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        print(f"✅ Prediction: {disease} ({confidence*100:.1f}%)")
        
        # Return response in same format as mobile app
        return jsonify({
            'disease': disease,
            'confidence': confidence,
            'probabilities': predictions.tolist(),
            'classIndex': class_index,
            'all_predictions': results
        })
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌿 CitrusCare API Server")
    print("="*60)
    print("Server starting...")
    print("="*60 + "\n")
    
    # Use PORT from environment variable (Railway provides this)
    import os
    port = int(os.environ.get('PORT', 5000))
    
    app.run(host='0.0.0.0', port=port, debug=False)