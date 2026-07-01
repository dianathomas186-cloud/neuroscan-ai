# ============================================================
#   BRAIN TUMOR CLASSIFICATION - FLASK WEB APPLICATION
#   With Grad-CAM Heatmap Visualization & Interactive RAG Chat
# ============================================================

from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
import uuid
import warnings
warnings.filterwarnings('ignore')

# Import the RAG module functions
from rag import get_rag_report, query_rag

# ── CONFIGURATION ─────────────────────────────────────────────
app = Flask(__name__)

MODEL_PATH         = 'brain_tumor_classification_model.h5'
UPLOAD_FOLDER      = 'static/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}
IMG_SIZE           = 224
CLASS_NAMES        = ['glioma', 'meningioma', 'notumor', 'pituitary']

CLASS_INFO = {
    'glioma'    : 'Glioma is a tumor that occurs in the brain and spinal cord, arising from glial cells.',
    'meningioma': 'Meningioma is a tumor that arises from the meninges, the membranes surrounding the brain.',
    'notumor'   : 'No tumor detected. The brain scan appears healthy and normal.',
    'pituitary' : 'Pituitary tumor is an abnormal growth in the pituitary gland at the base of the brain.'
}

CLASS_COLORS = {
    'glioma'    : '#FF6B6B',
    'meningioma': '#FFD93D',
    'notumor'   : '#6BCB77',
    'pituitary' : '#4D96FF'
}

CLASS_ICONS = {
    'glioma'    : '🔴',
    'meningioma': '🟡',
    'notumor'   : '🟢',
    'pituitary' : '🔵'
}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ── LOAD MODEL ────────────────────────────────────────────────
print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully! ✅")

# ── HELPER FUNCTIONS ──────────────────────────────────────────
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img       = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_tumor(image_path):
    img_array   = preprocess_image(image_path)
    predictions = model.predict(img_array, verbose=0)
    pred_index  = np.argmax(predictions[0])
    pred_class  = CLASS_NAMES[pred_index]
    confidence  = float(predictions[0][pred_index] * 100)
    all_probs   = {
        CLASS_NAMES[i]: round(float(predictions[0][i] * 100), 2)
        for i in range(len(CLASS_NAMES))
    }
    return pred_class, confidence, all_probs, pred_index

# ── GRAD-CAM FUNCTION ─────────────────────────────────────────
def generate_gradcam(image_path, pred_index, save_path):
    try:
        # Find last conv layer in ResNet50
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer.name
                break

        if last_conv_layer is None:
            return None

        # Create Grad-CAM model
        grad_model = Model(
            inputs  = model.inputs,
            outputs = [model.get_layer(last_conv_layer).output,
                      model.output]
        )

        # Preprocess image
        img       = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.cast(img_array, tf.float32)

        # Get gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, pred_index]

        grads       = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap      = tf.squeeze(heatmap)
        heatmap      = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap      = heatmap.numpy()

        # Resize heatmap to image size
        heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
        )

        # Load original image
        original_img = cv2.imread(image_path)
        original_img = cv2.resize(original_img, (IMG_SIZE, IMG_SIZE))

        # Superimpose heatmap on original image
        superimposed = cv2.addWeighted(original_img, 0.6,
                                       heatmap_colored, 0.4, 0)

        # Save Grad-CAM image
        cv2.imwrite(save_path, superimposed)
        return save_path

    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None

# ── ROUTES ────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html',
                               error='No file selected!')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html',
                               error='No file selected!')

    if file and allowed_file(file.filename):
        # Save uploaded file
        ext         = file.filename.rsplit('.', 1)[1].lower()
        unique_name = f"{uuid.uuid4().hex}.{ext}"
        filename    = secure_filename(unique_name)
        filepath    = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Predict
        pred_class, confidence, all_probs, pred_index = predict_tumor(filepath)

        # Generate Grad-CAM
        gradcam_filename = f"gradcam_{filename}"
        gradcam_path     = os.path.join(app.config['UPLOAD_FOLDER'],
                                        gradcam_filename)
        gradcam_result   = generate_gradcam(filepath, pred_index, gradcam_path)

        # Generate RAG report based on predicted class and confidence
        rag_report       = get_rag_report(pred_class, confidence)

        result = {
            'predicted_class' : pred_class,
            'confidence'      : round(confidence, 2),
            'all_probs'       : all_probs,
            'class_info'      : CLASS_INFO[pred_class],
            'class_color'     : CLASS_COLORS[pred_class],
            'class_icon'      : CLASS_ICONS[pred_class],
            'image_path'      : f'uploads/{filename}',
            'gradcam_path'    : f'uploads/{gradcam_filename}' \
                                  if gradcam_result else None,
            'rag_report'      : rag_report
        }

        return render_template('result.html', result=result)

    return render_template('index.html',
                           error='Invalid file type!')

# AJAX route for the interactive chat box
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400
    
    question = data['question']
    # Extract the predicted class passed from the webpage (defaults to 'notumor')
    predicted_class = data.get('predicted_class', 'notumor')
    
    # Query RAG with BOTH the question and the predicted class
    answer = query_rag(question, predicted_class)
    
    return jsonify({'answer': answer})

# ── RUN APP ───────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)