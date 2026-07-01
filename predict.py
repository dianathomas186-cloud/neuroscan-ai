# ============================================================
#   BRAIN TUMOR CLASSIFICATION - PREDICTION SCRIPT
#   Model: ResNet50 + CNN Transfer Learning
#   VS Code - Prediction and Visualization
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import warnings
warnings.filterwarnings('ignore')

# ── STEP 1 : CONFIGURATION ───────────────────────────────────
MODEL_PATH      = 'brain_tumor_classification_model.h5'
TEST_IMAGES_DIR = 'test_images'
IMG_SIZE        = 224
CLASS_NAMES     = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Class descriptions for better understanding
CLASS_INFO = {
    'glioma'     : 'Glioma - Tumor in brain/spine glial cells',
    'meningioma' : 'Meningioma - Tumor in brain membrane',
    'notumor'    : 'No Tumor - Healthy brain scan',
    'pituitary'  : 'Pituitary - Tumor in pituitary gland'
}

# ── STEP 2 : LOAD MODEL ──────────────────────────────────────
print("=" * 55)
print("  BRAIN TUMOR CLASSIFICATION - PREDICTION SYSTEM")
print("=" * 55)
print("\nLoading trained model...")

model = load_model(MODEL_PATH)
print("Model loaded successfully! ✅")
print(f"Model input shape : {model.input_shape}")
print(f"Model output shape: {model.output_shape}")

# ── STEP 3 : LOAD AND PREPROCESS IMAGES ──────────────────────
def preprocess_image(image_path):
    """
    Load and preprocess a single MRI image for prediction
    - Resize to 224x224 (ResNet50 input size)
    - Normalize pixel values to [0,1]
    - Expand dimensions for batch processing
    """
    img       = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img)
    img_array = img_array / 255.0          # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# ── STEP 4 : PREDICT SINGLE IMAGE ────────────────────────────
def predict_single_image(image_path):
    """
    Predict brain tumor class for a single MRI image
    Returns predicted class, confidence, and all probabilities
    """
    img_array    = preprocess_image(image_path)
    predictions  = model.predict(img_array, verbose=0)
    pred_index   = np.argmax(predictions[0])
    pred_class   = CLASS_NAMES[pred_index]
    confidence   = predictions[0][pred_index] * 100
    all_probs    = {CLASS_NAMES[i]: predictions[0][i] * 100
                    for i in range(len(CLASS_NAMES))}
    return pred_class, confidence, all_probs

# ── STEP 5 : GET ALL TEST IMAGES ─────────────────────────────
print(f"\nScanning test images from '{TEST_IMAGES_DIR}' folder...")
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
image_files = [
    f for f in os.listdir(TEST_IMAGES_DIR)
    if f.lower().endswith(valid_extensions)
]

if len(image_files) == 0:
    print("❌ No images found in test_images folder!")
    print("Please add some MRI images and try again.")
else:
    print(f"Found {len(image_files)} images ✅")

# ── STEP 6 : PREDICT ALL IMAGES ──────────────────────────────
print("\n" + "=" * 55)
print("  PREDICTIONS")
print("=" * 55)

results = []
for img_file in image_files:
    img_path              = os.path.join(TEST_IMAGES_DIR, img_file)
    pred_class, conf, all_probs = predict_single_image(img_path)
    results.append({
        'file'      : img_file,
        'path'      : img_path,
        'predicted' : pred_class,
        'confidence': conf,
        'all_probs' : all_probs
    })
    print(f"\nImage     : {img_file}")
    print(f"Predicted : {pred_class.upper()}")
    print(f"Confidence: {conf:.2f}%")
    print(f"Info      : {CLASS_INFO[pred_class]}")
    print(f"All Probabilities:")
    for cls, prob in all_probs.items():
        bar = '█' * int(prob / 5)
        print(f"  {cls:<12}: {prob:6.2f}% {bar}")
    print("-" * 55)

# ── STEP 7 : VISUALIZE PREDICTIONS ───────────────────────────
print("\nGenerating prediction visualization...")

# Color coding for each class
CLASS_COLORS = {
    'glioma'    : '#FF6B6B',
    'meningioma': '#FFD93D',
    'notumor'   : '#6BCB77',
    'pituitary' : '#4D96FF'
}

n_images = len(results)
fig, axes = plt.subplots(n_images, 2, figsize=(14, 5 * n_images))
fig.suptitle('Brain Tumor Classification - Prediction Results',
             fontsize=16, fontweight='bold', y=1.01)

if n_images == 1:
    axes = [axes]

for idx, result in enumerate(results):
    color = CLASS_COLORS[result['predicted']]

    # Left: MRI Image
    img = load_img(result['path'], target_size=(IMG_SIZE, IMG_SIZE))
    axes[idx][0].imshow(img, cmap='gray')
    axes[idx][0].set_title(
        f"File: {result['file']}\nPredicted: {result['predicted'].upper()} "
        f"({result['confidence']:.1f}%)",
        fontsize=11, color=color, fontweight='bold'
    )
    axes[idx][0].axis('off')

    # Add colored border based on prediction
    for spine in axes[idx][0].spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(3)

    # Right: Probability Bar Chart
    classes = list(result['all_probs'].keys())
    probs   = list(result['all_probs'].values())
    bar_colors = [CLASS_COLORS[c] for c in classes]

    bars = axes[idx][1].barh(classes, probs, color=bar_colors, edgecolor='white')
    axes[idx][1].set_xlim(0, 100)
    axes[idx][1].set_xlabel('Confidence (%)', fontsize=11)
    axes[idx][1].set_title('Class Probabilities', fontsize=11, fontweight='bold')
    axes[idx][1].grid(axis='x', alpha=0.3)

    # Add percentage labels on bars
    for bar, prob in zip(bars, probs):
        axes[idx][1].text(
            bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f'{prob:.1f}%', va='center', fontsize=10, fontweight='bold'
        )

plt.tight_layout()
plt.savefig('prediction_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nVisualization saved as 'prediction_results.png' ✅")

# ── STEP 8 : FINAL SUMMARY ───────────────────────────────────
print("\n" + "=" * 55)
print("  PREDICTION SUMMARY")
print("=" * 55)
print(f"Total images predicted : {len(results)}")
print(f"\nResults:")
for result in results:
    print(f"  {result['file']:<30} → {result['predicted'].upper():<12} "
          f"({result['confidence']:.1f}%)")

print("\n" + "=" * 55)
print("  PROJECT COMPLETE!")
print("  Brain Tumor Classification Using CNN")
print("  and Transfer Learning")
print("=" * 55)