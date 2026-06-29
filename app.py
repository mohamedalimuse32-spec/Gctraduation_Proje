import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mob_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as res_preprocess
from collections import Counter

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary', 'Unknown']
MIN_CONFIDENCE = 80.0

print("Loading models...")
models = {
    'EfficientNetB0': load_model('models/efficientnetb0_v2_fixed.keras', compile=False),
    'MobileNetV2':    load_model('models/mobilenetv2_v2_fixed.keras',    compile=False),
    'ResNet50':       load_model('models/resnet50_v2_fixed.keras',        compile=False)
}
print("All models loaded successfully!")

def prepare_image(img_path, model_name):
    img       = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    if model_name == 'EfficientNetB0':
        img_array = eff_preprocess(img_array)
    elif model_name == 'MobileNetV2':
        img_array = mob_preprocess(img_array)
    elif model_name == 'ResNet50':
        img_array = res_preprocess(img_array)
    return np.expand_dims(img_array, axis=0)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file     = request.files['file']
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(img_path)

    # Run all 3 models
    results      = {}
    pred_classes = []

    for name, mdl in models.items():
        img_array  = prepare_image(img_path, name)
        prediction = mdl.predict(img_array, verbose=0)[0]
        pred_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = float(np.max(prediction) * 100)
        pred_classes.append(pred_class)
        results[name] = {
            'prediction': pred_class,
            'confidence': round(confidence, 2)
        }

    # Check if majority vote is Unknown
    vote_counts_all   = Counter(pred_classes)
    initial_diagnosis = vote_counts_all.most_common(1)[0][0]

    if initial_diagnosis == 'Unknown':
        return jsonify({
            'valid'  : False,
            'message': 'This image does not appear to be a brain MRI scan. Please upload a valid brain MRI scan.',
            'results': results
        })

    # All 3 models must be above 80%
    low_confidence_models = [n for n, r in results.items() if r['confidence'] < MIN_CONFIDENCE]

    if len(low_confidence_models) > 0:
        low_info = ', '.join([f"{n} ({results[n]['confidence']}%)" for n in low_confidence_models])
        return jsonify({
            'valid'  : False,
            'message': f'This image does not appear to be a valid brain MRI scan. The following models were not confident enough: {low_info}.',
            'results': results
        })

    # All 3 passed — majority voting
    vote_counts     = Counter(pred_classes)
    final_diagnosis = vote_counts.most_common(1)[0][0]

    agreeing        = [n for n, r in results.items() if r['prediction'] == final_diagnosis]
    best_model      = max(agreeing, key=lambda n: results[n]['confidence'])
    best_confidence = results[best_model]['confidence']
    avg_confidence  = sum(r['confidence'] for r in results.values()) / 3

    return jsonify({
        'valid'          : True,
        'results'        : results,
        'final_diagnosis': final_diagnosis,
        'best_model'     : best_model,
        'best_confidence': best_confidence,
        'avg_confidence' : round(avg_confidence, 2),
        'votes'          : vote_counts[final_diagnosis]
    })

if __name__ == '__main__':
    app.run(debug=True)