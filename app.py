import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
 
from flask import Flask, render_template, request, url_for
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image as PILImage
from whitenoise import WhiteNoise
 
# --------------------- Flask Setup ---------------------
app = Flask(__name__)
app.wsgi_app = WhiteNoise(app.wsgi_app, root='static/', prefix='static')
 
app.config['UPLOAD_FOLDER'] = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
 
# --------------------- Load Model ---------------------
MODEL_PATH = 'maize_disease_fixed.h5'
 
try:
    import tf_keras as keras
    model = keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None
 
# --------------------- Class Labels ---------------------
class_labels = [
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
]
 
# --------------------- Severity Map ---------------------
disease_severity_map = {
    'Corn_(maize)___healthy': 'safe',
    'Corn_(maize)___Common_rust_': 'warning',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'warning',
    'Corn_(maize)___Northern_Leaf_Blight': 'danger'
}
 
# --------------------- Disease Recommendations ---------------------
disease_recommendations_en = {
    'Corn_(maize)___healthy': (
        "The plant is healthy. Maintain proper irrigation and nutrient balance. "
        "Continue regular monitoring for early detection of any disease. "
        "Use crop rotation to prevent soil depletion and keep pests under control."
    ),
    'Corn_(maize)___Common_rust_': (
        "Common Rust detected. Remove affected leaves and apply recommended fungicides. "
        "Ensure proper spacing between plants to reduce humidity. "
        "Monitor fields regularly to catch early infections and prevent spread."
    ),
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': (
        "Gray Leaf Spot detected. Use disease-free seeds, remove crop residues, "
        "and apply appropriate fungicides if necessary. "
        "Practice crop rotation and proper irrigation management."
    ),
    'Corn_(maize)___Northern_Leaf_Blight': (
        "Northern Leaf Blight detected. Rotate crops, avoid dense planting, "
        "and apply preventive fungicides as recommended. "
        "Maintain balanced fertilization to strengthen plant resistance."
    )
}
 
disease_recommendations_kinyarwanda = {
    'Corn_(maize)___healthy': (
        "Igihingwa kiri muzima. Komeza guha amazi bihagije no gukoresha ifumbire ku gipimo gikwiye. "
        "Komeza kugenzura buri gihe kugirango umenye indwara hakiri kare. "
        "Koresha guhinduranya imyaka y'ibihingwa kugirango ubutaka budasaza kandi ibyonnyi bikumirwe."
    ),
    'Corn_(maize)___Common_rust_': (
        "Haragaragaye Common Rust. Kuraho amababi yafashwe n'indwara no gukoresha imiti ikwiriye. "
        "Hagira intera ihagije hagati y'ibihingwa kugirango uhumure neza. "
        "Genzura umurima kenshi kugirango uhangane n'ikwirakwira ry'indwara."
    ),
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': (
        "Haragaragaye Gray Leaf Spot. Koresha imbuto zidafite indwara, kuraho ibisigazwa by'ibihingwa, "
        "no gukoresha imiti ikwiriye. "
        "Koresha guhinduranya imyaka y'ibihingwa no gucunga neza amazi."
    ),
    'Corn_(maize)___Northern_Leaf_Blight': (
        "Haragaragaye Northern Leaf Blight. Hinduranya imyaka y'ibihingwa, wirinde gutera cyane, "
        "no gukoresha imiti mbere y'igihe. "
        "Komeza gukoresha ifumbire ku gipimo gikwiye kugirango igihingwa kigire imbaraga."
    )
}
 
def get_recommendation(pred_class, language='en'):
    if language == 'kin':
        return disease_recommendations_kinyarwanda.get(pred_class, "Nta nama ibonetse.")
    return disease_recommendations_en.get(pred_class, "No recommendation available.")
 
# --------------------- Helper Functions ---------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 
def predict_image(img_path, language='en'):
    if model is None:
        return "Model Error", 0, "Model failed to load.", "neutral"
 
    img = PILImage.open(img_path).convert('RGB').resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array[..., ::-1]  # RGB -> BGR
    img_array[..., 0] -= 103.939
    img_array[..., 1] -= 116.779
    img_array[..., 2] -= 123.68
    img_array = np.expand_dims(img_array, axis=0)
 
    pred_probs = model.predict(img_array, verbose=0)
    pred_class = class_labels[np.argmax(pred_probs)]
    confidence = float(np.max(pred_probs) * 100)
 
    recommendation = get_recommendation(pred_class, language)
    severity = disease_severity_map.get(pred_class, 'neutral')
 
    return pred_class, confidence, recommendation, severity
 
# --------------------- Routes ---------------------
@app.route('/')
def home():
    return render_template('home.html')
 
@app.route('/predict', methods=['POST'])
def upload_and_predict():
    language = request.form.get('language', 'en')
 
    if 'file' not in request.files:
        return render_template('home.html', error="No file uploaded.")
 
    file = request.files['file']
 
    if file.filename == '':
        return render_template('home.html', error="No file selected.")
 
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
 
        pred_class, confidence, recommendation, severity = predict_image(file_path, language)
        display_class = pred_class.replace('Corn_(maize)___', '').replace('_', ' ').strip()
 
        return render_template(
            'predict.html',
            user_image=url_for('static', filename='uploads/' + filename),
            disease_name=display_class,
            disease_severity=severity,
            accuracy=round(confidence, 2),
            cure_recommendation=recommendation,
            language=language
        )
 
    return render_template('home.html', error="Invalid file type. Upload JPG or PNG.")
 
# --------------------- Run Server ---------------------
if __name__ == '__main__':
    app.run(debug=True)