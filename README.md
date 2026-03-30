# 🌿 MaizeDoc — AI Maize Disease Detection

> Instant maize leaf disease diagnosis powered by deep learning. Upload a photo, get a diagnosis in seconds — in English or Kinyarwanda.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.1-orange?style=flat-square&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-3.1.3-black?style=flat-square&logo=flask)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 🔍 What It Does

MaizeDoc is a web application that uses a Convolutional Neural Network (VGG16 Transfer Learning) to detect diseases on maize plant leaves. Farmers can take a photo of a maize leaf directly from their phone and receive an instant diagnosis with treatment recommendations.

### Detected Conditions
| Condition | Severity |
|---|---|
| ✅ Healthy | Safe |
| 🟡 Common Rust | Warning |
| 🟡 Gray Leaf Spot (Cercospora) | Warning |
| 🔴 Northern Leaf Blight | Danger |

---

## 🚀 Live Demo

**[→ Open MaizeDoc](https://maizedoc.onrender.com)**

Works on any phone browser — no app install needed.

---

## 🧠 Model

- **Architecture:** VGG16 pre-trained on ImageNet, fine-tuned on a custom maize disease dataset
- **Dataset:** [PlantVillage Dataset](https://www.kaggle.com/emmarex/plantdisease) + additional scraped images
- **Classes:** 4 (Healthy, Common Rust, Gray Leaf Spot, Northern Leaf Blight)
- **Input size:** 224 × 224 px
- **Languages supported:** English, Kinyarwanda

---

## 🛠 Tech Stack

- **Backend:** Flask 3.1.3, Python 3.10
- **ML:** TensorFlow 2.16.1, tf-keras 2.16.0
- **Frontend:** HTML5, CSS3, Vanilla JS
- **Deployment:** Render (free tier)
- **Static files:** WhiteNoise

---

## 💻 Run Locally

```bash
# Clone the repo
git clone https://github.com/Chryso1392001/MaizeDoc.git
cd MaizeDoc

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

---

## 📁 Project Structure

```
MaizeDoc/
├── app.py                      # Flask application
├── maize_disease_fixed.h5      # Trained model
├── requirements.txt
├── Procfile                    # For Render deployment
├── runtime.txt
├── static/
│   ├── images/
│   │   └── farmer.png          # Hero image
│   └── uploads/                # User uploaded images
└── templates/
    ├── home.html               # Upload page
    └── predict.html            # Results page
```

---

## 🌍 Deployment

Deployed on [Render](https://render.com) free tier.

- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `gunicorn app:app`
- **Runtime:** Python 3.10

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">Built with ❤️ for farmers in Rwanda 🇷🇼</p>
