# Emotion-Song-Recommendation
A machine learning project that recommends YouTube songs based on user's detected emotion using OpenCV and Keras.

## Features
- Real-time emotion detection
- Trained CNN emotion model
- YouTube API song recommendation
- Uses `opencv-python`, `keras`, etc.

## 📁 Folder Structure
project/

├── model/ # Trained model (.h5)

├── haarcascade/ # Face detection XML

├── data/ # FER-2013 dataset

├── emotion_detector.py

├── recommend.py

├── main.py

└── README.md


---

## 📥 Dataset Setup (Manual Download)

We use the **FER-2013 Facial Emotion Recognition dataset**.

### 🔹 Step 1: Download Dataset ZIP

Download it manually from:  
👉 [https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)

Click on **"Download"** to get `fer2013.zip`.

### 🔹 Step 2: Extract to `data/` Folder

1. Extract the downloaded `fer2013.zip`
2. Move the extracted folder into your project under `data/fer2013/`

Your structure should look like:
Emotion-YouTube-Song-Recommendation/

└── data/

├── train/

└── test/


### 🔹 Step 3: Use in Code

In your Python code, set dataset paths like:

```python
train_dir = "data/fer2013/train/"
test_dir = "data/fer2013/test/"

