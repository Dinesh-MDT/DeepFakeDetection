# 🎭 DeepFake Detection System

An AI-powered web application that detects whether a video is **Real or Fake** using deep learning techniques. This system analyzes video frames, extracts facial features, and performs classification using a trained model.

---

## 🚀 Overview

DeepFake content is becoming increasingly realistic and dangerous. This project aims to provide a reliable solution to identify manipulated videos using a trained deep learning model.

The application allows users to upload videos and get predictions along with confidence scores.

---

## 🧠 Features

* 🎥 Upload and analyze video files
* 🧍 Face extraction from video frames
* 🤖 Deep learning-based classification (Real vs Fake)
* 📊 Confidence score for predictions
* ⚡ Fast and interactive UI using Streamlit

---

## 🛠 Tech Stack

* **Frontend/UI:** Streamlit
* **Backend:** Python
* **ML Framework:** TensorFlow / Keras
* **Computer Vision:** OpenCV
* **Other Libraries:** NumPy

---

## 📦 Project Structure

```
Deep_Fake/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── model_final.keras/   (not included in repo)
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```
git clone https://github.com/Dinesh-MDT/DeepFakeDetection.git
cd DeepFakeDetection
```

---

### 2. Install dependencies

```
pip install -r requirements.txt
```

---

### 3. Download the trained model

Download the model from the link below:

👉 https://drive.google.com/file/d/1acCBpAedUVP0WkI_0j069dBxk-YIF_NW/view?usp=sharing

After downloading:

* Extract the ZIP file
* Place the folder inside the project directory

```
DeepFakeDetection/
└── model_final.keras/
```

---

### 4. Run the application

```
streamlit run app.py
```

---

## 📸 Demo

*Add screenshots of your application here*

Example:

* Upload screen
* Prediction result screen

---

## ⚠️ Important Notes

* The model file is not included due to size limitations
* Make sure the model is placed correctly before running the app
* If the model is missing, the application will not work

---

## 📌 Future Improvements

* 🔄 Real-time video detection
* 🌐 Deploy as a web application (Streamlit Cloud / AWS)
* 📈 Improve model accuracy with larger datasets
* 🧪 Add evaluation metrics and visualizations


## 🤝 Contributing

Contributions are welcome. Feel free to fork the repository and submit pull requests.

## 📜 License

This project is for educational purposes.



## ⭐ Support

If you found this project useful, consider giving it a star ⭐
