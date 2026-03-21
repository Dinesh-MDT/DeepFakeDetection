INTRODUCTION:

This project is an AI-powered DeepFake Detection Dashboard designed to identify manipulated videos using a hybrid deep learning approach. 
It processes video input by extracting facial frames and analyzing them through a combination of convolutional and sequential models to determine whether the content is real or fake.

Unlike basic classifiers, this system provides explainable insights, including frame-level analysis, confidence scoring, and visual evidence, all presented through an interactive, 
production-style dashboard.

## 📸 How It Works

1. Upload a video
2. Faces are extracted frame-by-frame
3. Frames are resized and preprocessed
4. Model predicts:

   * Overall video label (REAL / FAKE)
   * Frame-wise probabilities
5. Dashboard visualizes results

## 📊 Sample Output

* ✅ Prediction: REAL / FAKE
* 📈 Confidence Score
* 📉 Frame-wise graph
* 🎯 Top suspicious frames
* 📄 Downloadable report


## ⚠️ Limitations

* Accuracy ~88.56% (can improve with larger dataset)
* Slower inference on CPU
* Performance depends on face detection quality

## 🚀 Future Improvements

* 🔥 Improve model accuracy (EfficientNetB3 / ViT)
* 🎥 Real-time webcam detection
* ☁️ Cloud deployment (Streamlit Cloud / AWS)
* 📊 Advanced heatmap visualizations
* 📦 Multi-video batch processing

## ⭐ If you like this project

Give it a ⭐ on GitHub — it helps a lot!

---

## 📜 License

This project is for educational and research purposes.
