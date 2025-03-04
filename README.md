# PRODIGY_ML_04
📌 Overview
This project aims to develop an AI-powered hand gesture recognition system using Convolutional Neural Networks (CNNs) and OpenCV. The system is trained on the LeapGestRecog dataset, which contains grayscale images of different hand gestures. The goal is to classify hand gestures accurately and enable real-time recognition through a webcam.

🎯 Objectives
✅ Build a deep learning model for gesture classification
✅ Train the model on the LeapGestRecog dataset
✅ Implement real-time gesture recognition using a webcam
✅ Improve accuracy using data augmentation and transfer learning
✅ Deploy the model for real-world applications

📂 Dataset: LeapGestRecog
The LeapGestRecog dataset contains images of 10 different hand gestures, captured from multiple individuals. The images are grayscale and labeled according to gesture type.

📌 Gesture Classes:
✋ Palm
👊 Fist
👍 Thumb
☝️ Index
✌️ L
🤙 OK
👋 Palm Moved
✋ Fist Moved
🤏 C
👇 Down
🛠️ Tech Stack & Tools
🔹 Programming Language: Python
🔹 Deep Learning Framework: TensorFlow/Keras
🔹 Computer Vision: OpenCV
🔹 Dataset Handling: NumPy, Pandas
🔹 Deployment: Flask/Streamlit for Web App (Optional)

📌 Project Workflow
1️⃣ Data Preprocessing
Extract images from the dataset
Convert images to grayscale
Resize images to 64x64 pixels
Normalize pixel values
2️⃣ Model Development
Implement a CNN architecture with convolutional and pooling layers
Use Softmax activation for multi-class classification
Train on the dataset using Adam optimizer
3️⃣ Model Training & Evaluation
Split dataset into training and testing sets
Train the model on grayscale images
Evaluate accuracy and loss metrics
4️⃣ Real-Time Gesture Recognition
Capture video frames using OpenCV
Preprocess frames and make predictions
Display recognized gestures in real-time
5️⃣ Optimization & Future Improvements
Implement data augmentation to improve accuracy
Use transfer learning (MobileNetV2)
Convert model to TensorFlow Lite for mobile applications
