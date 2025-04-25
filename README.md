# Face Recognition Using Caffe Model

This project shows how to build a real-time face recognition system using OpenCV’s deep learning module and a pre-trained Caffe model. It uses a face detector and face embedder with an SVM classifier to recognize faces in images and live video.
---
📁 Project Files and Structure
1. `dataset/` - contains folders of face images for each person.
2. `face_detection_model/`
- includes:
- `deploy.prototxt`
- `res10_300x300_ssd_iter_140000.caffemodel`
3. `output/`
- stores generated models:
- `PyPower_embed.pickle`
- `PyPower_recognizer.pickle`
- `PyPower_label.pickle`
4. `images/` - test images for recognition.
5. `extract_embeddings.py` - script to create 128-d face embeddings.
6. `train_model.py` - script to train the SVM recognizer.
7. `recognize.py` - for static image recognition.
8. `recognize_video.py` - for real-time webcam face recognition.
---
📦 Download Required Model File
- Download openface_nn4.small2.v1.t7 from Google Drive : https://drive.google.com/file/d/13Epu_vfZjfRmxAzZdA_1kKiXnBQ2ahzR/view?usp=sharing
- After downloading, place it inside your main project folder.
---
⚙️ Installation
1. Make sure Python 3 is installed.
2. Install the required libraries:
opencv-python
imutils
numpy
scikit-learn

Command terminal: pip install opencv-python imutils numpy scikit-learn
---
📦 What each one does:
- Package	Purpose:
- opencv-python	Used for face detection, image I/O, drawing boxes, webcam feed
- imutils	Simplifies OpenCV functions like resizing and video streaming
- numpy	Handles array operations, image blobs, numerical computations
- scikit-learn	Used for training the SVM classifier and label encoding
---
📝 How to Run This Face Recognition Project (Step-by-Step)
1. 🔽 Clone or Download the Project
2. 🐍 Install Python (if not installed)
3. ⚙️ Install Required Libraries
4. 📂 Prepare the Dataset
5. 📥 Download the Embedding Model File (Download manually:Download openface_nn4.small2.v1.t7 from Google Drive - given above)
6. 🧠 Extract Embeddings from Your Dataset
   - Extract Embeddings
     Bash: python extract_embeddings.py -i dataset -e output/PyPower_embed.pickle -d face_detection_model -m openface_nn4.small2.v1.t7
   - Train the Model
     Bash: python train_model.py -e output/PyPower_embed.pickle -r output/PyPower_recognizer.pickle -l output/PyPower_label.pickle
   - Test with Image
     Bash: python recognize.py -i images/test.jpg -d face_detection_model -m openface_nn4.small2.v1.t7 -r output/PyPower_recognizer.pickle -l output/PyPower_label.pickle -c 0.5 -t 0.6
   - Real-Time Webcam Recognition
     Bash: python recognize_video.py -d face_detection_model -m openface_nn4.small2.v1.t7 -r output/PyPower_recognizer.pickle -l output/PyPower_label.pickle -c 0.5 -t 0.6
7. 🧪 Result
Green box = recognized person
Red box = unknown or below threshold
