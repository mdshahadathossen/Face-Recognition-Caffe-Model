Face Recognition Using Caffe Model
---
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
- opencv-python
- imutils
- numpy
- scikit-learn
- Bash: pip install opencv-python imutils numpy scikit-learn
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
     Bash:
     python extract_embeddings.py --dataset dataset --embeddings output/PyPower_embed.pickle --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7
   - Train the Model
     Bash: python train_model.py --embeddings output/PyPower_embed.pickle --recognizer output/PyPower_recognizer.pickle --le output/PyPower_label.pickle
   - Test with Image
     Bash: python recognize.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle --image images/test.jpg
   - Real-Time Webcam Recognition
     Bash: python recognize_video.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle
---
7. 🧪 Result:
- Green box = recognized person
- Red box = unknown or below threshold
---
9. Perform face recognition on a test image:  
![recognize_img3](https://github.com/user-attachments/assets/3dca598b-3d8a-4a9d-9397-08287a4db81b)
---
10. Recognize face(s) in real-time from webcam:
![recognize_video1](https://github.com/user-attachments/assets/99b5dbc3-8942-4b18-9de6-764ec9283421)
![recognize_video4](https://github.com/user-attachments/assets/ed027cdb-bcc6-45c2-8493-cddfe38e590e)
![recognize_video4](https://github.com/user-attachments/assets/dc30b407-063c-402f-a844-bdbca3b275fd)
![WhatsApp Image 2025-04-24 at 22 59 07](https://github.com/user-attachments/assets/ed878411-d274-4e81-9ab9-37261ec8f24d)





