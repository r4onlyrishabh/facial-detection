from imutils import face_utils
import imutils
import numpy as np
import dlib
import cv2

shapePredictorPath = 'dataset/shape_predictor_68_face_landmarks.dat'
imagePath = '17799012_1972178646326869_3536847167099903974_n.jpg'

faceDetector = dlib.get_frontal_face_detector()
facialLandmarkPredictor = dlib.shape_predictor(shapePredictorPath)
print 'done'

image = cv2.imread(imagePath)
#image = imutils.resize(image, width = 500) ##??
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceDetector(grayImage, 1)
for (i, face) in enumerate(faces):
	faceLandmarks = facialLandmarkPredictor(grayImage, face)
	faceLandmarks = face_utils.shape_to_np(faceLandmarks)
	
	for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
		clone = image.copy()
		cv2.putText(clone, name, (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		
		for (x, y) in faceLandmarks[i:j]:
			cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
		
		(x, y, w, h) = cv2.boundingRect(np.array([faceLandmarks[i:j]]))
		roi = image[y:y+h, x:x+w]
		roi = imutils.resize(roi,width=250,inter=cv2.INTER_CUBIC)

		cv2.imshow(name, roi)
		cv2.imshow('Image', clone)
		cv2.waitKey(0)

	output = face_utils.visualize_facial_landmarks(image, faceLandmarks)
	cv2.imshow("Output", output)
	cv2.waitKey(0)
