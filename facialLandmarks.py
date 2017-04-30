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
	facialLandmarks = facialLandmarkPredictor(grayImage, face)
	facialLandmarks = face_utils.shape_to_np(facialLandmarks)
	
	(x, y, w, h) = face_utils.rect_to_bb(face)
	cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
	
	cv2.putText(image, '#{}'.format(i+1), (x, y-10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	
	for (x, y) in facialLandmarks:
		cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

cv2.imshow("Output", image)
cv2.waitKey(0)
