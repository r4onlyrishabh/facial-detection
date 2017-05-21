from imutils.video import VideoStream
from imutils import face_utils
import imutils
import dlib
import cv2
import time

shapePredictorPath = 'dataset/shape_predictor_68_face_landmarks.dat'

faceDetector = dlib.get_frontal_face_detector()
facialLandmarkPredictor = dlib.shape_predictor(shapePredictorPath)

vs = VideoStream(usePiCamera = False).start()
time.sleep(2.0)

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = faceDetector(gray, 0)
	for (i, face) in enumerate(faces):
		facialLandmarks = facialLandmarkPredictor(gray, face)
		facialLandmarks = face_utils.shape_to_np(facialLandmarks)
 		
		(x, y, w, h) = face_utils.rect_to_bb(face)
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
		cv2.putText(frame, '#{}'.format(i+1), (x, y-10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

		for (x, y) in facialLandmarks:
			cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
	  
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
