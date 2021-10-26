# !pip install opencv-python
# !pip install opencv-python-headless
import tensorflow as tf
import cv2
import numpy as np

sess = tf.Session()

new_saver = tf.train.import_meta_graph('Sign_Detector.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('./'))

alphabets =  'ABCDEFGHIKLMNOPQRSTUVWXY'


cam = cv2.VideoCapture(0)

if not cam.isOpened():
  print ("Could not open cam")
  exit()

while(1):
    ret, frame = cam.read()
    if ret:
        
        frame = cv2.flip(frame,1)        
        ROI_frame = frame[100:400, 200:500].copy()
        ROI_gray = cv2.cvtColor(ROI_frame, cv2.COLOR_BGR2GRAY)
        ROI_small = cv2.resize(ROI_gray, (28,28))
        ROI_small_flat = cv2.resize(ROI_gray, (28*28,1))
        ROI = np.array(ROI_small_flat, dtype=np.float32)
        label = sess.run('Y_PRED_CLS:0', feed_dict={'X_INPUT:0':ROI})
        word = alphabets[int(label)]
        display = cv2.rectangle(frame.copy(),(200,100),(500,400),(0,255,0),2) 
        cv2.putText(display, word, (200, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.imshow('curFrame',display)
        cv2.imshow('Roi', ROI_gray)
        
        
        print()
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()