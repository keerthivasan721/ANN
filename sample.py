# Import essential libraries
import requests
import cv2
import numpy as np
import imutils
import tensorflow as tf
import numpy as np

# Replace the below URL with your own. Make sure to add "/shot.jpg" at last.
url = "http://10.238.60.248:8080/shot.jpg"

# While loop to continuously fetching data from the Url
while True:
 img_resp = requests.get(url)
 img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
 img = cv2.imdecode(img_arr, -1)
 img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 img1 = cv2.resize(img1,(28,28))
 img = imutils.resize(img, width=1000, height=1800)
#  img1 = imutils.resize(img, width=28, height=28)
 print(img1.shape)
 cv2.imshow("Android_cam", img)
 model = tf.keras.models.load_model("model.h5")
 pred = model.predict(img1.reshape(1,-1))
 print(np.argmax(pred,axis=1)[0])
 # Press Esc key to exit
 if cv2.waitKey(1) == 27:
  break

cv2.destroyAllWindows()
