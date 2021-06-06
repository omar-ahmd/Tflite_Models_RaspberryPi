
import tflite_runtime.interpreter as tflite
import cv2
import numpy as np
import time
start_time = time.time()
interpreter = tflite.Interpreter(model_path="deeplab.tflite")
original_image = cv2.imread("bicycle.jpg")
image = cv2.resize(original_image,(257,257))
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image = (image.astype(np.float32)/127.5)-1
input_image = np.expand_dims(image,0)

labels = np.loadtxt("labelmap.txt",dtype = str, delimiter="/n")


interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.

interpreter.set_tensor(input_details[0]['index'], input_image)
invoke_time = time.time()
interpreter.invoke()
print("invoke time:", time.time()-invoke_time, "sec")
# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])

categ = np.argmax(output_data[0],axis=2)
mask = (categ==2).astype(np.uint8)*255
mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))
seg_img = cv2.bitwise_and(original_image, original_image, mask = mask)
x,y,w,h = cv2.boundingRect(mask)
img = cv2.rectangle(original_image,(x,y),(x+w,y+h),(0,255,0),2)

img = cv2.putText(img, labels[2], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
print("whole time",time.time()-start_time,"sec")
cv2.imwrite("mask.jpg",mask)
cv2.imwrite("segmented_img.jpg", seg_img)
cv2.imwrite("seg_img.jpg", img)
