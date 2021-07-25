#!/usr/bin/python3

import tflite_runtime.interpreter as tflite
import cv2
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser(description='A test program.')
parser.add_argument("-c", "--confidence", help="probability")
parser.add_argument("-i", "--image_path", help="image path")
args = parser.parse_args()


start_time = time.time()
interpreter = tflite.Interpreter(model_path="efficientdet.tflite")
original_image = cv2.imread("Images/"+ args.image_path)
if original_image is  None:
    raise TypeError("Null Image")
image = cv2.resize(original_image,(320,320))
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
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
boxesPosition = interpreter.get_tensor(output_details[0]['index'])
boxesPosition[:,:,0] = boxesPosition[:,:,0]*original_image.shape[0]
boxesPosition[:,:,1] = boxesPosition[:,:,1]*original_image.shape[1]
boxesPosition[:,:,2] = boxesPosition[:,:,2]*original_image.shape[0]
boxesPosition[:,:,3] = boxesPosition[:,:,3]*original_image.shape[1]
boxesPosition = boxesPosition.astype(int)
probability = interpreter.get_tensor(output_details[2]['index'])

categories = interpreter.get_tensor(output_details[1]['index'])
categories = categories[probability>float(args.confidence)]

boxesPosition = boxesPosition[probability>float(args.confidence)]
probability = probability[probability>float(args.confidence)]
for i in range(len(categories)):
    label = labels[(int)(categories[i])]
    print(label, " " , probability[i])
    originali_mage = cv2.rectangle(original_image, (boxesPosition[i][1],boxesPosition[i][0]), (boxesPosition[i][3],boxesPosition[i][2]), (0,0,0), 2)
    cv2.putText(original_image, label, (boxesPosition[i][1],boxesPosition[i][0]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
cv2.imwrite("DetectedImages/"+ args.image_path, original_image)
print("whole time",time.time()-start_time,"sec")

