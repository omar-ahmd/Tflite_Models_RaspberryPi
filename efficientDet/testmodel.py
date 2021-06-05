import tflite_runtime.interpreter as tflite
import cv2
import numpy as np

interpreter = tflite.Interpreter(model_path="efficientdet.tflite")
original_image = cv2.imread("messi.jpeg")
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

interpreter.invoke()

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
categories = categories[probability>0.6]
boxesPosition = boxesPosition[probability>0.6]
print(boxesPosition)
for i in range(len(categories)):
    label = labels[(int)(categories[i])]
    print(label)
    originali_mage = cv2.rectangle(original_image, (boxesPosition[i][1],boxesPosition[i][0]), (boxesPosition[i][3],boxesPosition[i][2]), (36,255,12), 1)
    cv2.putText(original_image, label, (boxesPosition[i][1],boxesPosition[i][0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
cv2.imwrite("detect.jpg", original_image)
output_data4 = interpreter.get_tensor(output_details[3]['index'])
print(output_data4)

