# check opencv gpu
import cv2

print(cv2.__version__)

if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    print("CUDA is available")
else:
    print("CUDA is not available")

# check tensorflow gpu
import tensorflow as tf
print(tf.__version__)

if tf.test.is_gpu_available():
    print("CUDA is available")
else:
    print("CUDA is not available")