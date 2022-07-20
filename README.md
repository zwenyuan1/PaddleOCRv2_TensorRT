# PaddleOCRv2_TensorRT
## Enviroment
* cuda 11.1
* cudnn 11.1
* tensorrt-cuda11.1-trt7.2.1.6
* opencv 3.4.8

## How to run
* modify the cuda, cudnn, tensorrt path in **CmakeLists.txt**
* modify the path parameters.txt in **main.cpp**
* modify the img_dir in **parameters.txt**
```
 mkdir build
 cd build
 cmake ..
 make -j
 ./rec
```
## Result
![image](https://user-images.githubusercontent.com/87298337/179940172-4182773c-5786-4d5e-a1e1-8a63e98b4f10.png)

