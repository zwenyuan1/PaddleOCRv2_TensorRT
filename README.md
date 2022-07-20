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
 cd build
 cmake ..
 make -j
 ./rec
```
