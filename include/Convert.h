// create by zwy 2021,12,10
#pragma once

# include "NvInfer.h"
# include "logging.h"
# include "cuda_runtime_api.h"
# include "NvOnnxParser.h"

# include <iostream>
# include <fstream>
# include <unistd.h>
# include <string>

using namespace nvinfer1;
using namespace std;

template <typename T>
struct Destroy
{
    void operator()(T* t) const
    {
        t->destroy();
    }
};

// Allow TensorRT to use up to 1GB of GPU memory for tactic selection.
constexpr size_t MAX_WORKSPACE_SIZE = 1<<31; // 30 =1 GB 1ULL << 34
const int maxBatchSize = 1;

class Convert{
public:
    Convert(){};
    Convert(vector<int> MIN_DIMS, vector<int> OPT_DIMS, vector<int> MAX_DIMS){
        this->MIN_DIMS_ = MIN_DIMS;
        this->OPT_DIMS_ = OPT_DIMS;
        this->MAX_DIMS_ = MAX_DIMS;
    };

    // 通过onnx来创建engine
    ICudaEngine* createCudaEngine(string& onnx_path); /*/ input: onnxModelPath  /*/

    // ** 输入engine名，如果不存在，则通过createCudeEngine(onnx_path)来创建，并写成文件。
    bool getEngine(string& engine_path, string onnx_path=" "); 

    // 将engine保存到engine_path中去
    bool saveEngine(ICudaEngine* s_engine, string& engine_path); 

    // ** 解engine 
    bool deserializeEngine(string& engine_path);
    ~Convert();

    Logger gLogger; //实例化了一个类
    IRuntime *runtime = NULL;
    ICudaEngine *engine = NULL;
    IExecutionContext *context = NULL;

private:
    vector<int> MIN_DIMS_ = {1, 3, 20, 20};
    vector<int> OPT_DIMS_ = {1, 3, 256, 256};
    vector<int> MAX_DIMS_ = {1, 3, 640, 640};
};