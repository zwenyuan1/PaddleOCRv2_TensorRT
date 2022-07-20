# include "Convert.h"
# include "postprocess_op.h"
# include "preprocess_op.h"
# include <opencv2/opencv.hpp>

using namespace cv;

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

namespace OCR {

class TextDetect: public Convert{
public:
    TextDetect():Convert(){};
    int Model_Init(string& engine_path, string onnx_path=" ");
    void Model_Infer(cv::Mat& Input_Image, vector<vector<vector<int>>> &boxes, vector<double> &times);
    ~TextDetect();

private:
    //config
    
    //task
    double det_db_thresh_ = 0.3;
    double det_db_box_thresh_ = 0.5;
    double det_db_unclip_ratio_ = 2.0;
    bool use_polygon_score_ = false;

    // input/output layer 
    const char *INPUT_BLOB_NAME = "x";
    const char *OUTPUT_BLOB_NAME = "save_infer_model/scale_0.tmp_1";

    // input image
    int max_side_len_ = 640;
    ResizeImgType0 resize_op_;

    vector<float> mean_ = {0.485f, 0.456f, 0.406f};
    vector<float> scale_ = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f};
    Normalize normalize_op_;

    Permute permute_op_;

    // output result
    PostProcessor post_processor_;
  
};

}// namespace OCR

