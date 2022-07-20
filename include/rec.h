# include "Convert.h"
# include "postprocess_op.h"
# include "preprocess_op.h"
# include <opencv2/opencv.hpp>
# include "utility.h"

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

class TextRec: public Convert{
public:
    TextRec():Convert({1, 3, 32, 10}, {1, 3, 32, 320}, {8, 3, 32, 2000}){
        this->label_list_ = Utility::ReadDict(this->label_path);
        this->label_list_.insert(this->label_list_.begin(), "#"); // blank char for ctc
        this->label_list_.push_back(" ");
    };
    int Model_Init(string& engine_path, string onnx_path=" ");
    void Model_Infer(vector<cv::Mat> Input_Image, vector<pair< vector<string>, double> > &rec_res, vector<int>& idx_map, vector<double> &times);
    ~TextRec();

private:
    //task
    std::vector<std::string> label_list_;
    string label_path = "../models/txt/ppocr_keys_v1.txt";

    // input/output layer 
    const char *INPUT_BLOB_NAME = "x";
    const char *OUTPUT_BLOB_NAME = "save_infer_model/scale_0.tmp_1";

    // input image
    int rec_batch_num_=6;
    CrnnResizeImg resize_op_;

    std::vector<float> mean_ = {0.5f, 0.5f, 0.5f};
    std::vector<float> scale_ = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};
    Normalize normalize_op_;

    PermuteBatch permute_op_;

    // output result
    PostProcessor post_processor_;
  
};

}// namespace OCR