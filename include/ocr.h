# include "det.h"
# include "rec.h"

using namespace OCR;

class ocr{
public:
    ocr(){};
    void Model_Init(string det_engine_path, string det_onnx_path, string rec_engine_path, string rec_onnx_path);
    vector<pair< vector<string>, double>> Model_Infer(cv::Mat& inputImg, vector<double> & ocr_times);
    string TaskProcess(vector<pair< vector<string>, double>> &result);
    string MultiFrameSmooth(string door_result, int step);
    ~ocr();
private:

    TextDetect * td = NULL;
    TextRec * tr = NULL;

    // tast
    vector<int> REC_RANGE_ = {400, 599};
    float REC_THR_ = 0.85;

    //MultiFrameSmooth
    int count_img_ = 0;
    unordered_map<string, int> results_;

    bool visualize_= true;
    int count_name_ = 0;
};