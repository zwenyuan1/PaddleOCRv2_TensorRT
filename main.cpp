# include "ocr.h"
# include "parameterReader.h"

# include <dirent.h>
# include <sys/stat.h>
# include <sys/types.h>

using namespace OCR;
int camera_rec(){
    ParameterReader pr("/home/ubt/workspace/TensorRT/OCR/parameters.txt");

    string det_onnx_path = pr.getData("det_onnx_path");
    string det_engine_path = pr.getData("det_engine_path");

    string rec_onnx_path = pr.getData("rec_onnx_path");
    string rec_engine_path = pr.getData("rec_engine_path");

    ocr * ocr_test = new ocr();
    ocr_test->Model_Init(det_engine_path, det_onnx_path, rec_engine_path, rec_onnx_path);

    VideoCapture cap(-1);
    Size size1 = Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT));
    int myFourCC = VideoWriter::fourcc('m','p','4','v');
    Mat test_img;
    while(true){
        if(!cap.read(test_img)){
            std::cout <<"open camera is error!"<<std::endl;
            return -1;
        }
        vector<pair< vector<string>, double>> result;
        vector<double> ocr_times;
        result = ocr_test->Model_Infer(test_img, ocr_times);
        cout<< "per image inference time = "<< ocr_times[1]<<"ms" <<endl;
        cout<<"---------------------------------"<<endl;
    }
    delete ocr_test;
    return 0;
    
}

int imgs_rec(){
    ParameterReader pr("/home/ubt/workspace/TensorRT/OCR/parameters.txt");
    string img_dir = pr.getData("img_dir");

    string det_onnx_path = pr.getData("det_onnx_path");
    string det_engine_path = pr.getData("det_engine_path");

    string rec_onnx_path = pr.getData("rec_onnx_path");
    string rec_engine_path = pr.getData("rec_engine_path");
    
    ocr * ocr_test = new ocr();
    ocr_test->Model_Init(det_engine_path, det_onnx_path, rec_engine_path, rec_onnx_path);

    vector<String> all_img_names;
    cv::glob(img_dir, all_img_names);

    for(int i=0; i<all_img_names.size(); i++){
        cout << all_img_names[i] << endl;

        cv::Mat test_img = cv::imread(all_img_names[i], -1);

        vector<pair< vector<string>, double>> result;
        vector<double> ocr_times;
        result = ocr_test->Model_Infer(test_img, ocr_times);
        //string res = ocr_test-> TaskProcess(result);

        //int step = 20;
        //string door_res = ocr_test-> MultiFrameSmooth(res, step);

        //cout<< "per image preprocess time = "<< ocr_times[0]<<"ms"<<endl;
        cout<< "per image inference time = "<< ocr_times[1]<<"ms" <<endl;
        //cout<< "per image postprocess time = "<< ocr_times[2] <<"ms"<<endl;
        cout<<"---------------------------------"<<endl;

        //if((i+1)%step==0){
        //    cout<<"step=20 : "<< door_res <<endl;
        //}
        
        //cv::putText(test_img, door_res, cv::Point(10, 25), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0, 0, 255), 2);
        //cv::imwrite(all_img_names[i], test_img);
        
    }
    delete ocr_test;
    return 0;

}

int main(){
    imgs_rec();
    //camera_rec();
    cout<<" finish ocr !! "<<endl;
    return 0;
}