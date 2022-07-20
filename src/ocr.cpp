# include "ocr.h"
# include <stdio.h>
void ocr::Model_Init(string det_engine_path, string det_onnx_path, string rec_engine_path, string rec_onnx_path){
    this->td = new TextDetect();
    this->tr = new TextRec();
    this->td->Model_Init(det_engine_path, det_onnx_path);
    this->tr->Model_Init(rec_engine_path, rec_onnx_path);
}

vector<pair< vector<string>, double>> ocr::Model_Infer(cv::Mat& inputImg, vector<double> & ocr_times){
  
    if(inputImg.channels()==4){
        cv::Mat _chs[4];
        split(inputImg, _chs);

        cv::Mat new_img(inputImg.rows, inputImg.cols, CV_8UC3);
        cv::Mat _new_chas[3];
        split(new_img, _new_chas);
        for(int i=0; i<3; i++){
            _new_chas[i] = 255 - _chs[3];
        }
        cv::Mat _dst;
        merge(_new_chas, 3, _dst);
        inputImg = _dst;
    }

    if(inputImg.channels()==1){
        cv::Mat new_img(inputImg.rows, inputImg.cols, CV_8UC3);
        cv::Mat _new_chas[3];
        split(new_img, _new_chas);
        for(int i=0; i<3; i++){
            _new_chas[i] = 255 - inputImg;
        }
        cv::Mat _dst;
        merge(_new_chas, 3, _dst);
        inputImg = _dst;
    }
    //cv::imshow(to_string(this->count_name_) + "_input_img", inputImg);
    //cv::waitKey(0);

    Mat srcImg;
    inputImg.copyTo(srcImg);

    vector<vector<vector<int>>> boxes;
    vector<double> det_times;
    this->td->Model_Infer(inputImg, boxes, det_times);

    vector<Mat> img_list;
    for (int j = 0; j < boxes.size(); j++) {
        cv::Mat crop_img;
        crop_img = Utility::GetRotateCropImage(srcImg, boxes[j]);
        img_list.push_back(crop_img);
        //cv::imwrite(to_string(this->count_name_)+"_crop_"+to_string(j)+".png", crop_img);
        // cv::imshow(to_string(this->count_name_)+"_crop_"+to_string(j)+".png", crop_img);
        // cv::waitKey(0);
    }
    //cout<<"finish detect"<<endl;

    vector<pair< vector<string>, double>> rec_res;
    vector<int> idx_map;
    vector<double> rec_times;
    this->tr->Model_Infer(img_list, rec_res, idx_map, rec_times);
    // 根据idx_map调整boxes的顺序， 并删除掉boxes中识别结果为空的box
    //cout<<"origin box size is "<< boxes.size()<<endl;
    vector<vector<vector<int>>> erase_nan_boxes;
    for(int i=0; i<idx_map.size(); i++){
        erase_nan_boxes.push_back(boxes[idx_map[i]]);
    }
    //cout<<"final box size is "<< erase_nan_boxes.size()<<endl;
    //cout<<"finish rect"<<endl;

    for(int i=0; i<3; i++){
        ocr_times.push_back(det_times[i]+rec_times[i]);
    }

    //// visualization
    string img_name = to_string(this->count_name_)+ ".png";
    if (this->visualize_) {
        Utility::VisualizeBboxes(inputImg, erase_nan_boxes, rec_res, img_name); // 名字可变
    }
    //cout<<"finish visual"<<endl;
    this->count_name_ ++;
    if(this->count_name_%100000 == 0)
        this->count_name_ = 0;

    return rec_res;
}

bool isNumber(const string& str){
    for(char const &c : str){
        if(isdigit(c)==0)
            return false;         
    }
    return true;
}

bool isChinese(const string& str)
{
  unsigned char utf[4] = {0};
  unsigned char unicode[3] = {0};
  bool res = false;
  for (int i = 0; i < str.length(); i++) {
    if ((str[i] & 0x80) == 0) {   //ascii begin with 0
      res = false;
    }
    else /*if ((str[i] & 0x80) == 1) */{
      utf[0] = str[i];
      utf[1] = str[i + 1];
      utf[2] = str[i + 2];
      i++;
      i++;
      unicode[0] = ((utf[0] & 0x0F) << 4) | ((utf[1] & 0x3C) >>2);
      unicode[1] = ((utf[1] & 0x03) << 6) | (utf[2] & 0x3F);
//      printf("%x,%x\n",unicode[0],unicode[1]);
//      printf("aaaa %x,%x,%x\n\n",utf[0],utf[1],utf[2]);
      if(unicode[0] >= 0x4e && unicode[0] <= 0x9f){
         if (unicode[0] == 0x9f && unicode[1] >0xa5)
                res = false;
         else         
               res = true;
      }else
         res = false;
    }
  }
  return res;
}


string ocr::TaskProcess(vector<pair< vector<string>, double>> &result){
    for(int i=0; i<result.size(); i++){
        if((result[i].second > REC_THR_)&&(result[i].first.size()>=3)){ // 置信度较高
            vector<string> res_vec;
            for(int j=0; j<3; j++){
                if(!isChinese(result[i].first[j])){ // 不是汉字
                    if(isNumber(result[i].first[j].c_str())) // 是数字
                        res_vec.push_back(result[i].first[j]);
                }
            }    
            string res="";
            if(res_vec.size()==3){
                for(int j=0; j<3; j++)
                    res += res_vec[j];
                if(stoi(res.c_str())>= REC_RANGE_[0] && stoi(res.c_str())<= REC_RANGE_[1])
                    return res;
            }
        }
    }
    return "";
}

string ocr::MultiFrameSmooth(string door_result, int step){
    count_img_ ++ ;
    results_[door_result]++;
    if(door_result == "")
        results_[""] = 1; //当有门牌时，会正常返回门牌号，一直没有门牌则会返回“”

    if(count_img_ == step){
        int max_value = 0;
        string res= "";
        for(auto it = results_.begin(); it!= results_.end(); it++){
            if(it->second > max_value){
                max_value = it->second;
                res = it-> first;
            }
        } 
        cout<< res << " : " << max_value << endl; 
        results_.clear();
        count_img_ = 0;
        return res;
    }
    return "";
}

ocr::~ocr(){
    delete td;
    delete tr;   
}
