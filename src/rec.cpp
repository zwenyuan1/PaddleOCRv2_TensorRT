# include "rec.h"

namespace OCR{

int TextRec::Model_Init(string& engine_path, string onnx_path){
    if(getEngine(engine_path, onnx_path)){
        if(deserializeEngine(engine_path))
            cout<< "Sucessful deserialize engine file!" << endl;
        else{
            cout<<"deserialize engine failed, model init failed"<<endl;
            return -2;
        }
    }
    else{
        cout<<"can't get engine file, model init failed"<<endl;
        return -1;
    }

    assert(engine->getNbBindings() == 2); //check if is a input and a output

    return 0;
}

void TextRec::Model_Infer(vector<cv::Mat> img_list, vector<pair< vector<string>, double> > &rec_res, vector<int>& idx_map, vector<double> &times){
    std::chrono::duration<float> preprocess_diff = std::chrono::steady_clock::now() - std::chrono::steady_clock::now();
    std::chrono::duration<float> inference_diff = std::chrono::steady_clock::now() - std::chrono::steady_clock::now();
    std::chrono::duration<float> postprocess_diff = std::chrono::steady_clock::now() - std::chrono::steady_clock::now();
    
    int img_num = img_list.size();
    std::vector<float> width_list; //存储所有待识别图像的宽高比
    for (int i = 0; i < img_num; i++) 
        width_list.push_back(float(img_list[i].cols) / img_list[i].rows); 
    
    std::vector<int> indices = Utility::argsort(width_list);//对宽高比由小到大进行排序，并获取indices
    std::vector<int> copy_indices = indices;
    // 记录一个batch里识别结果为空的idx
    vector<int> nan_idx;

    for(int begin_img = 0; begin_img < img_num; begin_img += this->rec_batch_num_){

        /////////////////////////// preprocess ///////////////////////////////
        auto preprocess_start = std::chrono::steady_clock::now();
        int end_img = min (img_num, begin_img + this->rec_batch_num_);
        float max_wh_ratio = 0;
        for (int ino = begin_img; ino < end_img; ino ++) {
            int h = img_list[indices[ino]].rows;
            int w = img_list[indices[ino]].cols;
            float wh_ratio = w * 1.0 / h;
            max_wh_ratio = max(max_wh_ratio, wh_ratio);
        } //找最大的宽高比

        std::vector<cv::Mat> norm_img_batch;
        for (int ino = begin_img; ino < end_img; ino ++) {
            cv::Mat srcimg;
            img_list[indices[ino]].copyTo(srcimg);
            cv::Mat resize_img;
            this->resize_op_.Run(srcimg, resize_img, max_wh_ratio);
            this->normalize_op_.Run(&resize_img, this->mean_, this->scale_, true);
            norm_img_batch.push_back(resize_img);
        } //将一个batch里的img按照最大宽高比resize到高为32，宽为32*max_wh_ratio，并分别做归一化。

        auto preprocess_end = std::chrono::steady_clock::now();
        preprocess_diff += preprocess_end - preprocess_start;
      
        ////////////////////////// inference /////////////////////////
        void* buffers[2];

        // 为buffer[0]指针（输入）定义空间大小
        //int batch_width = 1999;
        int batch_width = int(32 * max_wh_ratio); // 这个batch里图像的宽度
        float *inBlob = new float[norm_img_batch.size() * 3 * 32 * batch_width];
        this->permute_op_.Run(norm_img_batch, inBlob);

        int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
        CHECK(cudaMalloc(&buffers[inputIndex], norm_img_batch.size() * 3 * 32 * batch_width * sizeof(float)));

        auto inference_start = std::chrono::steady_clock::now();
        // Create stream
        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));
        // 将数据放到gpu上
        CHECK(cudaMemcpyAsync(buffers[inputIndex], inBlob, norm_img_batch.size() * 3 * 32 * batch_width * sizeof(float), cudaMemcpyHostToDevice, stream));

        //#### 将输入图像的大小写入context中 #######
        context->setOptimizationProfile(0); // 让convert.h创建engine的动态输入配置生效
        auto in_dims = context->getBindingDimensions(inputIndex); //获取带有可变维度的输入维度信息
        in_dims.d[0]=norm_img_batch.size();
        in_dims.d[1]=3;
        in_dims.d[2]=32;
        in_dims.d[3]=batch_width;
        
        context->setBindingDimensions(inputIndex, in_dims); // 根据输入图像大小更新输入维度

        // 为buffer[1]指针（输出）定义空间大小
        int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
        auto out_dims = context->getBindingDimensions(outputIndex);

        int output_size=1;
        for(int j=0; j<out_dims.nbDims; j++) 
            output_size *= out_dims.d[j];

        float *outBlob = new float[output_size];
        CHECK(cudaMalloc(&buffers[outputIndex], output_size*sizeof(float)));

        // 做推理
        context->enqueue(1, buffers, stream, nullptr);
        // 从gpu取数据到cpu上
        CHECK(cudaMemcpyAsync(outBlob, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
        // Release stream and buffers
        cudaStreamDestroy(stream);
        CHECK(cudaFree(buffers[inputIndex]));
        CHECK(cudaFree(buffers[outputIndex]));

        auto inference_end = std::chrono::steady_clock::now();
        inference_diff += inference_end - inference_start;
      
        ////////////////////// postprocess ///////////////////////////
        auto postprocess_start = std::chrono::steady_clock::now();

        vector<int> predict_shape;
        for(int j=0; j<out_dims.nbDims; j++) 
            predict_shape.push_back(out_dims.d[j]);
        
        for (int m = 0; m < predict_shape[0]; m++) { // m = batch_size
            pair<vector<string>, double> temp_box_res;
            std::vector<std::string> str_res;
            int argmax_idx;
            int last_index = 0;
            float score = 0.f;
            int count = 0;
            float max_value = 0.0f;

            for (int n = 0; n < predict_shape[1]; n++) { // n = 2*l + 1
                argmax_idx =
                    int(Utility::argmax(&outBlob[(m * predict_shape[1] + n) * predict_shape[2]],
                                        &outBlob[(m * predict_shape[1] + n + 1) * predict_shape[2]]));
                max_value =
                    float(*std::max_element(&outBlob[(m * predict_shape[1] + n) * predict_shape[2]],
                                            &outBlob[(m * predict_shape[1] + n + 1) * predict_shape[2]]));

                if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
                    score += max_value;
                    count += 1;
                    str_res.push_back(this->label_list_[argmax_idx]);
                }
                last_index = argmax_idx;
            }
            score /= count;
            if (isnan(score)){
                nan_idx.push_back(begin_img+m);
                continue;
            }   
            //for (int i = 0; i < str_res.size(); i++) {
            //    std::cout<< str_res[i];
            //}
            //std::cout << "\tscore: " << score << std::endl;
            temp_box_res.first=str_res;
            temp_box_res.second=score;
            rec_res.push_back(temp_box_res);
        }

        delete [] inBlob;
        delete [] outBlob;
        auto postprocess_end = std::chrono::steady_clock::now();
        postprocess_diff += postprocess_end - postprocess_start;
    }

    for(int i=nan_idx.size()-1; i>=0; i--){
        copy_indices.erase(copy_indices.begin()+ nan_idx[i]);
    }

    if(copy_indices.size()==rec_res.size()){
        //cout<<"rec res size is equal to indices size"<<endl;
        idx_map = copy_indices;
    }

    times.push_back(double(preprocess_diff.count() * 1000));
    times.push_back(double(inference_diff.count() * 1000));
    times.push_back(double(postprocess_diff.count() * 1000));

}
TextRec::~TextRec(){

}

} // namespace OCR