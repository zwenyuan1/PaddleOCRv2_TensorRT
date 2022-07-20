# include "Convert.h"

ICudaEngine* Convert::createCudaEngine(string& onnxModelPath){
    /*/ input: onnxModelPath  /*/

    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH); 
    unique_ptr<nvinfer1::IBuilder, Destroy<nvinfer1::IBuilder>> builder{nvinfer1::createInferBuilder(gLogger)};
    unique_ptr<nvinfer1::INetworkDefinition, Destroy<nvinfer1::INetworkDefinition>> network{builder->createNetworkV2(explicitBatch)};
    unique_ptr<nvonnxparser::IParser, Destroy<nvonnxparser::IParser>> parser{nvonnxparser::createParser(*network, gLogger)};
    unique_ptr<nvinfer1::IBuilderConfig,Destroy<nvinfer1::IBuilderConfig>> config{builder->createBuilderConfig()};

    if (!parser->parseFromFile(onnxModelPath.c_str(), static_cast<int>(ILogger::Severity::kINFO)))
    {
        cout << "ERROR: could not parse input engine." << endl;
        return nullptr;
    }

    cout<<2*(1ULL<<31)<<endl;
    config->setMaxWorkspaceSize(2*(1ULL<<31));
    
    //config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 32);
    builder->setFp16Mode(builder->platformHasFastFp16());
    builder->setMaxBatchSize(maxBatchSize);
    
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMIN, Dims4{MIN_DIMS_[0], MIN_DIMS_[1], MIN_DIMS_[2], MIN_DIMS_[3]});
    profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kOPT, Dims4{OPT_DIMS_[0], OPT_DIMS_[1], OPT_DIMS_[2], OPT_DIMS_[3]});
    profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMAX, Dims4{MAX_DIMS_[0], MAX_DIMS_[1], MAX_DIMS_[2], MAX_DIMS_[3]});    
    config->addOptimizationProfile(profile);

    return builder->buildEngineWithConfig(*network, *config);   

}

bool Convert::saveEngine(ICudaEngine* s_engine, string& engineName){

    assert(s_engine != nullptr);
    // Serialize the engine
    IHostMemory* modelStream{ nullptr };
    modelStream = s_engine->serialize();
    assert(modelStream != nullptr);

    cout<<"finish create modelstram"<<endl;

    std::ofstream fout(engineName.c_str(), ios::out | std::ios::binary);
    if (!fout) {
        std::cerr << "could not open engine output file, saveEngine failed!" << std::endl;
        return false;
    }
    fout.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    modelStream->destroy();
    return true;

}


bool Convert::getEngine(string& engineName, string onnx_path){

    if(access(engineName.c_str(), F_OK ) == -1){  // 如果该engine文件不存在

        cout << "engine file is not exist, need to be created" << endl;

        if(access(onnx_path.c_str(), F_OK ) == -1){ // 如果onnx文件不存在
            cout << "onnx path is not exist, can't create a engine from it "<<endl;
            return false;
        }
        else{
            if(saveEngine(createCudaEngine(onnx_path), engineName))
                cout<<"sucessful from onnx create "<< engineName <<endl;
            else
                return false;
        }       
    }
    else
       cout<< engineName <<" already exist"<<endl; 
    return true;
}

bool Convert::deserializeEngine(string& engineName){
    std::ifstream file(engineName, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engineName << " error!" << std::endl;
        return false;
    }
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);

    delete[] trtModelStream;
    return true;
}

Convert::~Convert(){
    // destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
}