#include<iostream>
#include<map>
#include<string>
#include<vector>
#include<algorithm>
#include<torch/script.h>
#include<opencv2/opencv.hpp>
#include<sys/time.h>

using namespace std;
using namespace cv;
using namespace torch;

class Utils {
public:
    Utils() {
        labelMap = {
            {0, "Animal"},
            {1, "Building"},
            {2, "Mountain"},
            {3, "Street"}
        };
    }

    string getLabel(int val) {
        return labelMap[val];
    }
private:
    unordered_map<int, string> labelMap;
};

class Image {
public:
    Image(string path) {
        auto mat = cv::imread(path);
        assert(!mat.empty());

        // Center crop the image to size (8000x8000)
        // Rect cropRegion((mat.cols - 8000) / 2, (mat.rows - 8000) / 2, 8000, 8000);
        // Mat centerCropped = mat(cropRegion);

        Mat resizedImage;
        resize(mat, resizedImage, Size(4000, 4000), 0, 0, INTER_CUBIC);

        vector<Mat> channels(3);
        split(resizedImage, channels);

        auto R = torch::from_blob(channels[2].ptr(), {4000, 4000}, torch::kUInt8);
        auto G = torch::from_blob(channels[1].ptr(), {4000, 4000}, torch::kUInt8);
        auto B = torch::from_blob(channels[0].ptr(), {4000, 4000}, torch::kUInt8);

        inputTensor = torch::cat({R, G, B}).view({3, 4000, 4000}).to(torch::kFloat);
        inputTensor /= 255.0;

        Tensor mean = torch::tensor({0.485, 0.456, 0.406});
        Tensor std = torch::tensor({0.229, 0.224, 0.225});

        inputTensor = normalize(inputTensor, mean, std);
        inputTensor = inputTensor.unsqueeze(0);
    }

    Tensor getImageTensor() {
        return inputTensor;
    }

private:
    Tensor inputTensor;
    Tensor normalize(Tensor image, Tensor mean, Tensor std) {
        return (image - mean.unsqueeze(-1).unsqueeze(-1)) / std.unsqueeze(-1).unsqueeze(-1);
    }

};

class TorchModel {
public:

    TorchModel(torch::jit::Module &module, string device) {
        this->module = module;
        this->utils = Utils();
        this->device = device;
    }

    string predict(Tensor inputImage) {
        vector<torch::jit::IValue> inputs;
        inputs.push_back(inputImage.to(torch::Device(device)));

        // Execute the model and turn its output into a tensor.
        at::Tensor output = module.forward(inputs).toTensor();
        // cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
        auto max_index = torch::argmax(output);
        // cout << max_index.item<int64_t>() << endl;
        return utils.getLabel(max_index.item<int64_t>());
    }

    void to(string device) {
        this->device = device;
        module.to(torch::Device(device));
    }

private:
    torch::jit::Module module;
    Utils utils;
    string device;
};

int main(int argc, char* argv[]) {
    
    string modelPath = "/scratch/ar7996/HPML/project/saved_model/torchscript_hct.pth";
    string inputPath = "/scratch/ar7996/HPML/project/dataset/train/Animal/pexels-photo-1300378_1.png";
    string device = "cuda";
    if(argc == 4) {
        modelPath = argv[1];
        inputPath = argv[2];
        device = argv[3];
    }

    c10::InferenceMode guard(true);
    torch::jit::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(modelPath);
        std::cerr << "Model Loaded" << endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model" << e.what() << endl;
        return -1;
    }
    module.to(torch::Device(device));

    TorchModel hct(module, device);
    Image inputImage(inputPath);

    //WarmUp
    string pred = hct.predict(inputImage.getImageTensor());

    double start, stop;
    struct timeval time;

    if (gettimeofday( &time, NULL ) < 0)
	    perror("start_timer,gettimeofday");

    start = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000);

    pred = hct.predict(inputImage.getImageTensor());

    if (gettimeofday( &time, NULL ) < 0)
	    perror("stop_timer,gettimeofday");

    stop = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000);

    string filePath = "/scratch/ar7996/HPML/project/cpp/CPP_logs_" + device + ".txt";

    ofstream outfile(filePath, ios::out);
    if (outfile.is_open()) {
        outfile << "Inference Time (" << device << ") per Image: " << stop - start << " secs" << endl;
        outfile << "Predicted Label: " << pred << endl;
        outfile.close(); // Close file
    } else {
        cout << "Unable to open file for writing." << endl;
        return 1;
    }

    return 0;
}
