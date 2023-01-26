#include <torch/script.h> 
#include <torch/nn/functional/activation.h>
#include "json.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include <fstream>
#include <iostream>
#include <memory>
using json = nlohmann::json;
using namespace cv;

int main(int argc, const char* argv[]) {
  if (argc != 2) {
      std::cerr << "usage: SportTotalApp <path-to-input-image>\n";
      return -1;
    }
  
  std::ifstream f("../ImageNet_categories.json");
  json imagenet_class = json::parse(f);
  
  // Reading image and converting BGR 2 RGB
  // std::string image_path = samples::findFile("/Users/gchernomordik/Documents/dev/SportTotal/cat2.jpeg");
  Mat img = imread(argv[1], IMREAD_COLOR);
  cv::cvtColor(img, img, COLOR_BGR2RGB);

  // converting opencv image to Tensor

  at::Tensor tensor_image = torch::from_blob(img.data, {img.rows, img.cols, 3}, at::kByte);
  tensor_image = tensor_image.permute({2, 0, 1});
  tensor_image.unsqueeze_(0);

  tensor_image = tensor_image.to(at::kFloat).div(255);;

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(tensor_image);

  // imshow("Display window", img);
  // int k = waitKey(0); // Wait for a key stroke in the window

  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load("../ResNet50_ImageNet.pt");
  } 
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }
    
  // Execute the model and turn its output into a tensor.
  at::Tensor output = module.forward(inputs).toTensor();
  int class_id = output.argmax().item().toInt();
  std::cout << imagenet_class[class_id] << "\n";

}