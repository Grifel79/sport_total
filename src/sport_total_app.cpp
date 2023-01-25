#include <torch/script.h> 
#include <torch/nn/functional/activation.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <memory>

using namespace cv;

torch::Tensor to_tensor(cv::Mat img) {
    return torch::from_blob(img.data, { img.rows, img.cols, 3 }, torch::kUInt8);
}

int main(int argc, const char* argv[]) {
  if (argc != 2) {
      std::cerr << "usage: SportTotalApp <path-to-input-image>\n";
      return -1;
    }
  
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
  std::cout << output.argmax().item() << "\n";

  // namespace F = torch::nn::functional;
  // at::Tensor output_sm = F::softmax(output, F::SoftmaxFuncOptions(1));
  // std::tuple<at::Tensor, at::Tensor> top5_tensor = output_sm.topk(5);
  // at::Tensor top5 = std::get<1>(top5_tensor);

  // std::cout << top5[0] << "\n";

}