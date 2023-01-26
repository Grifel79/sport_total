import torch
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
import json

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet50()
        weights = torch.load("resnet50-11ad3fa6.pth")
        self.model.load_state_dict(weights)
        self.model.eval()
        self.transforms = ResNet50_Weights.DEFAULT.transforms()

    def forward(self, x):
        x = self.transforms(x)
        #x = x.unsqueeze(0)    # i already do it in C++ so no need to add a dimension here i guess
        return self.model(x)


img = read_image("./src/snake.jpeg")
#img = torch.ones(3, 500, 500)

# Step 1: Initialize model with the best available weights

# weights = torch.load("resnet50-11ad3fa6.pth")
#model.eval()
# Model class must be defined somewhere
# model = torch.load(PATH)
# model.eval()

weights = torch.load("resnet50-11ad3fa6.pth")
weights2 = ResNet50_Weights.DEFAULT # i need this to get classes and transforms to preprocess image
model = resnet50()
model.load_state_dict(weights)
model.eval()

# # Step 2: Initialize the inference transforms
preprocess = weights2.transforms()

# # Step 3: Apply inference preprocessing transforms
print(img.shape)
batch = preprocess(img).unsqueeze(0)
print(batch.shape)
#
# # Step 4: Use the model and print the predicted category

prediction = model(batch)

print(prediction.topk(5))
class_id = prediction.argmax().item()
print(class_id)
# score = prediction[class_id].item()

categories = weights2.meta["categories"]
categories_json = json.dumps(categories)
with open("ImageNet_categories.json", "w") as outfile:
    outfile.write(categories_json)

category_name = categories[class_id]
print(category_name)

my_module = MyModule()
output = my_module.forward(img.unsqueeze(0))
print("my_module.forward(img) = ", output.argmax().item())
sm = torch.jit.script(my_module)
sm.save("ResNet50_ImageNet.pt")

print('Python model top 5 results:\n  {}'.format(prediction.topk(5)))
print('TorchScript model top 5 results:\n  {}'.format(output.topk(5)))

# i don't use tracing because i want to include transforms into the C++ model
#traced_script_module = torch.jit.trace(model, batch)
#traced_script_module.save("resnet50-11ad3fa6-cpp.pt")

