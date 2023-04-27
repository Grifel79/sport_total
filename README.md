This is a repo with C++ app which takes an image as an input and gives it's class number in ImageNet dataset.

To build with docker:

////
docker build -t sport_total .
/////

To run with docker:

////
docker run -t sport_total
////



For now it uses the image specified in the end of the Dockerfile:
CMD ./sport_total_app ../snake.jpeg

Some example images are copied from src folder (cat1.jpeg, cat2.jpeg, cat3.jpeg, snake.jpeg)

main.py script is used to convert any PyTorch model to C++ lib torch format. Currently ResNet50 is used.

TODO:
Later will be implemented server version with input images being sent in POST requests.
