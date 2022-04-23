# trajectories prediction

This project aims for predicting trajectories of surrounding pedestrians from first-person perspective.

## Introduction
Predicting trajectory of pedestrians is one of the key factors for a better society such as automatic driving, guide for blind people, and social robots interacting with human. This task has been done by following existing methods, but these are not good enough.

- Predicting from bird's-eye coordinate:
Models using bird's-eye coordinate are, in fact, impractical in terms of ubiquitousness and availability of surveillance camera, which provides bird's-eye images.
- Predicting from first-person perspective:
Models predicting trajectories from first-person perspective need a variety of information, and thus are complex, overall. Since the objective is to be applied not only to intelligent driving, but also guidance for the visually impaired and social robots, it is obvious that a simpler model is better.

The advantages of proposal method are as follows:
- It is practical as trajectories are predicted from first-person perspective.
- Existing models using bird's-eye coordinate can be used because of high accuracy coordinate transformation from FPP to bird's-eye coordinate, based on the SLAM technology, provided by ARKit.
- The model is simple since the input data is the same as the existing model predicting from bird's-eye.
- A revolutional transformation that compresses and transforms the terrain information obtained from the first-person viewpoint image into the same shape as the inputs of the existing model(using bird's eye) is done.

Following techniques are being used in this project.
### Human Detection
When getting human's feet coordinates, I used pretrained model of yolov3 and regard the center of two bottom corner of predicted bounding box as that person's feet coordinate.

### Coordinate Transformation
I used ARkit, which enables us to get information about internal and external camera matrix. According to these matrices, I did coordinate transformation of the detected person's feet coordinate from screen coordinate to world coordinate, which is compatible with the dataset of trajectory prediction.

### Prediction model
I used socialGAN as model to predict future trajectory of 3.2s for each person.

### Semantic Segmentation
I used pretrained model of SegFormer, consisting of transformer as encorder and MLP as decorder, in order to integrate the terrain information with SGAN.
