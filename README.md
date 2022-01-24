# trajectories prediction

This project aims for predicting trajectories of surrounding pedestrians from first-person perspective.

## Human Detection
When getting human's feet coordinates, I used pretrained model of yolov3 and regard the center of two bottom corner of predicted bounding box as that person's feet coordinate.

## Coordinate Transformation
I used ARkit, which enables us to get information about internal and external camera matrix. According to these matrices, I did coordinate transformation of the detected person's feet coordinate from screen coordinate to world coordinate, which is compatible with the dataset of trajectory prediction.

## Trajectory Prediction
I used socialGAN as model to predict future trajectory of 3.2s for each person.

## Semantic Segmentation
I used pretrained model of neural networks, consisting of resnet101dilated as encorder and upernet as decorder, in order to integrate the terrain information with SGAN.