import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import tan

def get_foot_coordinates(path_image):
    image = plt.imread(path_image)
    res = cv2.resize(image, dsize=(416, 416), interpolation=cv2.INTER_CUBIC)

    classes = []
    with open('coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]


    Width = res.shape[1]
    Height = res.shape[0]

    # read pre-trained model and config file
    net = cv2.dnn.readNet('yolov3/yolov3.weights', 'yolov3/yolov3.cfg')

    # create input blob 
    # set input blob for the network
    net.setInput(cv2.dnn.blobFromImage(res, 0.00392, (416,416), (0,0,0), True, crop=False))

    # run inference through the network
    # and gather predictions from output layers
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)


    class_ids = []
    confidences = []
    boxes = []

    # create bounding box 
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])


    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)

    coordinates = []
    # get detected person's foot coordinate
    for i in indices:
        i = i[0]
        box = boxes[i]
        if class_ids[i]==0:
            label = str(classes[class_id])
            coordinate_x = round((box[0]+box[2]/2)*image.shape[1]/416)
            coordinate_y = round((box[1]+box[3])*image.shape[0]/416)
            coordinates.append((coordinate_x,coordinate_y))
            cv2.rectangle(res, (round(box[0]),round(box[1])), (round(box[0]+box[2]),round(box[1]+box[3])), (255, 0, 0), 1)
            cv2.putText(res, label, (round(box[0])-10,round(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # resize resized image to the original scale
    result = cv2.resize(res, dsize=(image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)

    #plt.imshow(result)
    #plt.show()

    return coordinates, image

def screenToCamera(pixel_x, pixel_y, image):
    f = 1.0/(2.0* tan(np.radians(120)/2.0))*image.shape[1]
    C_u = image.shape[1]/2
    C_v = image.shape[0]/2
    x_camera = (pixel_x-C_u)/f
    y_camera = (pixel_y-C_v)/f
    z_camera = 1
    temporary_coordinate_camera = np.array([x_camera, y_camera, z_camera])
    return temporary_coordinate_camera

def cameraToWorld(temporary_coordinate_camera, R, X, Y, Z):
    coordinate_camera = np.array([X,Y,Z]).reshape(3,1)
    temporary_coordinate_world = np.dot(R, (temporary_coordinate_camera).reshape(3,1)) + coordinate_camera
    direction = temporary_coordinate_world-coordinate_camera
    return temporary_coordinate_world, direction

def get_data_from_csv(path_csv, time):
    df = pd.read_csv(path_csv)
    df = df[df.time == time]
    r00 = df['r00'].iloc[0]
    r01 = df['r01'].iloc[0]
    r02 = df['r02'].iloc[0]
    r10 = df['r10'].iloc[0]
    r11 = df['r11'].iloc[0]
    r12 = df['r12'].iloc[0]
    r20 = df['r20'].iloc[0]
    r21 = df['r21'].iloc[0]
    r22 = df['r22'].iloc[0]

    R = np.array([[r00,r01,r02],[r10,r11,r12],[r20,r21,r22]])
    X = df['t0'].iloc[0]
    Y = df['t1'].iloc[0]
    Z = df['t2'].iloc[0]
    return R, X, Y, Z

def calculateRealCoordinate(X, Y, Z, direction, h):
    t = (h-Y)/direction[1][0]
    x_real = X + direction[0][0]*t
    z_real = Z + direction[2][0]*t
    return x_real, z_real


time = []
with open('timestamp.txt') as f:
    for line in f:
        time.append(line.strip())

data = []
for i in range(len(time)):
    coordinates, image = get_foot_coordinates('images/'+time[i]+'.jpg')
    for j in range(len(coordinates)):
        temporary_coordinate_camera = screenToCamera(coordinates[0][0], coordinates[0][1], image)
        print(int(time[i]))
        R, X, Y, Z = get_data_from_csv('1125_1537_36.csv', int(time[i]))
        temporary_coordinate_world, direction = cameraToWorld(temporary_coordinate_camera, R, X, Y, Z)
        x_real, z_real = calculateRealCoordinate(X, Y, Z, direction, -1.2)
        print(x_real)
        print(z_real)
        data.append([time[i], str(j+1), str(x_real), str(z_real)])

with open('datasets/original/scean1/data.txt', 'w') as f:
    for i in range(len(data)):
        f.write(data[i][0]+'\t'
        +data[i][1]+'\t'
        +data[i][2]+'\t'
        +data[i][3]+'\n')