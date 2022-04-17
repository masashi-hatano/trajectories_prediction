import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# human detection
def get_foot_coordinates(path_image):
    image = plt.imread(path_image)
    #image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    res = cv2.resize(image, dsize=(416, 416), interpolation=cv2.INTER_CUBIC)

    classes = []
    with open('yolov3/coco.names', 'r') as f:
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
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
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

    plt.imshow(result)
    plt.show()

    return coordinates, image


# coordinate transformation from fpp to bird's eye
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
    fx = df['fx'].iloc[0]
    ox = df['ox'].iloc[0]
    fy = df['fy'].iloc[0]
    oy = df['oy'].iloc[0]
    X = df['t0'].iloc[0]
    Y = df['t1'].iloc[0]
    Z = df['t2'].iloc[0]

    R = np.array([[r00,r01,r02],[r10,r11,r12],[r20,r21,r22]])
    K = np.array([[fx,0,ox],[0,fy,oy],[0,0,1]])
    T = np.array([X,Y,Z]).reshape(3,1)
    return R, K, T

def screenToCamera(coordinates, image, K):
    u = coordinates[0]
    v = coordinates[1]
    ox = K[0,2]
    oy = K[1,2]
    fx = K[0,0]
    fy = K[1,1]
    
    x_camera = (v-ox)/fx
    y_camera = (u-(image.shape[1]-oy))/fy
    z_camera = -1
    
    temporary_coordinate_camera = np.array([x_camera, y_camera, z_camera]).reshape(3,1)
    return temporary_coordinate_camera

def cameraToWorld(temporary_coordinate_camera, R, T):
    temporary_coordinate_world = np.dot(R, temporary_coordinate_camera) + T
    direction = temporary_coordinate_world-T
    return temporary_coordinate_world, direction

def calculateRealCoordinate(T, direction, h):
    t = (h-T[1][0])/direction[1][0]
    x_real = T[0][0] + direction[0][0]*t
    z_real = T[2][0] + direction[2][0]*t
    real_coordinate = np.array([x_real,h,z_real]).reshape(3,1)
    return real_coordinate

def createDataText(path, data):
    with open(path, 'w') as f:
        for i in range(len(data)):
            f.write(data[i][0]+'\t'
            +data[i][1]+'\t'
            +data[i][2]+'\t'
            +data[i][3]+'\n')


# coordinate transformation from bird's eye to fpp
def worldToCamera(real_coordinate, R, T):
    real_coordinate_camera = np.dot(R.T, (real_coordinate-T))
    return real_coordinate_camera

def cameraToScreen(real_coordinate_camera, image, K):
    ox = K[0,2]
    oy = K[1,2]
    fx = K[0,0]
    fy = K[1,1]

    u = (fy*real_coordinate_camera[1][0]/abs(real_coordinate_camera[2][0]) + (image.shape[1]-oy))
    v = (fx*real_coordinate_camera[0][0]/abs(real_coordinate_camera[2][0]) + ox)

    coordinates = (int(u),int(v))
    return coordinates

def plot(image, coordinates, path):
    for i in range(len(coordinates)):
        cv2.circle(image, coordinates[i], 5, (255,0,0), -1)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image_bgr)


# boundary extraction
def extract_coordinates(R,K,T,mask_image):
    coordinates=[]
    counter=100
    for delta_z in range(16):
        pre_color = None
        for delta_x in range(-500,501):
            real_coordinate = np.array([T[0][0]+0.01*delta_x,-1.35,T[2][0]-1*delta_z]).reshape(3,1)
            #print(real_coordinate)
            real_coordinate_camera = worldToCamera(real_coordinate, R, T)
            u,v = cameraToScreen(real_coordinate_camera, mask_image, K)
            #print([u,v])
            if u<0 or u>=1440 or v<0 or v>=1920:
                pass
            else:
                rgb = mask_image[v][u]
                if rgb[0]==128 and rgb[1]==63 and rgb[2]==127:
                    color = "violet"
                else:
                    color = "black"

                if counter==100:
                    if pre_color==None:
                        pre_color = color
                    elif pre_color != color:
                        coordinates.append([u,v])
                        pre_color = color
                        counter=0
                else:
                    counter+=1
    return coordinates


# timestamp generation
def filename_list(directory):
    imgs = os.listdir(directory)
    for i, filename in enumerate(imgs):
        imgs[i] = filename.replace('.jpg','')
    imgs.sort(key=int)
    return imgs

def extract_imgs(imgs):
    imgs_extracted = []
    imgs_extracted.append(imgs[0])
    time_next = int(imgs[0])+400
    for i in range(2,len(imgs)):
        diff_1 = abs(time_next-int(imgs[i-1]))
        diff_2 = abs(time_next-int(imgs[i]))
        if diff_1 < diff_2:
            imgs_extracted.append(imgs[i-1])
            time_next = int(imgs[i-1])+400
    return imgs_extracted

def create_timestamp(path, imgs_extracted):
    with open(path, 'w') as f:
        for i in range(len(imgs_extracted)):
            f.write(imgs_extracted[i]+'\n')

