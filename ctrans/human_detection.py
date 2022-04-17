import cv2
import numpy as np
import matplotlib.pyplot as plt

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