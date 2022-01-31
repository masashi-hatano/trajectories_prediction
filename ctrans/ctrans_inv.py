import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cv2
from ctrans import get_data_from_csv

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

def main():
    time = []
    with open('timestamp3.txt') as f:
        for line in f:
            time.append(line.strip())
    
    predictions = []
    sys.path.append(str(Path('ctrans_inv.py').resolve().parent.parent))
    with open(sys.path[-1]+'\\socialgan\\datasets\\original\\scean4\\output.txt') as f:
    #with open(sys.path[-1]+'\\socialgan\\datasets\\original\\scean1\\output.txt') as f:
        for line in f:
            predictions.append(line.strip().split('\t'))

    R, K, T = get_data_from_csv('0129_1712_17.csv', int(time[-1]))
    image_path = 'images/0129_1712_17/'+time[-1]+'.jpg'
    im = plt.imread(image_path)
    image = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
    plt.imshow(image)
    plt.show()
    plt.close()

    coordinates = []
    for i in range(len(predictions)):
        if predictions[i][1]!=str(0):
            real_coordinate = np.array([float(predictions[i][2]),-1.35,float(predictions[i][3])]).reshape(3,1)
            real_coordinate_camera = worldToCamera(real_coordinate, R, T)
            coordinates.append(cameraToScreen(real_coordinate_camera, image, K))
    
    plot(image, coordinates, 'images/result_four_people_1.jpg')
    
if __name__ == '__main__':
    main()