import numpy as np
import matplotlib.pyplot as plt
import cv2
from coordinate_transformation import get_data_from_csv

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

def plot(image, coordinates):
    for i in range(len(coordinates)):
        cv2.circle(image, coordinates[i], 5, (0,0,255), -1)
    plt.imshow(image)
    plt.show()
    cv2.imwrite('images/result.jpg', image)

def main():
    time = []
    with open('timestamp.txt') as f:
        for line in f:
            time.append(line.strip())
    print(time[-1])
    
    predictions = []
    with open('datasets/original/scean1/output.txt') as f:
        for line in f:
            predictions.append(line.strip().split('\t'))

    R, K, T = get_data_from_csv('1207_1444_12.csv', int(time[-1]))
    image_path = 'images/'+time[-1]+'.jpg'
    im = plt.imread(image_path)
    image = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)

    coordinates = []
    for i in range(len(predictions)):
        real_coordinate = np.array([float(predictions[i][2]),-1.35,float(predictions[i][3])]).reshape(3,1)
        real_coordinate_camera = worldToCamera(real_coordinate, R, T)
        coordinates.append(cameraToScreen(real_coordinate_camera, image, K))
    
    print(coordinates)
    plot(image, coordinates)
    
if __name__ == '__main__':
    main()