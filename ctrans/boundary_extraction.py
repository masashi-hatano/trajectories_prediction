from ctrans import get_data_from_csv, screenToCamera, cameraToWorld, calculateRealCoordinate, createDataText
from ctrans_inv import worldToCamera, cameraToScreen, plot
import matplotlib.pyplot as plt
import cv2
import numpy as np

def make_coordinaets(first_coordinate):
    coordinates = []
    for i in range(10):
        coordinates.append([first_coordinate[0]+i*14,first_coordinate[1]+i*9])
    return coordinates

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

def main():
    image = plt.imread('images/mask_road.jpg')
    R, K, T = get_data_from_csv('1207_1444_12.csv', 10218)
    coordinates = extract_coordinates(R,K,T,image)
    plot(image, coordinates, 'images/extracted.jpg')
    """
    coordinates = make_coordinaets([500,844])
    data=[]
    for i in range(len(coordinates)):
        R, K, T = get_data_from_csv('1207_1444_12.csv', 10218)
        temporary_coordinate_camera = screenToCamera(coordinates[i], image, K)
        print(temporary_coordinate_camera)
        temporary_coordinate_world, direction = cameraToWorld(temporary_coordinate_camera, R, T)
        print(temporary_coordinate_world)
        real_coordinate = calculateRealCoordinate(T, direction, -1.35)
        print(real_coordinate)
        data.append([str(10218), str(100+i), str(real_coordinate[0][0]), str(real_coordinate[2][0])])
        if i == len(coordinates)-1:
            temporary_coordinate_camera = screenToCamera([1440,1920], image, K)
            print(temporary_coordinate_camera)
            temporary_coordinate_world, direction = cameraToWorld(temporary_coordinate_camera, R, T)
            print(temporary_coordinate_world)
            real_coordinate = calculateRealCoordinate(T, direction, -1.35)
            print(real_coordinate)
            data.append([str(10218), str(100+i), str(real_coordinate[0][0]), str(real_coordinate[2][0])])
    print(data)
    #createDataText('./data.txt', data)
    """

if __name__ == '__main__':
    main()