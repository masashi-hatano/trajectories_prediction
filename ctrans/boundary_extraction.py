from ctrans import get_data_from_csv, screenToCamera, cameraToWorld, calculateRealCoordinate, createDataText
import matplotlib.pyplot as plt
import cv2

def make_coordinaets(first_coordinate):
    coordinates = []
    for i in range(10):
        coordinates.append([first_coordinate[0]+i*14,first_coordinate[1]+i*9])
    return coordinates

def main():
    im = plt.imread('images/10218.jpg')
    image = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
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
    print(data)
    createDataText('./data.txt', data)

if __name__ == '__main__':
    main()