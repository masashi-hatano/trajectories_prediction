from utils.util import get_data_from_csv, screenToCamera, cameraToWorld, calculateRealCoordinate, createDataText, plot, extract_coordinates
import matplotlib.pyplot as plt

def main():
    image = plt.imread('images/mask_road.jpg')
    R, K, T = get_data_from_csv('1207_1444_12.csv', 10218)
    coordinates = extract_coordinates(R,K,T,image)
    plot(image, coordinates, 'images/extracted.jpg')
    
    data=[]
    for i in range(len(coordinates)):
        R, K, T = get_data_from_csv('1207_1444_12.csv', 10218)
        temporary_coordinate_camera = screenToCamera(coordinates[i], image, K)
        #print(temporary_coordinate_camera)
        temporary_coordinate_world, direction = cameraToWorld(temporary_coordinate_camera, R, T)
        #print(temporary_coordinate_world)
        real_coordinate = calculateRealCoordinate(T, direction, -1.35)
        #print(real_coordinate)
        data.append([str(10218), str(100+i), str(real_coordinate[0][0]), str(real_coordinate[2][0])])
    #print(data)
    createDataText('./data.txt', data)
    

if __name__ == '__main__':
    main()