from pathlib import Path

from utils.util import get_foot_coordinates, get_data_from_csv, screenToCamera, cameraToWorld, calculateRealCoordinate, createDataText

def main(date):
    time = []
    data = []
    path_timestamp = Path('ctrans/timestamp', date+'.txt')
    path_img = Path('dataset/images', date)
    path_csv = Path('dataset/csv',date+'.csv')
    path_input = Path('socialgan/datasets/original',date,'data.txt')

    with open(path_timestamp) as f:
        for line in f:
            time.append(line.strip())

    for i in range(len(time)):
        path_img = path_img / Path(time[i]+'.jpg')
        coordinates, image = get_foot_coordinates(path_img)
        print(coordinates)
        j = -1
        flag=True
        while(j < len(coordinates)):
            if flag:
                R, K, T = get_data_from_csv(path_csv, int(time[i]))
                data.append([time[i], str(0), str(T[0][0]), str(T[2][0])])
                flag=False
                j+=1
            index = int(input("Input index number:"))
            if index == -1:
                x = int(input("Input coordinate x: "))
                y = int(input("Input coordinate y: "))
                coordinates.append((x,y))
                print(coordinates)
                j-=1
            if index != -2:
                if index != -1:
                    print(int(time[i]))
                    R, K, T = get_data_from_csv(path_csv, int(time[i]))
                    temporary_coordinate_camera = screenToCamera(coordinates[j], image, K)
                    print(temporary_coordinate_camera)
                    temporary_coordinate_world, direction = cameraToWorld(temporary_coordinate_camera, R, T)
                    print(temporary_coordinate_world)
                    real_coordinate = calculateRealCoordinate(T, direction, -1.35)
                    print(real_coordinate)
                    data.append([time[i], str(index), str(real_coordinate[0][0]), str(real_coordinate[2][0])])
                    print(data)
            j+=1

    createDataText(path_input, data)

if __name__ == '__main__':
    main('0413_1605_24')