import sys
from pathlib import Path

from utils.util import get_foot_coordinates, get_data_from_csv, screenToCamera, cameraToWorld, calculateRealCoordinate, createDataText

def main(text, csv):
    sys.path.append(str(Path('ctrans.py').resolve().parent.parent))
    time = []
    with open('timestamp/'+text) as f:
        for line in f:
            time.append(line.strip())

    data = []
    for i in range(len(time)):
        coordinates, image = get_foot_coordinates(sys.path[-1]+'/dataset/images/'+csv+'/'+time[i]+'.jpg')
        print(coordinates)
        j = 0
        flag=True
        while(j < len(coordinates)):
            if flag:
                R, K, T = get_data_from_csv(sys.path[-1]+'/dataset/csv/'+csv+'.csv', int(time[i]))
                data.append([time[i], str(0), str(T[0][0]), str(T[2][0])])
                flag=False
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
                    R, K, T = get_data_from_csv(sys.path[-1]+'/dataset/csv/'+csv+'.csv', int(time[i]))
                    temporary_coordinate_camera = screenToCamera(coordinates[j], image, K)
                    print(temporary_coordinate_camera)
                    temporary_coordinate_world, direction = cameraToWorld(temporary_coordinate_camera, R, T)
                    print(temporary_coordinate_world)
                    real_coordinate = calculateRealCoordinate(T, direction, -1.35)
                    print(real_coordinate)
                    data.append([time[i], str(index), str(real_coordinate[0][0]), str(real_coordinate[2][0])])
                    print(data)
            j+=1

    createDataText(sys.path[-1]+'\\socialgan\\datasets\\original\\scene5\\data.txt', data)

if __name__ == '__main__':
    main('timestamp5.txt', '0405_1433_18')