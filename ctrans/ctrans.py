import sys
from pathlib import Path

import numpy as np
import pandas as pd
from human_detection import get_foot_coordinates

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

def main(text, csv):
    time = []
    with open('timestamp/'+text) as f:
        for line in f:
            time.append(line.strip())

    data = []
    for i in range(len(time)):
        coordinates, image = get_foot_coordinates('images/'+csv+'/'+time[i]+'.jpg')
        print(coordinates)
        j = 0
        flag=True
        while(j < len(coordinates)):
            if flag:
                R, K, T = get_data_from_csv('csv/'+csv+'.csv', int(time[i]))
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
                    R, K, T = get_data_from_csv('csv/'+csv+'.csv', int(time[i]))
                    temporary_coordinate_camera = screenToCamera(coordinates[j], image, K)
                    print(temporary_coordinate_camera)
                    temporary_coordinate_world, direction = cameraToWorld(temporary_coordinate_camera, R, T)
                    print(temporary_coordinate_world)
                    real_coordinate = calculateRealCoordinate(T, direction, -1.35)
                    print(real_coordinate)
                    data.append([time[i], str(index), str(real_coordinate[0][0]), str(real_coordinate[2][0])])
                    print(data)
            j+=1

    sys.path.append(str(Path('ctrans.py').resolve().parent.parent))
    createDataText(sys.path[-1]+'\\socialgan\\datasets\\original\\scene5\\data.txt', data)

if __name__ == '__main__':
    main('timestamp5.txt', '0405_1433_18')