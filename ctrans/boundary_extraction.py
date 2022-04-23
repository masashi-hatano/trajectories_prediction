import cv2
from argparse import ArgumentParser
import glob

from utils.util import get_data_from_csv, screenToCamera, cameraToWorld, calculateRealCoordinate, createDataText, plot, extract_coordinates

def parse_arg():
    parser = ArgumentParser()
    parser.add_argument('--imgdir', default='semanseg/output/mask/')
    parser.add_argument('--savedir', default='ctrans/output/')
    parser.add_argument('--date', default='0413_1605_24')
    args = parser.parse_args()
    return args

def main():
    args = parse_arg()

    imgs = glob.glob(args.imgdir+args.date+'/*.jpg')
    data=[]
    for i in range(len(imgs)):
        image = cv2.imread(imgs[i])
        time = imgs[i].replace('.jpg','').replace(args.imgdir+args.date+'\\','')
        R, K, T = get_data_from_csv('dataset/csv/'+args.date+'.csv', int(time))
        coordinates = extract_coordinates(R,K,T,image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        plot(image, coordinates, args.savedir+'extracted/'+args.date+'/'+time+'.jpg', color=(0,0,255))
        
        nb_data = len(data)
        for j in range(nb_data,nb_data+len(coordinates)):
            R, K, T = get_data_from_csv('dataset/csv/'+args.date+'.csv', int(time))
            temporary_coordinate_camera = screenToCamera(coordinates[j-nb_data], image, K)
            #print(temporary_coordinate_camera)
            temporary_coordinate_world, direction = cameraToWorld(temporary_coordinate_camera, R, T)
            #print(temporary_coordinate_world)
            real_coordinate = calculateRealCoordinate(T, direction, -1.35)
            #print(real_coordinate)
            data.append([time, str(100+j), str(real_coordinate[0][0]), str(real_coordinate[2][0])])
    #print(data)
    createDataText(args.savedir+'terrain_info/'+args.date+'.txt', data)
    

if __name__ == '__main__':
    main()