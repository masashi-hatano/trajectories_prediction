import cv2
from argparse import ArgumentParser
import glob
from pathlib import Path
import os
from utils.util import fuseTerrainInfo
from utils.util import get_data_from_csv, screenToCamera, cameraToWorld, calculateRealCoordinate, createDataText, plot, extract_coordinates

def parse_arg():
    parser = ArgumentParser()
    parser.add_argument('--imgdir', default='semanseg/output/mask/')
    parser.add_argument('--savedir', default='ctrans/output/')
    parser.add_argument('--date', default='0413_1613_21')
    parser.add_argument('--inputdir', default='socialgan/datasets/original/')
    args = parser.parse_args()
    return args

def main():
    args = parse_arg()

    with open('ctrans/dates.txt') as f:
        for date in f:
            date = date.strip()
            imgs = glob.glob(args.imgdir+date+'/*.jpg')
            data=[]
            for i in range(len(imgs)):
                image = cv2.imread(imgs[i])
                time = imgs[i].replace('.jpg','').replace(args.imgdir+date+'\\','')
                R, K, T = get_data_from_csv('dataset/csv/'+date+'.csv', int(time))
                coordinates = extract_coordinates(R,K,T,image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                path_extracted = Path(args.savedir, 'extracted', date)
                if not os.path.exists(path_extracted):
                    os.mkdir(path_extracted)
                plot(image, coordinates, path_extracted/Path(time+'.jpg'), color=(0,0,255))
                
                nb_data = len(data)
                for j in range(nb_data,nb_data+len(coordinates)):
                    R, K, T = get_data_from_csv('dataset/csv/'+date+'.csv', int(time))
                    temporary_coordinate_camera = screenToCamera(coordinates[j-nb_data], image, K)
                    _, direction = cameraToWorld(temporary_coordinate_camera, R, T)
                    real_coordinate = calculateRealCoordinate(T, direction, -1.35)
                    data.append([time, str(100+j), str(real_coordinate[0][0]), str(real_coordinate[2][0])])

            path_terrain = Path(args.savedir, 'terrain_info')
            if not os.path.exists(path_terrain):
                os.mkdir(path_terrain)
            createDataText(path_terrain/Path(date+'.txt'), data)

            fused = fuseTerrainInfo(args.inputdir, date, data)

            path_withSS = Path(args.inputdir, date, 'withSS')
            if not os.path.exists(path_withSS):
                os.mkdir(path_withSS)
            createDataText(path_withSS/Path('data.txt'), fused)
    
if __name__ == '__main__':
    main()