import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
from utils.util import get_data_from_csv, worldToCamera, cameraToScreen, plot, plot_gt

def main(date):
    time = []
    with open('ctrans/timestamp/'+date+'.txt') as f:
        for line in f:
            time.append(line.strip())
    
    sys.path.append(str(Path('ctrans_inv.py').resolve().parent.parent))
    gt = []
    with open(sys.path[-1]+'/socialgan/datasets/original/'+date+'/data.txt') as f:
        for line in f:
            gt.append(line.strip().split('\t'))
    with open(sys.path[-1]+'/output/'+date+'/pred_traj.json') as f:
        dict_json = json.load(f)

    for i in range(7,len(time)-8):
        R, K, T = get_data_from_csv(sys.path[-1]+'/dataset/csv/'+date+'.csv', int(time[i]))
        image_path = sys.path[-1]+'/dataset/images/'+date+'/'+time[i]+'.jpg'
        image = plt.imread(image_path)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #plt.imshow(image)
        #plt.show()
        #plt.close()

        time_current = []
        for j in range(1,9):
            time_current.append(time[i+j])
        coordinates_gt = []
        j=0
        while gt[j][0] not in time_current:
            j+=1
        while gt[j][0] in time_current:
            if gt[j][1]!=str(0):
                real_coordinate = np.array([float(gt[j][2]),-1.35,float(gt[j][3])]).reshape(3,1)
                real_coordinate_camera = worldToCamera(real_coordinate, R, T)
                coordinates_gt.append(cameraToScreen(real_coordinate_camera, image, K))
            j+=1
            if j >= len(gt):
                break
    
        image_ploted = plot(image, coordinates_gt, sys.path[-1]+'/output/'+date+'/result_'+time[i]+'.jpg', color=(0,255,0))

        coordinates_pred = []
        pedlist = dict_json["PredTimeList"][i-7]["PedList"]
        for k in range(1,len(pedlist)):
            pred_traj = pedlist[k]["pred_traj"]
            for pred_traj_coordinates in pred_traj:
                real_coordinate = np.array([pred_traj_coordinates[0],-1.35,pred_traj_coordinates[1]]).reshape(3,1)
                real_coordinate_camera = worldToCamera(real_coordinate, R, T)
                coordinates_pred.append(cameraToScreen(real_coordinate_camera, image, K))
        
        plot(image_ploted,coordinates_pred,sys.path[-1]+'/output/'+date+'/result_'+time[i]+'.jpg')

if __name__ == '__main__':
    main('0413_1605_24')