from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cv2
import json

from utils.util import get_data_from_csv, worldToCamera, cameraToScreen, plot

def main(date):
    time = []
    gt = []
    path_timestamp = Path('ctrans/timestamp', date+'.txt')
    path_img = Path('dataset/images', date)
    path_csv = Path('dataset/csv',date+'.csv')
    path_input = Path('socialgan/datasets/original',date)
    path_input_withoutSS = path_input / Path('withoutSS/data.txt')
    path_output = Path('output',date)
    path_output_withoutSS_json = path_output / Path('withoutSS/pred_traj.json')
    path_output_withSS_json = path_output / Path('withSS/pred_traj.json')

    with open(path_timestamp) as f:
        for line in f:
            time.append(line.strip())
    
    with open(path_input_withoutSS) as f:
        for line in f:
            gt.append(line.strip().split('\t'))
    with open(path_output_withoutSS_json) as f:
        dict_json_withoutSS = json.load(f)
    with open(path_output_withSS_json) as f:
        dict_json_withSS = json.load(f)

    for i in range(7,len(time)-8):
        R, K, T = get_data_from_csv(path_csv, int(time[i]))
        path_image = path_img / Path(time[i]+'.jpg')
        image = plt.imread(path_image)
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
            if gt[j][1]!=str(0) and int(gt[j][1])<100:
                real_coordinate = np.array([float(gt[j][2]),-1.35,float(gt[j][3])]).reshape(3,1)
                real_coordinate_camera = worldToCamera(real_coordinate, R, T)
                coordinates_gt.append(cameraToScreen(real_coordinate_camera, image, K))
            j+=1
            if j >= len(gt):
                break
        
        path_output_img = path_output / Path('result_'+time[i]+'.jpg')
        image_ploted = plot(image, coordinates_gt, path_output_img, color=(0,255,0))

        coordinates_pred = []
        pedlist_withoutSS = dict_json_withoutSS["PredTimeList"][i-7]["PedList"]
        for k in range(1,len(pedlist_withoutSS)):
            if int(pedlist_withoutSS[k]["index"]) < 100:
                pred_traj = pedlist_withoutSS[k]["pred_traj"]
                for pred_traj_coordinates in pred_traj:
                    real_coordinate = np.array([pred_traj_coordinates[0],-1.35,pred_traj_coordinates[1]]).reshape(3,1)
                    real_coordinate_camera = worldToCamera(real_coordinate, R, T)
                    coordinates_pred.append(cameraToScreen(real_coordinate_camera, image, K))
        
        plot(image_ploted,coordinates_pred, path_output_img)

        coordinates_pred = []
        pedlist_withSS = dict_json_withSS["PredTimeList"][i-7]["PedList"]
        for k in range(1,len(pedlist_withoutSS)):
            if int(pedlist_withSS[k]["index"]) < 100:
                pred_traj = pedlist_withSS[k]["pred_traj"]
                for pred_traj_coordinates in pred_traj:
                    real_coordinate = np.array([pred_traj_coordinates[0],-1.35,pred_traj_coordinates[1]]).reshape(3,1)
                    real_coordinate_camera = worldToCamera(real_coordinate, R, T)
                    coordinates_pred.append(cameraToScreen(real_coordinate_camera, image, K))

        plot(image_ploted,coordinates_pred, path_output_img, color=(0,0,255))

if __name__ == '__main__':
    main('0129_1411_23')