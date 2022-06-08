from pathlib import Path
import cv2
import os

from utils.util import get_foot_coordinates, get_data_from_csv, screenToCamera, cameraToWorld, calculateRealCoordinate, createDataText

def main(date):
    time = []
    path_timestamp = Path('ctrans/timestamp', date+'.txt')
    path_img = Path('dataset/images', date)
    path_output = Path('ctrans/output/object_detection', date)

    with open(path_timestamp) as f:
        for line in f:
            time.append(line.strip())

    for i in range(len(time)):
        path_image = path_img / Path(time[i]+'.jpg')
        _, result = get_foot_coordinates(path_image)
        if not os.path.exists(path_output):
            os.mkdir(path_output)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(Path(path_output, time[i]+'.jpg')), result)

if __name__ == '__main__':
    main('0413_1605_24')