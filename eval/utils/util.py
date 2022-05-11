from operator import itemgetter
import numpy as np
import pandas as pd


def convertToJson(data, time_list, ped_list):
    ped_list.remove(0)
    TimeList = []
    data = sorted(data, key=itemgetter(1,0))
    for time_current in time_list[7:-8]:
        dict_time = {'time_start': time_current, 'PedList':[]}
        TimeList.append(dict_time)
        for id_current in ped_list:
            traj = []
            counter= 0
            for time, id, x, y in data:
                if time in time_list and id == id_current:
                   traj.append([float(x),float(y)])
                   counter +=1
                   if counter==8:
                       break
                
            dict_traj = {'index':id_current, 'traj':traj}
            TimeList[-1]['PedList'].append(dict_traj)
    dict_gt = {'TimeList': TimeList}
    return dict_gt

def _findPredList(dict_pred, time_start, ped_id):
    for PredTimeList in dict_pred['PredTimeList']:
                if PredTimeList['time_start'] == time_start:
                    for PedList in PredTimeList['PedList']:
                        if PedList['index'] == ped_id:
                            pred_list = PedList['pred_traj']
                            return pred_list

def eval(dict_gt, dict_pred):
    ade_list = []
    fde_list = []

    for TimeList in dict_gt['TimeList']:
        for PedList in TimeList['PedList']:
            time_start = str(TimeList['time_start'])
            ped_id = str(PedList['index'])
 
            gt_list = PedList['traj']
            pred_list = _findPredList(dict_pred, time_start, ped_id)
            diff = np.subtract(np.array(gt_list), np.array(pred_list))
            list = np.linalg.norm(diff, axis=1)
            ade_list.append(np.mean(np.linalg.norm(diff, axis=1)))
            fde_list.append(np.linalg.norm(diff[-1]))

    return np.mean(ade_list), np.mean(fde_list)

def writeCSV(path, date, eval1, eval2):
    ade1, fde1 = eval1
    ade2, fde2 = eval2

    df = pd.read_csv(path)
    for i, date_ in enumerate(df.date):
        if date_ == date:
            df['w/o ctrans ADE'].iloc[i] = ade1
            df['w/o ctrans FDE'].iloc[i] = fde1
            df['w/ ctrans ADE'].iloc[i] = ade2
            df['w/ ctrans FDE'].iloc[i] = fde2

    df.to_csv('eval/result.csv', index=False)
    df.to_excel('eval/result.xlsx', index=False)