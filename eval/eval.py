from pathlib import Path
import json

from utils.util import convertToJson, eval, writeCSV, culculateAVG

def main():
    path_dates = Path('eval/dates.txt')

    with open(path_dates) as f:
        for date in f:
            date = date.strip()
            path_gt = Path('socialgan/datasets/original', date, 'withoutSS/data.txt')
            path_withoutCtrans = Path('output', date, 'withoutCtrans')
            path_withoutSS = Path('output', date, 'withoutSS')
            path_csv_result = Path('eval/result.csv')

            with open(path_gt) as f:
                gt = []
                time_list = []
                ped_list = []
                for line in f:
                    gt.append(line.strip().split('\t'))
                    gt[-1][0] = int(gt[-1][0])
                    gt[-1][1] = int(gt[-1][1])
                    if gt[-1][0] not in time_list:
                        time_list.append(gt[-1][0])
                    if gt[-1][1] not in ped_list:
                        ped_list.append(gt[-1][1])
            dict_gt = convertToJson(gt, time_list, ped_list)

            with open(path_withoutCtrans/Path('pred_traj_world.json')) as f:
                dict_withoutCtrans = json.load(f)

            with open(path_withoutSS/Path('pred_traj.json')) as f:
                dict_withoutSS = json.load(f)

            # eval withoutCtrans
            eval_withoutCtrans = eval(dict_gt, dict_withoutCtrans)

            # eval withoutSS
            eval_withoutSS = eval(dict_gt, dict_withoutSS)
            
            writeCSV(path_csv_result, date, eval_withoutCtrans, eval_withoutSS)
        
    culculateAVG(path_csv_result)

if __name__=='__main__':
    main()