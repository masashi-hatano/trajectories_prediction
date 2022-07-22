from pathlib import Path
import json

from utils.util import convertToJson, eval, writeCSV, culculateAVG

def main():
    path_dates = Path('eval/dates.txt')

    with open(path_dates) as f:
        for date in f:
            date = date.strip()
            path_gt = Path('socialgan/datasets/original', date, 'multi/data.txt')
            path_withoutSP = Path('output', date, 'single','merge')
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

            with open(path_withoutSP/Path('pred_traj.json')) as f:
                dict_withoutSP = json.load(f)

            with open(path_withoutSS/Path('pred_traj.json')) as f:
                dict_withoutSS = json.load(f)

            # eval withoutCtrans
            eval_withoutSP = eval(dict_gt, dict_withoutSP)

            # eval withoutSS
            eval_withoutSS = eval(dict_gt, dict_withoutSS)
            
            writeCSV(path_csv_result, date, eval_withoutSP, eval_withoutSS)
        
    culculateAVG(path_csv_result)

if __name__=='__main__':
    main()