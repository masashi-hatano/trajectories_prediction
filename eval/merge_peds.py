import argparse
from pathlib import Path

import json

def merge(json_file):
    merge_dict = json_file[0].copy()
    for i,json_dict in enumerate(json_file):
        if not i:
            continue
        for j,dict in enumerate(merge_dict["PredTimeList"]):
            dict["PedList"].append(json_dict["PredTimeList"][j]["PedList"][0])
    return merge_dict

def main(
    inputdir,
    num_peds
):  
    json_file = []
    for i in range(1,num_peds+1):
        json_path = Path(inputdir, str(i), "pred_traj.json")
        with open(str(json_path)) as f:
            json_file.append(json.load(f))
    merge_dict = merge(json_file)
    outputdir = Path(inputdir, "merge")
    outputdir.mkdir(parents=True, exist_ok=True)
    with open(Path(outputdir, "pred_traj.json"), "w") as f:
        json.dump(merge_dict, f, indent=4)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputdir", default=r"C:\Users\masashi\Python\TrajectoryPrediction\output\0413_1628_24\single")
    parser.add_argument("--num_peds", default=2)
    args = parser.parse_args()
    main(
        args.inputdir,
        args.num_peds
    )