import argparse
import os
import sys
import torch
import logging
import json
from pathlib import Path

from attrdict import AttrDict
from torch.utils import data

from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path

sys.path.append(str(Path('prediction.py').resolve().parent.parent))

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default=sys.path[0]+'/models/sgan-p-models/zara1_8_model.pt')
parser.add_argument('--num_samples', default=1000, type=int)
parser.add_argument('--dset_type', default='test', type=str)
parser.add_argument('--date', default='0413_1638_54', type=str)
parser.add_argument('--input_type', default='withSS', choices=['withoutCtrans','withoutSS','withSS'])

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        #pooling_type="spool",
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)
    generator.load_state_dict(checkpoint['g_state'])
    #generator.cuda()
    generator.train()
    return generator


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def evaluate(args, loader, generator, num_samples):
    ade_outer, fde_outer = [], []
    total_traj = 0
    with torch.no_grad():
        for batch in loader:
            batch = [tensor for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch

            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)

            for _ in range(num_samples):
                pred_traj_fake_rel = generator(
                    obs_traj, obs_traj_rel, seq_start_end
                )
                pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel, obs_traj[-1]
                )
                ade.append(displacement_error(
                    pred_traj_fake, pred_traj_gt, mode='raw'
                ))
                fde.append(final_displacement_error(
                    pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
                ))

            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)
        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / (total_traj)
        return ade, fde, pred_traj_fake

def convertToJson(folder, data_dir, pred_traj_fake):
    data_list = []
    time_list = []
    PredTimeList = []
    counter = 0
    with open(data_dir/Path('data.txt')) as f:
        for line in f:
            if counter >= pred_traj_fake.shape[1]:
                break
            data_list.append(line.rstrip().split('\t'))
            if data_list[-1][0] not in time_list:
                time_list.append(data_list[-1][0])
                if len(time_list) >= 8:
                    pred_traj=[]
                    for i in range(8):
                        pred_traj.append(pred_traj_fake[i][counter].tolist())
                    dict = {"time_start":data_list[-1][0], "PedList":[{"index":data_list[-1][1], "pred_traj":pred_traj}]}
                    PredTimeList.append(dict)
                    counter+=1
            elif len(time_list) >= 8:
                pred_traj=[]
                for i in range(8):
                    pred_traj.append(pred_traj_fake[i][counter].tolist())
                dict = {"index":data_list[-1][1], "pred_traj":pred_traj}
                PredTimeList[-1]["PedList"].append(dict)
                counter+=1
    dict_all = {"PredTimeList":PredTimeList}
    #dict = sortJson(dict_all)
    if not os.path.exists(folder):
        os.mkdir(folder)
    with open(folder/Path("pred_traj.json"), "w") as f:
        json.dump(dict_all, f, indent=4)

# def sortJson(dict):
#     predlist = dict["PredTimeList"]
#     counter=0
#     inverse = False
#     while counter < len(predlist)-1:
#         if not inverse:
#             dict1_after, dict2_after, changed = sort(predlist[counter], predlist[counter+1])
#             if changed:
#                 predlist = change(predlist, counter, inverse, dict1_after, dict2_after)
#                 if counter > 0:
#                     inverse = True
#                 else:
#                     counter+=1
#             else:
#                 counter+=1
#         else:
#             dict1_after, dict2_after, changed = sort(predlist[counter-1], predlist[counter])
#             if changed:
#                 predlist = change(predlist, counter, inverse, dict1_after, dict2_after)
#                 if counter > 1:
#                     inverse = True
#                     counter-=1
#                 else:
#                     inverse = False
#             else:
#                 inverse = False
#                 counter+=1
#     dict["PredTimeList"] = predlist
#     return dict

# def sort(dict1, dict2):
#     changed = False
#     if dict1["PedList"][0]["pred_traj"][0][1] > dict2["PedList"][0]["pred_traj"][0][1]:
#         dict1_after = dict1
#         dict2_after = dict2
#     else:
#         dict1_after = dict2
#         dict2_after = dict1
#         changed = True
#     return dict1_after, dict2_after, changed

#  def change(predlist, index, inverse, dict1, dict2):
#     dict1["PedList"], dict2["PedList"] = dict2["PedList"], dict1["PedList"]
#     if not inverse:
#         predlist[index] = dict2
#         predlist[index+1] = dict1
#     else:
#         predlist[index-1] = dict2
#         predlist[index] = dict1
#     return predlist

def main(args):
    with open('socialgan/dates.txt') as f:
        for date in f:
            date = date.strip()
            folder = Path('output', date, args.input_type)
            path_datasets = Path('socialgan/datasets/original', date, args.input_type)
            if os.path.isdir(args.model_path):
                filenames = os.listdir(args.model_path)
                filenames.sort()
                paths = [
                    os.path.join(args.model_path, file_) for file_ in filenames
                ]
                
            else:
                paths = [args.model_path]

            for path in paths:
                checkpoint = torch.load(path, map_location=torch.device('cpu'))
                generator = get_generator(checkpoint)
                logger.info(generator)
                _args = AttrDict(checkpoint['args'])
                dset, loader = data_loader(_args, path_datasets)
                _, _, pred_traj_fake = evaluate(_args, loader, generator, args.num_samples)
                print(pred_traj_fake)
                convertToJson(folder, dset.data_dir, pred_traj_fake)
                #print('Dataset: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}'.format(_args.dataset_name, _args.pred_len, ade, fde))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
