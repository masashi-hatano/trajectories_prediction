from operator import itemgetter
from pathlib import Path
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--date', default='0413_1628_24', type=str)
    parser.add_argument('--input_type', default='withoutCtrans', choices=['withoutCtrans','withoutSS','withSS'])
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    data = []
    path_data = Path('socialgan/datasets/original', args.date, args.input_type, 'data.txt')

    with open(path_data) as f:
        for line in f:
            data.append(line.strip().split('\t'))
            data[-1][0] = int(data[-1][0])
            data[-1][1] = int(data[-1][1])
    
    data = sorted(data, key=itemgetter(0,1))

    with open(path_data, 'w') as f:
        for i in range(len(data)):
            f.write(str(data[i][0])+'\t'
            +str(data[i][1])+'\t'
            +data[i][2]+'\t'
            +data[i][3]+'\n')

if __name__=='__main__':
    main()