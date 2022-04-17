import sys
from pathlib import Path
from utils.util import filename_list, extract_imgs, create_timestamp

def main(foldername):
    sys.path.append(str(Path('timestamp_gen.py').resolve().parent.parent))
    imgs = filename_list(sys.path[-1]+'/dataset/images/'+foldername)

    imgs_extracted = extract_imgs(imgs)

    create_timestamp('timestamp/'+foldername+'.txt', imgs_extracted)

if __name__=='__main__':
    main('0413_1638_54')