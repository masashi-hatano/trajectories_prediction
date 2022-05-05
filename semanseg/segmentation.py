from argparse import ArgumentParser
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
import torch
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2

from utils.util import convertToRGB
from utils.class_names import get_palette

def paletteToRGB(pred, size, palette):
    pred = F.interpolate(input=pred, size=size, mode='bilinear', align_corners=False)
    pred = np.asarray(np.argmax(pred.cpu(), axis=1), dtype=np.uint8)
    pred = np.squeeze(pred, axis=0)
    pred_rgb = convertToRGB(pred, palette)
    return pred_rgb

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img', default='dataset/images/', help='Image file')
    parser.add_argument('--model', default='nvidia/segformer-b5-finetuned-cityscapes-1024-1024')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--size', default=(1990,1440))
    parser.add_argument('--savedir', default='semanseg/output/')
    parser.add_argument('--palette', default='cityscapes', help='Color palette used for segmentation map')
    parser.add_argument('--timestamp', default='ctrans/timestamp/')
    parser.add_argument('--date', default='0129_1712_17')
    parser.add_argument('--interval', default=20, type=int)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # loarding model and feature extraction
    model = SegformerForSemanticSegmentation.from_pretrained(args.model)
    feature_extractor = SegformerFeatureExtractor.from_pretrained(args.model)

    # model to device
    device = torch.device(args.device)
    model = model.to(device)
    
    with open(args.timestamp+args.date+'.txt') as f:
        time = []
        for line in f:
            time.append(line.strip())
        
    for i in range(0,len(time),int(1.6*args.interval)):
        # prepare image
        image = cv2.imread(args.img+args.date+'/'+time[i]+'.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # test a single image
        model.to(args.device)
        model.eval()
        with torch.no_grad():
            inputs = feature_extractor(images=image, return_tensors="pt")
            inputs_tensor = inputs['pixel_values'].cuda()
            outputs = model(inputs_tensor)
            pred = outputs.logits

        # palette to rgb
        pred_rgb = paletteToRGB(pred, args.size, get_palette(args.palette))

        # create masked image
        masked = paletteToRGB(pred, args.size, get_palette('mask'))

        # save the segmented and masked images
        pred_rgb = cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR)
        masked = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
        cv2.imwrite(args.savedir+'seg/'+args.date+'/'+time[i]+'.jpg', pred_rgb)
        cv2.imwrite(args.savedir+'mask/'+args.date+'/'+time[i]+'.jpg', masked)


if __name__ == '__main__':
    main()
