import os
import cv2
import torch
import numpy as np
from pathlib import Path
import argparse
import json
from tqdm import tqdm

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', type=str, default='/home/mkhan/embclip-rearrangement/Personalize-SAM/persam_f/3232allPhD/valmasksnormal/')
    parser.add_argument('--gt_path', type=str, default='/home/mkhan/embclip-rearrangement/data/newSCD/SAM/val/')
    parser.add_argument('--outdir', type=str, default='/home/mkhan/embclip-rearrangement/Personalize-SAM/persam_f/3232allPhD/')
    args = parser.parse_args()
    return args

def main():
    args = get_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    outdir = args.outdir
    mIoU, mAcc = 0, 0
    count = 0

    class_names2 = [
        "AlarmClock",
        "ArmChair",
        "BaseballBat",
        "BasketBall",
        "Bathtub",
        "Bed",
        "Blinds",
        "Book",
        "Bowl",
        "Box",
        "Cabinet",
        "CoffeeTable",
        "CounterTop",
        "Desk",
        "DiningTable",
        "Drawer",
        "Dresser",
        "Fridge",
        "GarbageCan",
        "Laptop",
        "LaundryHamper",
        "Microwave",
        "Mug",
        "Newspaper",
        "Ottoman",
        "Pan",
        "PaperTowelRoll",
        "Plate",
        "Plunger",
        "Pot",
        "Safe",
        "Shelf",
        "ShowerCurtain",
        "ShowerDoor",
        "SideTable",
        "Sink",
        "SoapBottle",
        "Sofa",
        "Statue",
        "StoveBurner",
        "TVStand",
        "TissueBox",
        "Toilet",
        "ToiletPaper",
        "ToiletPaperHanger",
        "Vase",
        "WateringCan",
        "ScrubBrush"
]

    
    pbar2 = tqdm(total=len(class_names2), desc=f"Process 1", leave=False)
    # Group ground truth and predicted images by scene and object in dictionaries
    
    for class_name in class_names2:
        print(class_name, len(class_names2))

        gt_files = os.listdir(args.gt_path)
        pred_files = os.listdir(args.pred_path)

        # Filter only .png files
        gt_images = [str(img_path) for img_path in gt_files if class_name in img_path and img_path.endswith('.png')]
        pred_images = [str(img_path) for img_path in pred_files if class_name in img_path and img_path.endswith('.png')]
        
        gt_dict = {}
        pred_dict = {}
        for gt_im in gt_images:
                gt_scene = gt_im.split("-")[0] + "-" + gt_im.split("-")[1]
                gt_obj = gt_im.split("-")[-1].split(".")[0]
                gt_type = gt_im.split("-")[2]
                gt_dict[(gt_scene, gt_obj, gt_type)] = gt_im
                
        for pred_im in pred_images:
                pred_scene = pred_im.split("-")[0]+ "-" + pred_im.split("-")[1]
                pred_type = pred_im.split("-")[2]
                pred_obj = pred_im.split("-")[-1].split(".")[0]
                pred_dict[(pred_scene, pred_obj, pred_type)] = pred_im

        print("img count", len(gt_images), len(pred_images))


        

        

        # Initialize metrics
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        num = 0
        
        # Process change files
        pbar3 = tqdm(total=len(pred_images), desc=f"Process 2", leave=False)
        for image in gt_images:
                pbar3.update(1)
                scene = image.split("-")[0] + "-" + image.split("-")[1]
                obj = image.split("-")[-1].split(".")[0]
                type = image.split("-")[2]
                key = (scene, obj, type)
                
                if key in gt_dict and key in pred_dict:
                    gt_im = gt_dict[key]
                    pred_im = pred_dict[key]
                    num += 1

                    gt_img = cv2.imread(args.gt_path + gt_im, cv2.IMREAD_GRAYSCALE) > 0
                    gt_img = torch.tensor(gt_img, dtype=torch.uint8, device=device)

                    pred_img = cv2.imread(args.pred_path + pred_im, cv2.IMREAD_GRAYSCALE) > 0
                    pred_img = torch.tensor(pred_img, dtype=torch.uint8, device=device)

                    # Compute intersection and union for IoU metric
                    intersection, union, target = intersectionAndUnion(pred_img, gt_img)
                    intersection_meter.update(intersection)
                    union_meter.update(union)
                    target_meter.update(target)
                else:
                    print (key)
                    gt_img = cv2.imread(args.gt_path + gt_im, cv2.IMREAD_GRAYSCALE) > 0
                    gt_img = torch.tensor(gt_img, dtype=torch.uint8, device=device)
                    pred_img = torch.zeros_like(gt_img)

                    # Compute intersection and union with the all-zero prediction
                    area_intersection = torch.logical_and(pred_img, gt_img).sum().item()
                    area_union = torch.logical_or(pred_img, gt_img).sum().item()
                    area_target = gt_img.sum().item()

                    # Update the meters
                    intersection_meter.update(torch.tensor(area_intersection))
                    union_meter.update(torch.tensor(area_union))
                    target_meter.update(torch.tensor(area_target))
                    


        # Calculate IoU and accuracy for the class
        print ("num, count", num, count)
        if num >0:
            count += 1
            iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
            accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
            print(class_name + ',', "IoU: %.2f," % (100 * iou_class), "Acc: %.2f" % (100 * accuracy_class), "Num: %.2f\n" % (num))
            outdir = Path(args.outdir)
            
            with (outdir / "validation26464saom.txt").open("a") as f:
                    f.write(json.dumps((class_name + ',', "IoU: %.2f," % (100 * iou_class), "Acc: %.2f" % (100 * accuracy_class), "Num: %.2f" % (num))) + "\n")
            mIoU += 100 * iou_class
            mAcc += 100 * accuracy_class
        
        pbar2.update(1)
    mIoU = mIoU / count
    mAcc = mAcc / count
    with (outdir / "validation26464saom.txt").open("a") as f:
        f.write(json.dumps({
            "mIoU": "%.2f" % mIoU,
            "mAcc": "%.2f" % mAcc,
            "Count": "%.2f" % count
        }) + "\n")
    
    
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def intersectionAndUnion(output, target):
    assert (output.ndimension() in [1, 2, 3])
    assert output.shape == target.shape

    # Use PyTorch functions for tensor operations
    area_intersection = torch.logical_and(output, target).sum().item()
    area_union = torch.logical_or(output, target).sum().item()
    area_target = target.sum().item()

    return area_intersection, area_union, area_target

if __name__ == '__main__':
    main()
      