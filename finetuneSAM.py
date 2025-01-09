import numpy as np
import torch
from torch.nn import functional as F
import json
from pathlib import Path
import os
import cv2
from tqdm import tqdm
import argparse
import random
import time
import datetime
from per_segment_anything.utils.amg import build_all_layer_point_grids_pano
from show import *
from per_segment_anything.samwrapperpano import SamWrapper
from per_segment_anything import sam_model_registry
from dataset.SCDTrack2PhD import Scene_Change_Finetune_Dataset2
import misc as utils
from torch.cuda.amp import autocast, GradScaler
import multiprocessing


def get_args_parser():
    
    parser = argparse.ArgumentParser('SAM fine-tuninig', add_help=False)

    parser.add_argument('--outdir', type=str, default='/home/mkhan/embclip-rearrangement/Personalize-SAM/persam_f/testdir/')
    parser.add_argument('--meta_data', type=str, default='/home/mkhan/embclip-rearrangement/DSmetadataPanoSAM.json')
    parser.add_argument('--data_path', type=str, default='/home/mkhan/stitching/data/track2')
    parser.add_argument('--ckpt', type=str, default='./sam_vit_h_4b8939.pth')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--batch_size', default=1, type=int) 
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--lr', type=int, default=1e-3)
    parser.add_argument('--train_epoch', type=int, default=500)
    parser.add_argument('--log_epoch', type=int, default=2)
    

    return parser
def adjust_omp_threads(num_processes):
        total_cores = multiprocessing.cpu_count()
        omp_threads = max(1, total_cores // num_processes)
        os.environ["OMP_NUM_THREADS"] = str(omp_threads)

def main(args):
    
    utils.init_distributed_mode(args)
    

    


    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    dataset_train = Scene_Change_Finetune_Dataset2(meta_data=args.meta_data,
                                            data_path=args.data_path,
                                            split='train')
    dataset_val = Scene_Change_Finetune_Dataset2(meta_data=args.meta_data,
                                            data_path=args.data_path,
                                            split='val')
    
    if args.distributed:
        sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train)
        sampler_val = torch.utils.data.distributed.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True) 
    batch_sampler_val = torch.utils.data.BatchSampler(
        sampler_val, args.batch_size, drop_last=True) 
    
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_sampler=batch_sampler_train, collate_fn = utils.custom_collate_fn,
                                   num_workers=args.num_workers)#, pin_memory=True
    dataloader_val = torch.utils.data.DataLoader(dataset_val, args.batch_size, batch_sampler=batch_sampler_val, collate_fn = utils.custom_collate_fn,
                                 drop_last=False, num_workers=args.num_workers)
    
    sam_type, sam_ckpt = 'vit_b', '/home/mkhan/embclip-rearrangement/Personalize-SAM/sam_vit_b_01ec64.pth'#sam_vit_h_4b8939.pth'
    sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).to(device)
    
    model = SamWrapper(sam, class_num=55)#SamWrapper(sam, class_num=len(dataset_train.classes))
    model.to(device)
    predictor = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        predictor = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    
    param_dicts = [
        {"params": [p for n, p in predictor.named_parameters() if p.requires_grad]},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, eps=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.train_epoch)


    
    
    outdir = Path(args.outdir)
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        predictor.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            print(f"RESUME EPOCH: {args.start_epoch}") 
    print (args.start_epoch)

    
    if not args.eval_only and not args.greedy_eval and not args.semsam:
        print('======> Start Training')
        start_time = time.time()
        for train_idx in range(args.start_epoch, 500):
            if args.distributed:
                sampler_train.set_epoch(train_idx)
            model.train()
            metric_logger = utils.MetricLogger(delimiter="  ")
            metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            header = 'Epoch: [{}]'.format(train_idx)
            print_freq = 25
            point_grids = build_all_layer_point_grids_pano(320,64,0,1)
            pbar = tqdm(total=len(dataloader_train), desc=f"Process 1", leave=False)
            
            scaler = GradScaler()  # Initialize GradScaler
            
            for batch in metric_logger.log_every(dataloader_train, print_freq, header):
                image, labels, gt_masks, ref_masks = batch[0][0], batch[0][1], batch[0][2], batch[0][3]
                ref_masks = ref_masks.to(device)

                with autocast():  # Enable mixed precision
                    in_points, in_labels, remove_indices = model.module.compute_points(image, ref_masks, point_grids[0])
                    labels = labels.to(device)
                    gt_masks = gt_masks.to(device)
                    gt_masks = gt_masks[remove_indices]
                    labels = labels[remove_indices]

                    masks, scores, logits_high, entity_logits = model.module.predict(in_points, in_labels)
                    logits_high = torch.diagonal(torch.index_select(logits_high, dim=1, index=scores.argmax(-1))).permute(2, 0, 1)
                    logits_high = logits_high.flatten(1)

                    entity_loss = F.cross_entropy(entity_logits, labels)

                    dice_loss = utils.calculate_dice_loss(logits_high, gt_masks)
                    focal_loss = utils.calculate_sigmoid_focal_loss(logits_high, gt_masks)
                    
                    losss = dice_loss + focal_loss + entity_loss

                optimizer.zero_grad()
                scaler.scale(losss).backward()  # Scale the loss before backward pass
                scaler.step(optimizer)  # Step the optimizer with scaled gradients
                scaler.update()  # Update the scaler for next iteration

                loss_dict = {"dice_loss": dice_loss, "focal_loss": focal_loss, "entity_loss": entity_loss}
                loss_dict_reduced = utils.reduce_dict(loss_dict)
                loss_dict_reduced_unscaled = {f'{k}': v for k, v in loss_dict_reduced.items()}
                losses_reduced = sum(loss_dict_reduced.values())
                loss_value = losses_reduced.item()

                metric_logger.update(loss=loss_value, **loss_dict_reduced_unscaled)
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])

                pbar.update(1)
            metric_logger.synchronize_between_processes()
            print("Averaged stats:", metric_logger)
            train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
            scheduler.step()
            pbar.close()

        

            
            if args.outdir:
                print ("outdir", outdir)
                checkpoint_paths = [outdir / 'checkpoint.pth']
                checkpoint_paths.append(outdir / f'checkpoint{train_idx:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                    'model': predictor.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': scheduler.state_dict(),
                    'epoch': train_idx,
                    'args': args,
                }, checkpoint_path)
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                    'epoch': train_idx,
                    'n_parameters': n_parameters}
            if args.outdir and utils.is_main_process():
                with (outdir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        
    if args.eval or args.eval_only:
        print('======> Start Validation')
        model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Eval'
        print_freq = 25
        pbar2 = tqdm(total=len(dataloader_val), desc=f"Process 1", leave=False)
        meta = {}
        val_data = []
        point_grids = build_all_layer_point_grids_pano(32,32,0,1)#(64,64,0,1) for SAOMv2 or (320,64,0,1) for PanoSAM
        path = str(outdir)+ '/valmasksnormal/'
        maskslist = [file for file in os.listdir(path)]
        #print (maskslist)
        for batch in dataloader_val:
                
                image, labels, gt_masks, ref_masks, image_name, mask_names = batch[0][0], batch[0][1], batch[0][2], batch[0][3], batch[0][4], batch[0][5]
                image_name = image_name
                imagedata = {}
                if not any(image_name in item for item in maskslist):
                    
                    print (image_name)
                    imagedata['image_name'] = image_name#'SCDimages/val/' + image_name
                    imagedata['entity_masks'] = []
                    imagedata['entity_indices'] = []

                    labels = labels.to(device)#[0].to(device)
                    gt_masks = gt_masks.to(device)#[0].to(device)
                    ref_masks = ref_masks.to(device)#[0].to(device)

                    with autocast():
                        in_points, in_labels, remove_indices = model.module.compute_points(image, ref_masks, point_grids[0])
                        #use this code if you need to save image features:
                        #features = model.module.features
                        #pathfea = str(outdir)+ '/valmasksnormal/' + "img_features/"
                        #torch.save(features, pathfea+ image_name+'.pt')
                        gt_masks = gt_masks[remove_indices]
                        labels = labels[remove_indices]
                        remove_indexes_cpu = remove_indices.cpu().numpy()

                        # Use boolean indexing to filter the list
                        mask_names = [item for item, keep in zip(mask_names, remove_indexes_cpu) if keep]
                        masks, scores, logits_high, entity_logits = model.module.predict(in_points, in_labels)
                        

                        probs = F.softmax(entity_logits, dim=1)

                        # Get the predicted class for each sample in the batch
                        predicted_classes = torch.argmax(probs, dim=1)
                        predicted_classes = predicted_classes.tolist()
                        
                        masks = torch.diagonal(torch.index_select(logits_high, dim=1, index=scores.argmax(-1))).permute(2, 0, 1)
                        
                        for i in range(masks.size(0)):
                            
                            # Detach the tensor, move to CPU, and convert to NumPy
                            final_mask = masks[i].detach().cpu().numpy()

                            # Threshold the mask to make it binary (0 or 1)
                    

                            final_mask_binary = (final_mask > 0).astype(np.uint8)

                            # Create an empty BGR mask (black background)
                            height, width = final_mask_binary.shape
                            bgr_mask = np.zeros((height, width, 3), dtype=np.uint8)

                            # Set the red channel to the mask (mask as red, background as black)
                            bgr_mask[:, :, 2] = final_mask_binary * 255  # Red channel (OpenCV uses BGR format)

                            # Save the image using OpenCV
                            mask_output_path = os.path.join(outdir, 'valmasksnormal', image_name + "-" + mask_names[i] + '.png')
                            
                            cv2.imwrite(mask_output_path, bgr_mask)
                            imagedata['entity_masks'].append(mask_output_path)
                            imagedata['entity_indices'].append(predicted_classes[i])

                else:
                        print ("existing already", image_name)
                        pass
                
                val_data.append(imagedata)

                pbar2.update(1)
                torch.cuda.empty_cache()
                print ("UPDATE!!!")

        meta["val_data"] = val_data
        path = str(outdir)+ "/DSmetadata-predval.json"
        with open(path, "a") as jsonfile:
                    json.dump(meta, jsonfile, sort_keys=True, indent=4)

        
        pbar2.close()




entity_classes2 = [
        "AlarmClock",
        "ArmChair",
        "BaseballBat",
        "BasketBall",
        "Bathtub",
        "BathtubBasin",
        "Bed",
        "Blinds",
        "Book",
        "Bowl",
        "Box",
        "Bread",
        "Cabinet",
        "CoffeeTable",
        "CounterTop",
        "Cup",
        "Desk",
        "DiningTable",
        "DishSponge",
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
        "ScrubBrush",
        "Shelf",
        "ShowerCurtain",
        "ShowerDoor",
        "SideTable",
        "Sink",
        "SinkBasin",
        "SoapBar",
        "SoapBottle",
        "Sofa",
        "SprayBottle",
        "Statue",
        "StoveBurner",
        "TVStand",
        "TissueBox",
        "Toilet",
        "ToiletPaper",
        "ToiletPaperHanger",
        "Vase",
        "WateringCan"
    ]



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.outdir:
        Path(args.outdir).mkdir(parents=True, exist_ok=True)
        if args.eval_only or args.eval:
            Path(args.outdir + "valmasks/").mkdir(parents=True, exist_ok=True)
            Path(args.outdir+ "val/").mkdir(parents=True, exist_ok=True)
    # Example: Adjust based on 4 processes running in parallel
    adjust_omp_threads(4)
    main(args)
