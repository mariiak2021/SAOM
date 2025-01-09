import json
import os
import torch
from torch.utils.data import Dataset
import cv2
from per_segment_anything.utils.transforms import ResizeLongestSide
from collections import Counter

class Scene_Change_Finetune_Dataset2(Dataset):

    def __init__(self, meta_data, data_path, split, img_size=1024):
        self.data_path = data_path
        with open(meta_data, 'r') as f:
            meta_data = json.load(f)
        self.split = split
        self.classes = meta_data['entity classes']
        self.data = meta_data[split+'_data']

        self.transform = ResizeLongestSide(img_size)


    def __getitem__(self, index):
        image_info = self.data[index]
        
        labels = torch.tensor(image_info['entity_indices'], dtype=torch.long)

        image = cv2.imread(os.path.join(self.data_path, image_info['image_name']))#cv2.imread(os.path.join(self.data_path, image_info['image_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_name = image_info['image_name'].split("/")[-1].split(".")[0]

        ref_masks = []
        gt_masks = []
        mask_names = []
        for i in image_info['entity_masks']:
            mask_name = i.split("/")[-1].split("-")[3]
            mask = cv2.imread(self.data_path + i)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            gt_mask = torch.tensor(mask)[:, :, 0] > 0
            gt_mask = gt_mask.float().unsqueeze(0).flatten(1)
            gt_masks.append(gt_mask)
            mask_names.append(mask_name)

            input_mask = self.transform.apply_image(mask)
            input_mask = torch.as_tensor(input_mask)
            input_mask = input_mask.permute(2, 0, 1).contiguous()

            ref_masks.append(input_mask)

        
        gt_masks = torch.cat(gt_masks, dim=0)
        ref_masks = torch.stack(ref_masks)

        if self.split == "train":
            return image, labels, gt_masks, ref_masks, image_name
        elif self.split == "val" or self.split == "test":
            return image, labels, gt_masks, ref_masks, image_name, mask_names

    def __len__(self):
        return len(self.data)



