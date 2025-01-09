import cv2, numpy as np
import matplotlib.pyplot as plt
from per_segment_anything import SamAutomaticMaskGenerator3_panoPhD, sam_model_registry, SamWrapper
from PIL import Image
import os
from show import *




sam_checkpoint = "/home/mkhan/embclip-rearrangement/Personalize-SAM/sam_vit_b_01ec64.pth"
model_type = "vit_b"

device = "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamWrapper(sam, 55)


sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator3_panoPhD(sam)


def test(name, indir, outdir):


    imageo = cv2.imread(indir+name)

    name = name.rsplit('.', 1)[0]
    print ("here", imageo.shape)

    image = cv2.cvtColor(imageo, cv2.COLOR_BGR2RGB)

    darkness_factor = 0.95

    # Apply the darkness factor to each channel (R, G, B)
    darkened_image_array = (image * darkness_factor).astype(np.uint8)

    masks = mask_generator.generate(image)

    masks = sorted(masks, key=(lambda x: x['predicted_iou']), reverse=False)

    darkened_image = Image.fromarray(darkened_image_array)

    # Save the image using PIL to ensure exact dimensions
    vis_mask_output_path = os.path.join(outdir, f'vis_mask_dark_{name}.jpg')
    
    if len(masks)>0:
        w,h = masks[0]['segmentation'].shape[0], masks[0]['segmentation'].shape[1]
        plt.figure(figsize=(w/100, h/100))
        plt.imshow(darkened_image)
        for (i, mask) in enumerate(masks):
            print (i)
            color_mask = np.concatenate([np.random.random(3), [0.65]])
                    
            
            show_mask2(masks[i]['segmentation'], plt.gca(), color = color_mask)
            print (masks[i]["predicted_iou"])


            plt.axis('off')

        vis_mask_output_path = os.path.join(outdir, f'vis_mask_{name }.jpg')
        with open(vis_mask_output_path, 'wb') as outfile:
            plt.savefig(outfile, format='jpg', bbox_inches='tight', pad_inches=0, dpi=500)
        plt.close()
    print ("")

    