# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch.nn import functional as F
from scipy.spatial import cKDTree
from PIL import Image
from .utils.transforms import ResizeLongestSide
import cv2


class SamWrapper(nn.Module):
    def __init__(self, sam_model, class_num, original_size=(224, 224), input_size=(1024,1024)) -> None:
        """
        Uses SAM to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam): The model to use for mask prediction.
          class_num: The number of the entitiy classes
        """
        super().__init__()
        self.model = sam_model
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        self.classifier = nn.Linear(256, class_num)
        self.img_features = None
        self.reset_image()
        
    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None

    def set_image(
        self,
        image: np.ndarray,
        
        ref_masks: np.ndarray = None,
        image_format: str = "RGB",
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray): The image for calculating masks. Expects an
            image in HWC uint8 format, with pixel values in [0, 255].
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        ##print ("old", image.shape)
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        input2 = self.set_torch_image(input_image_torch, image.shape[:2], transformed_mask=ref_masks)
        return input2

    @torch.no_grad()
    def set_torch_image(
        self,
        transformed_image: torch.Tensor,
        original_image_size: Tuple[int, ...],
        transformed_mask: torch.Tensor = None,
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method. Expects the input
        image to be already transformed to the format expected by the model.

        Arguments:
          transformed_image (torch.Tensor): The input image, with shape
            1x3xHxW, which has been transformed with ResizeLongestSide.
          original_image_size (tuple(int, int)): The size of the image
            before transformation, in (H, W) format.
        """
        assert (
            len(transformed_image.shape) == 4
            and transformed_image.shape[1] == 3
            and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."
        self.reset_image()
        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])
        input_image = self.model.preprocess(transformed_image)
        self.features = self.model.image_encoder(input_image)
        self.is_image_set = True
        if transformed_mask is not None:
          input_mask = self.model.preprocess(transformed_mask)
          return input_mask

    
    def process(self, image, size):
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        if all(isinstance(x, int) for x in size):
            # If all elements are integers, convert them to floats
            pass
        else:
            # If any element is not an integer, convert all elements to integers
            size = tuple(tensor.item() for tensor in size)
        self.reset_image()
        self.original_size = image.shape[-2:]#size
        self.input_size = tuple(input_image_torch.shape[-2:])
        input_image2 = self.model.preprocess(input_image_torch)
        self.features = self.model.image_encoder(input_image2)
        self.is_image_set = True


    def point_selection(self, mask_sim, topk=1):
        # Top-1 point selection
        w, h = mask_sim.shape
        topk_xy = mask_sim.flatten(0).topk(topk)[1]
        topk_x = torch.div(topk_xy, h, rounding_mode='floor').unsqueeze(0)
        topk_y = (topk_xy - topk_x * h)
        topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
        topk_label = np.array([1] * topk)
        topk_xy = topk_xy.cpu().numpy()

        return topk_xy, topk_label

    @torch.no_grad()
    def compute_points(self, image, ref_masks, point_grids):
        ref_masks = self.set_image(image, ref_masks)
        ref_feat = self.features.squeeze().permute(1, 2, 0)
        ref_masks = F.interpolate(ref_masks, size=(ref_feat.shape[0: 2]), mode="bilinear", align_corners=False)
        ref_masks = ref_masks[:, 0]


        remove_indices = torch.zeros(ref_masks.shape[0], dtype=torch.bool).to(ref_masks.device)
        h, w, C = ref_feat.shape

        points = []
        labels = []
        # Construct a KD-tree from the grid points
        # Define the scale factor for the translation
        scale_factor = 1024

        # Translate the grid of point coordinates
        point_grids = (point_grids * scale_factor).astype(int)
        kdtree = cKDTree(point_grids)
        
        for i in range(ref_masks.shape[0]):
        # Target feature extraction
            if (ref_masks[i] > 0).sum()>0:
                remove_indices[i] = True
                target_feat = ref_feat[ref_masks[i] > 0]
                target_feat_mean = target_feat.mean(0)
                target_feat_max = torch.max(target_feat, dim=0)[0]
                target_feat = (target_feat_max / 2 + target_feat_mean / 2).unsqueeze(0)
                target_feat = target_feat / target_feat.norm(dim=-1, keepdim=True)
                ref_feat_norm = ref_feat / ref_feat.norm(dim=-1, keepdim=True)
                ref_feat_norm = ref_feat_norm.permute(2, 0, 1).reshape(C, h * w)
                sim = target_feat @ ref_feat_norm
                sim = sim.reshape(1, 1, h, w)
                sim = F.interpolate(sim, scale_factor=4, mode="bilinear", align_corners=False)
                sim = self.model.postprocess_masks(sim,
                                                   input_size=self.input_size,
                                                   original_size=self.original_size).squeeze()
                # Positive location prior
                topk_xy, topk_label = self.point_selection(sim, topk=1)
                # Query the nearest neighbor
                _, nearest_idx = kdtree.query(topk_xy)

                # Get the nearest neighbor point
                nearest_point = point_grids[nearest_idx]
                points.append(nearest_point)
                labels.append(topk_label)
        return np.concatenate(points, axis=0), np.concatenate(labels, axis=0), remove_indices
    
    
    def predict(self, point_coords, point_labels):
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")
        point_coords = self.transform.apply_coords(point_coords, self.original_size)

        coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
        labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
        coords_torch, labels_torch = coords_torch[:, None, :], labels_torch[:, None]
        if point_coords is not None:
            points = (coords_torch, labels_torch)
        else:
            points = None

        # Embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=None,
            masks=None,
        )

        # Predict masks
        low_res_masks, iou_predictions, iou_token_out = self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            attn_sim=None,
            target_embedding=None
        )

        # Upscale the masks to the original image resolution
        high_res_masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)

        masks = high_res_masks > self.model.mask_threshold

        entity_logits = self.classifier(iou_token_out)

        return masks, iou_predictions, high_res_masks, entity_logits

    @torch.no_grad()
    def predict_torch(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        attn_sim = None,
        target_embedding = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using ResizeLongestSide.

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          boxes (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        if point_coords is not None:
            points = (point_coords, point_labels)
            #print (type(point_coords), type(point_labels))
        else:
            points = None

        # Embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )

        # Predict masks
        low_res_masks, iou_predictions, iou_token_out = self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            attn_sim=attn_sim,
            target_embedding=target_embedding
        )

        # Upscale the masks to the original image resolution
        high_res_masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)
        entity_logits = self.classifier(iou_token_out)
        if not return_logits:
            masks = high_res_masks > self.model.mask_threshold  # 0.0
            return masks, iou_predictions, low_res_masks, high_res_masks 
        else:
            return high_res_masks, iou_predictions, low_res_masks, entity_logits 
        
    def get_image_embedding(self) -> torch.Tensor:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert self.features is not None, "Features must exist if an image has been set."
        #print (self.features.shape)
        return self.features

    @property
    def device(self) -> torch.device:
        return self.model.device

