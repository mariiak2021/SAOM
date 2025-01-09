# SEGMENT ANY OBJECT MODEL (SAOM): Real-to-Simulation Fine-Tuning Strategy for Multi-Class Multi-Instance Segmentation 
<p>Mariia Khan⋆†, Yue Qiu⋆, Yuren Cong†, Bodo Rosenhahn†, Jumana Abu-Khalaf⋆†, David Suter⋆†</p>


<p>⋆† School of Science, Edith Cowan University (ECU), Australia</p>

<p>⋆Artificial Intelligence Research Center (AIRC), AIST, Japan</p>

<p>†Institute for Information Processing, Leibniz University of Hannover (LUH), Germany</p>

[[`Paper`](https://arxiv.org/abs/2403.10780)] - accepted to [ICIP24](https://2024.ieeeicip.org/)

<p float="left">
  <img src="main.JPG?raw=true" width="30%" />
  <img src="pipeline.JPG?raw=true" width="65%" /> 
</p>

Multi-class multi-instance segmentation is the task of identifying masks for multiple object classes and multiple instances of the same class within an image. The foundational Segment Anything Model (SAM) is designed for promptable multi-class multi-instance segmentation but tends to output part or sub-part masks in the "everything" mode for various real-world applications. Whole object segmentation masks play a crucial role for indoor scene understanding, especially in robotics applications. We propose a new domain invariant Real-to-Simulation (Real-Sim) fine-tuning strategy for SAM. We use object images and ground truth data collected from Ai2Thor simulator during fine-tuning (real-to-sim). To allow our **Segment Any Object Model (SAOMv1)** to work in the "everything" mode, we propose the novel **nearest neighbour assignment** method, updating point embeddings for each ground-truth mask. SAOM is evaluated on our own dataset collected from Ai2Thor simulator. SAOM significantly improves on SAM, with a 28% increase in mIoU and a 25% increase in mAcc for 54 frequently-seen indoor object classes. Moreover, our Real-to-Simulation fine-tuning strategy demonstrates promising generalization performance in real environments without being trained on the real-world data (sim-to-real). The dataset and the code will be released after publication.

I also experiment with the point grid size, used for SAOMv1 training. This adaptation aims to address SAM's bias towards selecting foreground object masks, leading to the development of the **SAOMv2 model (trained with 64x64 point grid)**. I also propose using the panoramic object segmentation as an extension of embodied visual recognition. I leverage the agent’s ability to navigate a simulated 3D environment, capture images from multiple viewpoints and then stitch them into panoramas. I modify the grid size of SAOMv1 model to enable it's work in panoramic settings, resulting in the **PanoSAM model (trained with 320x64 point grid)**. Both SAOMv2 and PanoSAM retain SAOMv1's core structure.

## News
The code for training, testing and evaluation of SAOMv1, SAOMv2 and PanoSAM are released on 09.01.25. The re-training code can e used to fine-tune SAM model with any poind grid size and is topic agnostic (can be used for any simulator or domain) to output whole object segmentation masks.

The SAOM and PanoSCU datasets will be released shortly.

## 💻 Installation

To begin, clone this repository locally
```bash
git clone git@github.com:mariiak2021/SAOMv1-SAOMv2-and-PanoSAM.git 
```
<details>
<summary><b>See here for a summary of the most important files/directories in this repository</b> </summary> 
<p>

Here's a quick summary of the most important files/directories in this repository:
* `finetuneSAM.py` the fine-tuning script, which can be used for any of SAOMv1, SAOMv2 or PanoSAM training;
* `environment.yml` the file with all requirements to set up conda environment
* `show.py the file` used for saving output masks during testing the model
* `testbatch.py` the file to use while testing the re-trained model performance
* `eval_miou.py` the file to use for evaluating the output masks
* `DSmetadataPanoSAM.json` the mapping between masksa and images for PanoSAM model DS
* `DSmetadataSAOMv1.json` the mapping between masksa and images for SAOMv1 model DS
  `DSmetadataSAOMv2.json` the mapping between masksa and images for SAOMv2 model DS
* `per_segment_anything/`
    - `automatic_mask_generator.py` - The file used for testing fine-tuned SAM version, where you can set all parameters like IoU threshold.
    - `samwrapperpano.py` - The file used for training the model, e.g. finding the location prior for each object and getting it's nearest neighbor from the point grid.
* `persamf/` - the foder for output of the testing/training stages
* `dataset/`
    - `SCDTrack2PhD.py` - The file used for setting up the dataset files for traing/testing/validation




You can then install requirements by using conda, we can create a `embclone` environment with our requirements by running
```bash
export MY_ENV_NAME=embclip-rearrange
export CONDA_BASE="$(dirname $(dirname "${CONDA_EXE}"))"
export PIP_SRC="${CONDA_BASE}/envs/${MY_ENV_NAME}/pipsrc"
conda env create --file environment.yml --name $MY_ENV_NAME
```

Download weights for the original SAM  model (ViT-H SAM model and ViT-B SAM model.) from here (place the download .ph file into the root of the folder): 
```bash
https://github.com/facebookresearch/segment-anything
```
<details>


</p>
</details>

To train the model on several GPUs run:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 finetuneSAM.py --world_size 4
```

To evaluate the model run:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 finetuneSAM.py --world_size 4  --eval_only
```

After you get the output masks for evaluation run:
```bash
eval_miou.py
```

To run the re-trained model in the everything mode run:
```bash
tesbatch.py
```

