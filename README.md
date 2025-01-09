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

I also experiment with the point grid size, used for SAOMv1 training. This adaptation aims to address SAM's bias towards selecting foreground object masks, leading to the development of the SAOMv2 model (trained with 64x64 point grid). 
I also propose using the panoramic object segmentation as an extension of embodied visual recognition. I leverage the agent’s ability to navigate a simulated 3D environment, capture images from multiple viewpoints and then stitch them into panoramas. I modify the grid size of SAOMv1 model to enable it's work in panoramic settings, resulting in the PanoSAM model (trained with 320x64 point grid). Both SAOMv2 and PanoSAM retain SAOMv1's core structure.

## News
The code for training, testing and evaluation of SAOMv1, SAOMv2 and PanoSAM are released on 09.01.25.
The SAOM and PanoSCU datasets will be released shortly.
