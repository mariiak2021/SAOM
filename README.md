# SEGMENT ANY OBJECT MODEL (SAOM): Real-to-Simulation Fine-Tuning Strategy for Multi-Class Multi-Instance Segmentation 
Mariia Khan⋆†, Yue Qiu⋆, Yuren Cong†, Bodo Rosenhahn†, Jumana Abu-Khalaf⋆†, David Suter⋆†


⋆† School of Science, Edith Cowan University (ECU), Australia

⋆Artificial Intelligence Research Center (AIRC), AIST, Japan

†Institute for Information Processing, Leibniz University of Hannover (LUH), Germany

[[`Paper`](https://arxiv.org/abs/2403.10780)] - accepted to [ICIP24](https://2024.ieeeicip.org/)

<p float="left">
  <img src="main.JPG?raw=true" width="30%" />
  <img src="pipeline.JPG?raw=true" width="65%" /> 
</p>

Multi-class multi-instance segmentation is the task of identifying masks for multiple object classes and multiple instances of the same class within an image. The foundational Segment Anything Model (SAM) is designed for promptable multi-class multi-instance segmentation but tends to output part or sub-part masks in the "everything" mode for various real-world applications. Whole object segmentation masks play a crucial role for indoor scene understanding, especially in robotics applications. We propose a new domain invariant Real-to-Simulation (Real-Sim) fine-tuning strategy for SAM. We use object images and ground truth data collected from Ai2Thor simulator during fine-tuning (real-to-sim). To allow our **Segment Any Object Model (SAOM)** to work in the "everything" mode, we propose the novel **nearest neighbour assignment** method, updating point embeddings for each ground-truth mask. SAOM is evaluated on our own dataset collected from Ai2Thor simulator. SAOM significantly improves on SAM, with a 28% increase in mIoU and a 25% increase in mAcc for 54 frequently-seen indoor object classes. Moreover, our Real-to-Simulation fine-tuning strategy demonstrates promising generalization performance in real environments without being trained on the real-world data (sim-to-real). The dataset and the code will be released after publication.

## News
The dataset and the code will be released shortly.
