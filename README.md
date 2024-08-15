# Multicamera Labelling and training

This repository is designed to:
1. Create Jarvis formatted labelling sets from [multicamera_acquisition](https://github.com/dattalab-6-cam/multicamera_acquisition) recorded videos. This allows the user to manually label frames in their environment/rig to fine tune keypoint detection. 
2. Collate Jarvis formatted labelling sets into a COCO format training set. 
3. Train 2D [MMDetection](https://github.com/open-mmlab/mmdetection) models to detect mice. 
4. Train 2D [MMPose](https://github.com/open-mmlab/mmpose) models to find the keypoints on mice. 

Previously, I also had code that uploaded Jarvis labelling sets to scale.ai, and downloaded labels. scale.ai no longer supports keypoints. In the future, we could integrate with other labelling services. 

These trained models will output a `.pth` weights file, and a `.py` config file for the models you train. These two files are what you use in the "multicamera_airflow_pipeline" codebase. 


### Installation:

You will need a conda environment for training the neural networks. You will also need to *download* the mmpose and mmdetection repositories, so that you can load files from them. 

1. [Install mmpose](https://mmpose.readthedocs.io/en/latest/installation.html)
2. [Install mmdetection](https://mmdetection.readthedocs.io/en/latest/get_started.html)
2. install this repo with:

```
pip install -e .
```

### Usage

In the notebooks folder, there are notebooks to run through in order. They comprise these steps:

1. Create a labelling set for you videos. Make sure you have a created a calibration folder for that recording using  [multicam-calibration](https://github.com/dattalab-6-cam/multicam-calibration) in the Jarvis format. 
2. Manually label your videos using the [Jarvis Annotation Tool](https://jarvis-mocap.github.io/jarvis-docs/downloads/downloads/) (works on linux, mac, windows)
3. Generate a COCO formatted trainingset for use with MMDetection and MMPose. 
4. Train your models. 

In addition, there are notebooks to visualize how well your models perform on your data. 