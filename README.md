# Breast-Cancer-MaskRCNN
This is a breast cancer diagnosis system based on Mask RCNN with mammography. Basically, this is an instance segmentation task.
Given a mammography, the system is supposed to:
1. locate the masses (object detection)
2. classify the pathology (malignant/benign) of the masses (image classification)
3. segment the mass (instance segmentation)   
   
An example is: ![avatar]()
This is the implementation of my bachelor thesis. The thesis is titled *Breast Cancer Diagnosis Based on Artificial Intelligence with Mammography*.
In the thesis, I explored different aproaches on mammography segmentation. 
This code is based on these two repositories: https://github.com/chevyng/Mammo_MaskRCNN and https://github.com/matterport/Mask_RCNN

## Dataset
1. Download from [CBIS-DDSM](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM#7890e3c70fcf46819474c918d9817b1d)
2. Run `dataset.py`. The dataset will be organized as the file structures shown below.

## Train
1. Run `mammography/Extract_metadata_to_JSON.ipynb` to get data description file(s). Put the file(s) at `dataset/mammo/` (your dataset file).
2. Run `mammography/train.ipynb`.

## Test
Run `mammography/evaluate.ipynb`.

## Finetune
If you use the model and data provided by this work, you will get 0.7 on mass detection sensitivity and 0.53 on mass segmentation IOU.
Please tune the parameters and train more epoch to get a better model.

## Tips
The plot loss cube in training does not work well, please use the tools in `loss/` instead.

## File Structure