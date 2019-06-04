# Breast-Cancer-MaskRCNN
This is a breast cancer diagnosis system based on Mask RCNN with mammography. 
Given a mammography, the system is supposed to:
1. locate the masses (object detection)
2. classify the pathology (malignant/benign) of the masses (image classification)
3. segment the mass (instance segmentation) 
Basically, this is an instance segmentation task. 

This is the implementation of my bachelor thesis. The title of the thesis is Breast Cancer Diagnosis Based on Artificial Intelligence with Mammography.
In the thesis, I explored different aproaches on mammography segmentation. 
This code is based on these two repositories: https://github.com/chevyng/Mammo_MaskRCNN and https://github.com/matterport/Mask_RCNN
