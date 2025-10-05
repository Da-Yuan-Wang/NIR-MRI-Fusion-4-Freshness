# Classification: Implementation of Classification Models in PyTorch
---

## Table of Contents
1. [Repository Updates Top News](#repository-updates)
2. [Required Environment](#required-environment)
3. [File Download](#file-download)
4. [Training Steps How2train](#training-steps)
5. [Prediction Steps How2predict](#prediction-steps)
6. [Evaluation Steps How2eval](#evaluation-steps)
7. [References](#references)

## Repository Updates
**`2022-03`**:**Major updates have been made, supporting step and cos learning rate decay methods, supporting adam and sgd optimizer selection, and supporting adaptive learning rate adjustment according to batch_size.**  
The original repository address in the BiliBili video is: https://github.com/bubbliiiing/classification-pytorch/tree/bilibili

**`2021-01`**:**Repository created, supporting model training, extensive comments, and multiple adjustable parameters. Supports top1-top5 accuracy evaluation.**   

## Required Environment
pytorch == 1.2.0

## File Download
Pre-trained weights required for training can all be downloaded from Baidu Cloud.     
Link: https://pan.baidu.com/s/1Jxeyeni45PvGDuPNdhAjCw    
Extraction code: uyke    

The sample cat-dog dataset used for training can also be downloaded from Baidu Cloud.   
Link: https://pan.baidu.com/s/1hYBNG0TnGIeWw1-SwkzqpA     
Extraction code: ass8    

## Training Steps
1. The images stored in the datasets folder are divided into two parts, with training images in train and test images in test.  
2. Before training, you need to first prepare the dataset. Create different folders in the train or test folder, with each folder named after the corresponding class name, and the images under the folder being images of that class. The file format can be referenced as follows:
```
|-datasets
    |-train
        |-cat
            |-123.jpg
            |-234.jpg
        |-dog
            |-345.jpg
            |-456.jpg
        |-...
    |-test
        |-cat
            |-567.jpg
            |-678.jpg
        |-dog
            |-789.jpg
            |-890.jpg
        |-...
```
3. After preparing the dataset, you need to run txt_annotation.py in the root directory to generate the cls_train.txt required for training. Before running, you need to modify the classes in it to match the classes you want to classify.   
4. Then modify cls_classes.txt in the model_data folder to also correspond to the classes you want to classify.  
5. After adjusting the network and weights you want to choose in train.py, you can start training!  

## Prediction Steps
### a. Using Pre-trained Weights
1. After downloading and extracting the library, there is already a trained cat-dog model mobilenet025_catvsdog.h5 in model_data. Run predict.py and input  
```python
img/cat.jpg
```
### b. Using Your Own Trained Weights
1. Train according to the training steps.  
2. In the classification.py file, modify model_path, classes_path, backbone and alpha in the following section to correspond to the trained files; **model_path corresponds to the weight file under the logs folder, classes_path is the classes corresponding to model_path, backbone corresponds to the backbone feature extraction network used, and alpha is the alpha value when using mobilenet**.  
```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   When using your own trained model for prediction, be sure to modify model_path and classes_path!
    #   model_path points to the weight file under the logs folder, classes_path points to the txt under model_data
    #   If there is a shape mismatch, also pay attention to modifying the model_path and classes_path parameters during training
    #--------------------------------------------------------------------------#
    "model_path"    : 'model_data/mobilenet_catvsdog.pth',
    "classes_path"  : 'model_data/cls_classes.txt',
    #--------------------------------------------------------------------#
    #   Input image size
    #--------------------------------------------------------------------#
    "input_shape"   : [224, 224],
    #--------------------------------------------------------------------#
    #   Types of models used:
    #   mobilenet, resnet50, vgg16 are commonly used classification networks
    #   cspdarknet53 is used as an example of how to use mini_imagenet to train your own pre-trained weights
    #--------------------------------------------------------------------#
    "backbone"      : 'mobilenet',
    #-------------------------------#
    #   Whether to use Cuda
    #   Can be set to False without GPU
    #-------------------------------#
    "cuda"          : True
}
```
3. Run predict.py and input  
```python
img/cat.jpg
```  

## Evaluation Steps
1. The images stored in the datasets folder are divided into two parts, with training images in train and test images in test. For evaluation, we use the images in the test folder.  
2. Before evaluation, you need to first prepare the dataset. Create different folders in the train or test folder, with each folder named after the corresponding class name, and the images under the folder being images of that class. The file format can be referenced as follows:
```
|-datasets
    |-train
        |-cat
            |-123.jpg
            |-234.jpg
        |-dog
            |-345.jpg
            |-456.jpg
        |-...
    |-test
        |-cat
            |-567.jpg
            |-678.jpg
        |-dog
            |-789.jpg
            |-890.jpg
        |-...
```
3. After preparing the dataset, you need to run txt_annotation.py in the root directory to generate the cls_test.txt required for evaluation. Before running, you need to modify the classes in it to match the classes you want to classify.   
4. Then in the classification.py file, modify the following section model_path, classes_path, backbone and alpha to correspond to the trained files; **model_path corresponds to the weight file under the logs folder, classes_path is the classes corresponding to model_path, backbone corresponds to the backbone feature extraction network used, and alpha is the alpha value when using mobilenet**.  
```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   When using your own trained model for prediction, be sure to modify model_path and classes_path!
    #   model_path points to the weight file under the logs folder, classes_path points to the txt under model_data
    #   If there is a shape mismatch, also pay attention to modifying the model_path and classes_path parameters during training
    #--------------------------------------------------------------------------#
    "model_path"    : 'model_data/mobilenet_catvsdog.pth',
    "classes_path"  : 'model_data/cls_classes.txt',
    #--------------------------------------------------------------------#
    #   Input image size
    #--------------------------------------------------------------------#
    "input_shape"   : [224, 224],
    #--------------------------------------------------------------------#
    #   Types of models used:
    #   mobilenet, resnet50, vgg16 are commonly used classification networks
    #   cspdarknet53 is used as an example of how to use mini_imagenet to train your own pre-trained weights
    #--------------------------------------------------------------------#
    "backbone"      : 'mobilenet',
    #-------------------------------#
    #   Whether to use Cuda
    #   Can be set to False without GPU
    #-------------------------------#
    "cuda"          : True
}
```
5. Run eval_top1.py and eval_top5.py to evaluate model accuracy.

## References
https://github.com/keras-team/keras-applications