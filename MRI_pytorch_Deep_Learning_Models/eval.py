import os

import numpy as np
import torch

from classification import (Classification, cvtColor, letterbox_image,
                            preprocess_input)
from utils.utils import letterbox_image
from utils.utils_metrics import evaluteTop1_5

#------------------------------------------------------#
#   test_annotation_path    Test image path and labels
#------------------------------------------------------#
test_annotation_path    = 'MRI_pytorch_Deep_Learning_Models\cls_test.txt'
#------------------------------------------------------#
#   metrics_out_path        Folder to save metrics
#------------------------------------------------------#
metrics_out_path        = "MRI_pytorch_Deep_Learning_Models\metrics_out"

class Eval_Classification(Classification):
    def detect_image(self, image):        
        #---------------------------------------------------------#
        #   Convert image to RGB image here to prevent grayscale image from reporting errors during prediction.
        #   The code only supports RGB image prediction, all other types of images will be converted to RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------#
        #   Resize image without distortion
        #---------------------------------------------------#
        image_data  = letterbox_image(image, [self.input_shape[1], self.input_shape[0]], self.letterbox_image)
        #---------------------------------------------------------#
        #   Normalize + add batch_size dimension + transpose
        #---------------------------------------------------------#
        image_data  = np.transpose(np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0), (0, 3, 1, 2))

        with torch.no_grad():
            photo   = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.cuda:
                photo = photo.cuda()
            #---------------------------------------------------#
            #   Pass image into network for prediction
            #---------------------------------------------------#
            preds   = torch.softmax(self.model(photo)[0], dim=-1).cpu().numpy()

        return preds

if __name__ == "__main__":
    if not os.path.exists(metrics_out_path):
        os.makedirs(metrics_out_path)
            
    classfication = Eval_Classification()
    
    with open("MRI_pytorch_Deep_Learning_Models/cls_test.txt","r") as f:
        lines = f.readlines()
    top1, top5, Recall, Precision = evaluteTop1_5(classfication, lines, metrics_out_path)
    
    print("top-1 accuracy = %.2f%%" % (top1*100))
    print("top-5 accuracy = %.2f%%" % (top5*100))
    print("mean Recall = %.2f%%" % (np.mean(Recall)*100))
    print("mean Precision = %.2f%%" % (np.mean(Precision)*100))