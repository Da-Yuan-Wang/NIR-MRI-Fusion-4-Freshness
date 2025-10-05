import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from nets import get_model_from_name
from utils.utils import (cvtColor, get_classes, letterbox_image,
                         preprocess_input)

#--------------------------------------------#
#   To use your own trained model for prediction, you need to modify 3 parameters
#   model_path, classes_path, and backbone all need to be modified!
#--------------------------------------------#
class Classification(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   To use your own trained model for prediction, you must modify model_path and classes_path!
        #   model_path points to the weight file in the logs folder, classes_path points to the txt in model_data
        #   If shape mismatch occurs, also note the modification of model_path and classes_path parameters during training
        #--------------------------------------------------------------------------#
        "model_path"        : 'MRI_pytorch_Deep_Learning_Models\logs\Ghostnet-loss_2025_08_07_10_02_36\ep1130-loss0.363-val_loss0.405-val_accuracy0.901-used.pth',
        "classes_path"      : 'MRI_pytorch_Deep_Learning_Models/model_data/cls_classes.txt',
        #--------------------------------------------------------------------#
        #   Input image size
        #--------------------------------------------------------------------#
        "input_shape"       : [224, 224],
        #--------------------------------------------------------------------#
        #   Model types:
        #   mobilenet, resnet50, vgg16, vit
        #--------------------------------------------------------------------#
        "backbone"           : 'ghostnet',
        #"backbone"          : 'mobilenet',
        #"backbone"          : 'resnet50',
        #"backbone"          : 'vit',
        #"backbone"          : 'vgg16',
        #"backbone"           : 'cls_hrnet',
        #"backbone"            :"Xception",
        #--------------------------------------------------------------------#
        #   This variable controls whether to use letterbox_image to resize the input image without distortion
        #   Otherwise, CenterCrop is applied to the image
        #--------------------------------------------------------------------#
        "letterbox_image"   : False,
        #-------------------------------#
        #   Whether to use Cuda
        #   Can be set to False without GPU
        #-------------------------------#
        #"cuda"              : False
        "cuda"              : True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   Initialize classification
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        #---------------------------------------------------#
        #   Get classes
        #---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.generate()

    #---------------------------------------------------#
    #   Get all classifications
    #---------------------------------------------------#
    def generate(self):
        #---------------------------------------------------#
        #   Load model and weights
        #---------------------------------------------------#
        if self.backbone != "vit":
            #self.model  = get_model_from_name[self.backbone](num_classes = self.num_classes, pretrained = False)
            self.model  = get_model_from_name[self.backbone](num_classes = self.num_classes) #No pre-trained model used
        else:
            self.model  = get_model_from_name[self.backbone](input_shape = self.input_shape, num_classes = self.num_classes, pretrained = False)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.load_state_dict(torch.load(self.model_path, map_location=device))
        self.model  = self.model.eval()
        print('{} model, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.model = nn.DataParallel(self.model)
            self.model = self.model.cuda()

    #---------------------------------------------------#
    #   Detect image
    #---------------------------------------------------#
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
            photo   = torch.from_numpy(image_data)
            if self.cuda:
                photo = photo.cuda()
            #---------------------------------------------------#
            #   Pass image into network for prediction
            #---------------------------------------------------#
            preds   = torch.softmax(self.model(photo)[0], dim=-1).cpu().numpy()
        #---------------------------------------------------#
        #   Get the class
        #---------------------------------------------------#
        class_name  = self.class_names[np.argmax(preds)]
        probability = np.max(preds)

        #---------------------------------------------------#
        #   Draw and add text
        #---------------------------------------------------#
        plt.subplot(1, 1, 1)
        plt.imshow(np.array(image))
        plt.title('Class:%s Probability:%.3f' %(class_name, probability))
        plt.show()
        return class_name