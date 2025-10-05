import os
from os import getcwd

from utils.utils import get_classes

#-------------------------------------------------------------------#
#   classes_path    Points to the txt under model_data, related to your own training dataset 
#                   Must modify classes_path before training to match your own dataset
#                   The txt file contains the categories you want to distinguish
#                   The classes_path used for training and prediction should be consistent
#-------------------------------------------------------------------#
classes_path    = 'MRI_pytorch_Deep_Learning_Models\model_data\cls_classes.txt'
#-------------------------------------------------------#
#   datasets_path   Points to the path where the dataset is located
#-------------------------------------------------------#
datasets_path   = 'data'

sets            = ["train", "test"]
classes, _      = get_classes(classes_path)

if __name__ == "__main__":
    for se in sets:
        # Modify file save path, save the generated file in the MRI_pytorch_Deep_Learning_Models folder
        output_file_path = os.path.join('MRI_pytorch_Deep_Learning_Models', 'cls_' + se + '.txt')
        list_file = open(output_file_path, 'w')

        datasets_path_t = os.path.join(datasets_path, se)
        types_name      = os.listdir(datasets_path_t)
        for type_name in types_name:
            if type_name not in classes:
                continue
            cls_id = classes.index(type_name)
            
            photos_path = os.path.join(datasets_path_t, type_name)
            photos_name = os.listdir(photos_path)
            for photo_name in photos_name:
                _, postfix = os.path.splitext(photo_name)
                if postfix not in ['.jpg', '.png', '.jpeg','.bmp']:  # Previous code did not have .bmp
                    continue
                list_file.write(str(cls_id) + ";" + '%s'%(os.path.join(photos_path, photo_name)))
                list_file.write('\n')
        list_file.close()