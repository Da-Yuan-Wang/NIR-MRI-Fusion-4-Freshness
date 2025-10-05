'''
predict.py has several important points
1. Batch prediction is not possible. If you want to perform batch prediction, you can use os.listdir() to traverse the folder and use Image.open to open image files for prediction.
2. If you want to save the prediction results as txt, you can use open to open a txt file and use the write method to write to txt. You can refer to the txt_annotation.py file.
'''
from PIL import Image

from classification import Classification

classfication = Classification()

while True:
    img = input('Input image dir and filename:') # Prompt, please enter the image path in the terminal, such as: img/cat.jpg
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        class_name = classfication.detect_image(image)
        print(class_name)