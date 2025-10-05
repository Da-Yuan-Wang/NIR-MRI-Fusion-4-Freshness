from .mobilenet import mobilenet_v2
from .resnet50 import resnet50
from .vgg16 import vgg16
from .vit import vit
from .mobilenetv1 import mbv1
from .ghostnet import ghostnet
from .cls_hrnet import hr_cls_net
from .shufflenet_v2 import ShuffleNetV2
from .Xception import Xception
get_model_from_name = {
    "mobilenet"     : mobilenet_v2,
    "resnet50"      : resnet50,
    "vgg16"         : vgg16,
    "vit"           : vit,
    "mobilenetv1"   : mbv1,
    "ghostnet"      : ghostnet,
    "cls_hrnet"     : hr_cls_net,
    "shufflenet_v2"  : ShuffleNetV2,
    "Xception"  : Xception
}

# Empty __init__.py file, used to resolve module import issues
