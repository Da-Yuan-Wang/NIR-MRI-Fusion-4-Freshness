import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets import get_model_from_name
from utils.callbacks import LossHistory
from utils.dataloader import DataGenerator, detection_collate
from utils.utils import (download_weights, get_classes, get_lr_scheduler,
                         set_optimizer_lr, weights_init)
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    #----------------------------------------------------#
    #   Whether to use Cuda
    #   Can be set to False without GPU
    #----------------------------------------------------#
    Cuda            = True # With GPU
    # Cuda            = False
    #---------------------------------------------------------------------#
    #   distributed     Used to specify whether to use single-machine multi-GPU distributed running
    #                   Terminal commands only support Ubuntu. CUDA_VISIBLE_DEVICES is used to specify GPUs under Ubuntu.
    #                   Windows system defaults to using DP mode to call all GPUs, does not support DDP.
    #   DP mode:
    #       Set            distributed = False
    #       Enter in terminal    CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP mode:
    #       Set            distributed = True
    #       Enter in terminal    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    #---------------------------------------------------------------------#
    distributed     = False
    #---------------------------------------------------------------------#
    #   sync_bn     Whether to use sync_bn, available for DDP mode multi-GPU
    #---------------------------------------------------------------------#
    sync_bn         = False
    #---------------------------------------------------------------------#
    #   fp16        Whether to use mixed precision training
    #               Can reduce about half of the GPU memory, requires pytorch1.7.1 or above
    #---------------------------------------------------------------------#
    fp16            = False
    #----------------------------------------------------#
    #   When training your own dataset, be sure to modify classes_path
    #   Modify to your corresponding class txt
    #----------------------------------------------------#
    classes_path    = 'MRI_pytorch_Deep_Learning_Models/model_data/cls_classes.txt' 
    #----------------------------------------------------#
    #   Input image size
    #----------------------------------------------------#
    input_shape     = [224, 224]
    #------------------------------------------------------#
    #   Model types:
    #   mobilenet, resnet50, vgg16, vit, and other types of lightweight networks
    #------------------------------------------------------#
    # backbone = "mobilenet" #MobileNetV2
    # backbone = "vit"
    # backbone = "mobilenetv1"
    # backbone = "ghostnet"
    # backbone = "cls_hrnet"
    # backbone = "shufflenet_v2"
    #backbone = "Xception"
    # backbone = "vgg16"
    backbone = "resnet50"

    #----------------------------------------------------------------------------------------------------------------------------#
    #   Whether to use the pre-trained weights of the backbone network. The backbone weights are loaded when the model is built.
    #   If model_path is set, the backbone weights do not need to be loaded, and the value of pretrained is meaningless.
    #   If model_path is not set and pretrained = True, only the backbone is loaded to start training.
    #   If model_path is not set and pretrained = False, Freeze_Train = False, training starts from 0 without freezing the backbone.
    #----------------------------------------------------------------------------------------------------------------------------#
    pretrained      = True # With pre-trained model
    #pretrained = False # Without pre-trained model
    #----------------------------------------------------------------------------------------------------------------------------#
    #   Please refer to README for downloading weight files, which can be downloaded through cloud storage. 
    #   The pre-trained weights of the model are universal for different datasets because features are universal.
    #   The more important part of the pre-trained weights is the weight part of the backbone feature extraction network, 
    #   which is used for feature extraction.
    #   Pre-trained weights are necessary for 99% of cases. Without them, the backbone weights are too random, 
    #   the feature extraction effect is not obvious, and the network training results will not be good.
    #
    #   If there is an interruption during the training process, you can set model_path to the weight file in the logs folder 
    #   to load the partially trained weights again.
    #   At the same time, modify the parameters of the freezing stage or unfreezing stage below to ensure the continuity of the model epoch.
    #   
    #   When model_path = '', the weights of the entire model are not loaded.
    #
    #   The weights of the entire model are used here, so they are loaded in train.py. Pretrain does not affect the weight loading here.
    #   If you want the model to start training from the backbone's pre-trained weights, 
    #   set model_path = '' and pretrain = True, then only the backbone is loaded.
    #   If you want the model to start training from 0, set model_path = '' and pretrain = False, then training starts from 0.
    #----------------------------------------------------------------------------------------------------------------------------#
    # model_path = ''
    # model_path = 'MRI_pytorch_Deep_Learning_Models/model_data/hrnetv2_w30_imagenet_pretrained.pth'
    # model_path = "MRI_pytorch_Deep_Learning_Models/model_data/vgg16-397923af.pth" # With pre-trained model
    # model_path = "MRI_pytorch_Deep_Learning_Models/model_data/vit-patch_16.pth" # With pre-trained model
    # model_path = "MRI_pytorch_Deep_Learning_Models/model_data/mobilenet_v2-b0353104.pth" # With pre-trained model
    model_path = "MRI_pytorch_Deep_Learning_Models/model_data/resnet50-19c8e357.pth"  # With pre-trained model, this is downloaded from the internet
    #----------------------------------------------------------------------------------------------------------------------------#
    #   Training is divided into two stages: freezing stage and unfreezing stage. 
    #   Setting the freezing stage is to meet the training needs of students with insufficient machine performance.
    #   Freezing training requires less GPU memory. In the case of very poor GPU, 
    #   you can set Freeze_Epoch equal to UnFreeze_Epoch, in which case only freezing training is performed.
    #      
    #   Several parameter setting recommendations are provided here. 
    #   Trainers can flexibly adjust according to their own needs:
    #   （一）Start training from the pre-trained weights of the entire model: 
    #       Adam：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'adam'，Init_lr = 1e-3。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'adam'，Init_lr = 1e-3。（不冻结）
    #       SGD：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'sgd'，Init_lr = 1e-2。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'sgd'，Init_lr = 1e-2。（不冻结）
    #       Among them: UnFreeze_Epoch can be adjusted between 100-300.
    #   （二）Start training from 0:
    #       Adam：
    #           Init_Epoch = 0，UnFreeze_Epoch = 300，Unfreeze_batch_size >= 16，Freeze_Train = False，optimizer_type = 'adam'，Init_lr = 1e-3。（不冻结）
    #       SGD：
    #           Init_Epoch = 0，UnFreeze_Epoch = 300，Unfreeze_batch_size >= 16，Freeze_Train = False，optimizer_type = 'sgd'，Init_lr = 1e-2。（不冻结）
    #       Among them: UnFreeze_Epoch should not be less than 300.
    #   （三）batch_size setting:
    #       Within the range that the GPU can accept, the larger the better. GPU memory is not related to the dataset size. 
    #       If you get an out-of-memory (OOM) error, reduce the batch_size.
    #       Affected by the BatchNorm layer, the minimum batch_size is 2, not 1.
    #       Normally, Freeze_batch_size is recommended to be 1-2 times Unfreeze_batch_size. 
    #       It is not recommended to set a large difference, as it affects the automatic adjustment of the learning rate.
    #----------------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Freezing stage training parameters
    #   At this time, the backbone of the model is frozen, and the feature extraction network does not change
    #   Less GPU memory is occupied, only the network is fine-tuned
    #   Init_Epoch          The current training epoch of the model, its value can be greater than Freeze_Epoch, such as setting:
    #                       Init_Epoch = 60、Freeze_Epoch = 50、UnFreeze_Epoch = 100
    #                       It will skip the freezing stage and start directly from epoch 60, and adjust the corresponding learning rate.
    #                       (Used for resuming training, set two places pretrained = False # Without pre-trained model)
    #   Freeze_Epoch        The freezing training epoch of the model
    #                       (Invalid when Freeze_Train=False)
    #   Freeze_batch_size   The batch_size of the freezing training
    #                       (Invalid when Freeze_Train=False)
    #------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 100
    Freeze_batch_size   = 16
    #------------------------------------------------------------------#
    #   Unfreezing stage training parameters
    #   At this time, the backbone of the model is not frozen, and the feature extraction network changes
    #   More GPU memory is occupied, all parameters of the network will change
    #   UnFreeze_Epoch          The total training epoch of the model
    #   Unfreeze_batch_size     The batch_size of the model after unfreezing
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 1200
    Unfreeze_batch_size = 16
    #------------------------------------------------------------------#
    #   Freeze_Train    Whether to perform freezing training
    #                   Default is to freeze the backbone first and then unfreeze training.
    #------------------------------------------------------------------#
    Freeze_Train        = True # With pre-trained model
    #Freeze_Train = False
    #------------------------------------------------------------------#
    #   Other training parameters: learning rate, optimizer, learning rate decay related
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr         The maximum learning rate of the model
    #                   When using the Adam optimizer, it is recommended to set  Init_lr=1e-3
    #                   When using the SGD optimizer, it is recommended to set   Init_lr=1e-2
    #   Min_lr          The minimum learning rate of the model, default is 1% of the maximum learning rate
    #------------------------------------------------------------------#
    Init_lr             = 1e-2
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    #   optimizer_type  The type of optimizer used, options are adam, sgd
    #                   When using the Adam optimizer, it is recommended to set  Init_lr=1e-3
    #                   When using the SGD optimizer, it is recommended to set   Init_lr=1e-2
    #   momentum        The momentum parameter used inside the optimizer
    #   weight_decay    Weight decay, can prevent overfitting
    #                   There will be an error when using the adam optimizer, it is recommended to set to 0
    #------------------------------------------------------------------#
    optimizer_type      = "sgd"
    momentum            = 0.9
    weight_decay        = 5e-4
    #------------------------------------------------------------------#
    #   lr_decay_type   The type of learning rate decay used, options are step, cos; step is suitable for training with a clear cycle; cos is suitable for training with many epochs
    #------------------------------------------------------------------#
    lr_decay_type       = "step"
    #------------------------------------------------------------------#
    #   save_period     How often to save weights, default is to save every epoch
    #------------------------------------------------------------------#
    save_period         = 1
    #------------------------------------------------------------------#
    #   save_dir        The folder to save weights and log files
    #------------------------------------------------------------------#
    save_dir            = 'MRI_pytorch_Deep_Learning_Models\logs'
    #------------------------------------------------------------------#
    #   num_workers     Used to set whether to use multi-threaded data reading
    #                   Enabling it will speed up data reading, but will occupy more memory
    #                   Computers with less memory can be set to 2 or 0  
    #------------------------------------------------------------------#
    num_workers         = 2

    #------------------------------------------------------#
    #   train_annotation_path   Training image path and label
    #   test_annotation_path    Validation image path and label (using test set as validation set)
    #------------------------------------------------------#
    train_annotation_path   = "MRI_pytorch_Deep_Learning_Models\cls_train.txt"
    test_annotation_path    = 'MRI_pytorch_Deep_Learning_Models\cls_test.txt'

    #------------------------------------------------------#
    #   Set the GPUs to be used
    #------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0

    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(backbone)  
            dist.barrier()
        else:
            download_weights(backbone)

    #------------------------------------------------------#
    #   Get classes
    #------------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)

    # if backbone != "vit":
    #     model = get_model_from_name[backbone](num_classes = num_classes, pretrained = pretrained)
    # else:
    #     model = get_model_from_name[backbone](input_shape = input_shape, num_classes = num_classes, pretrained = pretrained)

    if backbone == "vit":
        model = get_model_from_name[backbone](input_shape=input_shape, num_classes=num_classes, pretrained=pretrained)
    elif backbone == "mobilenet" or backbone == "mobilenetv1" or backbone == "resnet50" or backbone == "vgg16":
        model = get_model_from_name[backbone](num_classes=num_classes, pretrained=pretrained)
    # elif backbone == "ghostnet":
    #     model = get_model_from_name[backbone](num_classes=num_classes)
    else:
        model = get_model_from_name[backbone](num_classes=num_classes)
    if not pretrained:
        weights_init(model)
    if model_path != "":
        if local_rank == 0:
            #------------------------------------------------------#
            #   Load pre-trained weights
            #------------------------------------------------------#
            print('Loading weights into state dict...')
        model_dict = model.state_dict()
        # Fix device mismatch issue: always load weights on CPU first, then move to target device
        pretrained_dict = torch.load(model_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    if local_rank == 0:
        loss_history = LossHistory(save_dir, model, input_shape=input_shape)
    else:
        loss_history = None
        
    if fp16:
        #------------------------------------------------------------------#
        #   torch 1.2 does not support amp, it is recommended to use torch 1.7.1 and above to correctly use fp16
        #   Therefore, torch1.2 will show "could not be resolve"
        #------------------------------------------------------------------#
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    #----------------------------#
    #   Multi-GPU sync Bn
    #----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            #----------------------------#
            #   Multi-GPU parallel running
            #----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    #---------------------------#
    #   Read the txt corresponding to the dataset
    #---------------------------#
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(test_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)
    np.random.seed(10101)
    np.random.shuffle(train_lines)
    np.random.seed(None)

    #------------------------------------------------------#
    #   Backbone feature extraction network features are general, freezing training can speed up training speed
    #   It can also prevent weights from being destroyed in the initial training phase.
    #   Init_Epoch is the starting epoch
    #   Freeze_Epoch is the epoch for freezing training
    #   UnFreeze_Epoch is the total training epoch
    #   If you get an OOM or insufficient GPU memory error, reduce the Batch_size
    #------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        #------------------------------------#
        #   Freeze a certain part for training
        #------------------------------------#
        if Freeze_Train:
            model.freeze_backbone()

        #-------------------------------------------------------------------#
        #   If not freezing training, directly set batch_size to Unfreeze_batch_size
        #-------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        #-------------------------------------------------------------------#
        #   Determine the current batch_size, adaptively adjust the learning rate
        #-------------------------------------------------------------------#
        nbs             = 64
        lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
        if backbone == 'vit':
            nbs             = 256
            lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 1e-1
            lr_limit_min    = 1e-5 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
        
        optimizer = {
            'adam'  : optim.Adam(model_train.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay=weight_decay),
            'sgd'   : optim.SGD(model_train.parameters(), Init_lr_fit, momentum = momentum, nesterov=True)
        }[optimizer_type]
        
        #---------------------------------------#
        #   Get the learning rate decay formula
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        #---------------------------------------#
        #   Determine the length of each epoch
        #---------------------------------------#
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("Dataset is too small, unable to continue training, please expand the dataset.")
        # train_dataset = DataGenerator(train_lines, input_shape, False) #Data augmentation is turned on for the training set. Reason for training loss > validation loss, https://aman.ai/primers/ai/train-val-loss/#google_vignette
        train_dataset   = DataGenerator(train_lines, input_shape, True)
        val_dataset     = DataGenerator(val_lines, input_shape, False)
        
        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True
            
        gen             = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True, 
                                drop_last=True, collate_fn=detection_collate, sampler=train_sampler)
        gen_val         = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=detection_collate, sampler=val_sampler)
        #---------------------------------------#
        #   Start model training
        #---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            #---------------------------------------#
            #   If the model has a frozen learning part
            #   Then unfreeze and set parameters
            #---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                #-------------------------------------------------------------------#
                #   Determine the current batch_size, adaptively adjust the learning rate
                #-------------------------------------------------------------------#
                nbs             = 64
                lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 1e-1
                lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
                if backbone == 'vit':
                    nbs             = 256
                    lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 1e-1
                    lr_limit_min    = 1e-5 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                #---------------------------------------#
                #   Get the learning rate decay formula
                #---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                
                model.Unfreeze_backbone()

                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("Dataset is too small, unable to continue training, please expand the dataset.")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen             = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                        drop_last=True, collate_fn=detection_collate, sampler=train_sampler)
                gen_val         = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                        drop_last=True, collate_fn=detection_collate, sampler=val_sampler)

                UnFreeze_flag = True

            if distributed:
                train_sampler.set_epoch(epoch)
                
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            
            fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir, local_rank)

        if local_rank == 0:
            loss_history.writer.close()
