import datetime
import os

import torch
import matplotlib
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import heapq


class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        self.log_dir    = os.path.join(log_dir, "loss_" + str(time_str))
        self.losses     = []
        self.val_loss   = []
        self.val_accuracy = []
        self.cal_accuracy = []
        
        # Heap for saving best models
        self.best_models = []
        self.save_dir = log_dir
        
        os.makedirs(self.log_dir)
        self.writer     = SummaryWriter(self.log_dir)
        try:
            dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss, cal_accuracy, val_accuracy):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)
        self.cal_accuracy.append(cal_accuracy)
        self.val_accuracy.append(val_accuracy)
        
        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_cal_accuracy.txt"), 'a') as f:
            f.write(str(cal_accuracy))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_accuracy.txt"), 'a') as f:
            f.write(str(val_accuracy))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.writer.add_scalar('cal_accuracy', cal_accuracy, epoch)
        self.writer.add_scalar('val_accuracy', val_accuracy, epoch)

        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        plt.plot(iters, self.cal_accuracy, 'blue', linewidth=2, label='calibration accuracy')
        plt.plot(iters, self.val_accuracy, 'black', linewidth=2, label='val_accuracy')

        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.cal_accuracy, num, 3), 'blue', linestyle='--', linewidth=2, label='smooth calibration accuracy')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_accuracy, num, 3), 'black', linestyle='--', linewidth=2, label='smooth val accuracy')

            
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")
        
    def save_best_models(self, model, epoch, loss, val_loss, val_accuracy):
        """
        Save top 3 models with highest validation accuracy
        """
        # Get all pth files in save directory
        all_files = [f for f in os.listdir(self.save_dir) if f.endswith('.pth')]
        
        # If current number of files exceeds 3, delete files with lower val_accuracy
        if len(all_files) > 3:
            # Parse val_accuracy values in filenames
            model_files = []
            for file in all_files:
                # Parse val_accuracy values in filenames
                parts = file.split('-')
                for i, part in enumerate(parts):
                    if part.startswith('val_accuracy') and i < len(parts) - 1:
                        try:
                            val_acc = float(parts[i + 1].split('.')[0])
                            model_files.append((val_acc, file))
                            break
                        except (IndexError, ValueError):
                            continue
            
            # If enough valid filenames are parsed
            if len(model_files) >= 3:
                # Sort by val_accuracy in descending order
                model_files.sort(reverse=True, key=lambda x: x[0])
                
                # Delete files with lower val_accuracy
                for val_acc, file in model_files[3:]:
                    removed_file = os.path.join(self.save_dir, file)
                    if os.path.exists(removed_file):
                        os.remove(removed_file)
        
        # Save current model
        torch.save(model.state_dict(), os.path.join(self.save_dir, "ep%03d-loss%.3f-val_loss%.3f-val_accuracy%.3f.pth" % (epoch + 1, loss, val_loss, val_accuracy)))