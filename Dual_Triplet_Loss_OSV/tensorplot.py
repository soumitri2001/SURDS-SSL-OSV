import os
import shutil
import torch
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Visualizer:
    def __init__(self, folder):
        folder = f'{folder}/TensorBoard_logs'
        if os.path.exists(folder):
            shutil.rmtree(folder)
        self.writer = SummaryWriter(folder, flush_secs=10)
        # self.mean = torch.tensor([-1.0, -1.0, -1.0]).to(device)
        # self.std = torch.tensor([1 / 0.5, 1 / 0.5, 1 / 0.5]).to(device)
        # value.sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None])  #Denormalization for tensors

    def plot_image(self, img_tensor, step=1):
        for key, value in img_tensor.items():
            value = value.to(device)
            self.writer.add_images(key, value, global_step=step, dataformats='NCHW')  # NCHW = [32, 3, 256, 256]

    def plot_current_losses(self, losses=None, step=1, individual=True):
        if individual:
            for key, value in losses.items():
                self.writer.add_scalar(key, value, step)
        else:
            self.writer.add_scalars('Losses', losses, step)