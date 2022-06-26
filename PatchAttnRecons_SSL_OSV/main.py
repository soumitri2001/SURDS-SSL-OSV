

import os
import math
import copy
import time
from datetime import datetime, timezone
import webbrowser
import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from dataset import *
from model import *
from tensorplot import *

    # torch.manual_seed(1)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('UNet Decoder -- PatchAttnRecons | SSL for Writer Identification')
    parser.add_argument('--base_dir', type=str, default=os.getcwd())
    parser.add_argument('--dataset', type=str, default='./../BHSig260/Bengali')
    parser.add_argument('--saved_models', type=str, default='./saved_models')
    parser.add_argument('--batchsize', type=int, default=1)  # change for Testing; default=16
    parser.add_argument('--print_freq', type=int,default=100)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--fmap_dims', type=int, default=8)
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--ptsz', type=int, default=64)
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--is_pretrained', type=bool, default=False)
    parser.add_argument('--round', type=int, choices=[1,2,3,4,5])
    args = parser.parse_args()

    if not os.path.exists(args.saved_models):
        os.mkdir(args.saved_models)

    print("\n--------------------------------------------------\n")    
    print(args)

    # 1. load data
    train_loader, test_loader = get_dataloader(args)
    print(len(train_loader), len(test_loader))

    print('-'* 50)

    '''
    # ex_train, ex_test = iter(train_loader), iter(test_loader)
    # batch_train, batch_test = ex_train.next(), ex_test.next()

    # print(batch_train['image'].shape, batch_train['patches'].shape)
    # print(batch_test['image'].shape, batch_test['patches'].shape)

    # save_image(batch_train['image'][0], './saved_models/imgtrain.png')
    # save_image(make_grid(batch_train['patches'][0], nrow=256//args.ptsz), './saved_models/imgtrain_patches.png')

    # save_image(batch_test['image'][0], './saved_models/imgtest.png')
    # save_image(make_grid(batch_test['patches'][0], nrow=256//args.ptsz), './saved_models/imgtest_patches.png')


    ### setting up tensorboard ###
    # dt_curr = datetime.now(timezone.utc).strftime("%b:%d_%H:%M:%S")
    # tfb_dir = os.path.join(os.getcwd(), f"{args.saved_models}/tboard_logs/BHSig260Bengali_SSL_{dt_curr}")
    # Vis = Visualizer(tfb_dir)
    # os.system(f'tensorboard --logdir={tfb_dir}/TensorBoard_logs --host 0.0.0.0')
    # webbrowser.open("http://localhost:6006/")
    '''

    # 2. declare model
    ssl_model = SSL_Model(args)

    epoch = 0
    if args.load_model is not None:
        checkpoint = torch.load(args.load_model)
        ssl_model.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epochs']
        print(f">> Resume training from {epoch} epochs")

    ssl_model = ssl_model.to(device)

    logs = {'epoch' : [], 'step' : [], 'avg_recons_loss' : []}

    # 3. train model
    while epoch != args.max_epochs:
        epoch_loss = 0.0
        for ii, batch_train in enumerate(train_loader):
            
            step_count = (ii+1)+epoch*len(train_loader)
            start = time.time()
            ssl_model.train()

            _, recons_loss, image_results = ssl_model.train_SSL(batch=batch_train)
            epoch_loss += recons_loss
            # Vis.plot_current_losses(losses={'Recons_Loss' : recons_loss},
            #                         step=step_count,
            #                         individual=False)

            if (ii+1) % args.print_freq == 0:

                logs['epoch'].append(epoch+1)
                logs['step'].append(ii+1)
                logs['avg_recons_loss'].append(epoch_loss/(ii+1))

                pd.DataFrame.from_dict(logs).to_csv(f"logs/{os.path.basename(args.dataset)}_R={args.round}.csv", index=False)

                # print(f'Epoch: {epoch+1}/{args.max_epochs} |' \
                #       f'Step: {ii+1}/{len(train_loader)} |' \
                #       f'Avg. Recons_Loss: {epoch_loss/(ii+1):.4f} |' \
                #       f'Time: {time.time() - start}')

            # if recons_loss < 0.25:
            #     save_image(make_grid(image_results), f"./recons_{os.path.basename(args.dataset)}/Recons_Ep={epoch+1}_S={ii+1}_R={args.round}.png")
        
        ssl_model.scheduler.step()

        # save model after every 5 epochs
        if (epoch+1) % 5 == 0:
            torch.save({'model': ssl_model.state_dict(), 'epochs': epoch+1},
                        f'{args.saved_models}/BHSig260_{os.path.basename(args.dataset)}_R={args.round}_SSL.pt')
            torch.save(ssl_model.encoder.state_dict(), f'{args.saved_models}/BHSig260_{os.path.basename(args.dataset)}_R={args.round}_SSL_Encoder_RN18.pth')

        epoch += 1

    print('Training complete !!')

    # finally, save model backbone
    torch.save(ssl_model.encoder.state_dict(), f'{args.saved_models}/BHSig260_{os.path.basename(args.dataset)}_R={args.round}_SSL_Encoder_RN18.pth')
    print('Model backbone saved !! \n\n')

