import os
import time
from datetime import datetime, timezone
import torch
from torchvision.utils import make_grid, save_image
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from dataset import *
from model import *
from tensorplot import *

torch.manual_seed(1)

if __name__ =='__main__':

    import argparse
    parser = argparse.ArgumentParser('Dual Triplet')
    parser.add_argument('--base_dir', type=str, default=os.getcwd())
    parser.add_argument('--dataset', type=str, default='./../../../DATASETS/BHSig260/Bengali')
    parser.add_argument('--saved_models', type=str, default='./saved_models')
    parser.add_argument('--load_model', type=str, default='./../Final models/[FINAL_PAR-Encoder]BHSig260_Bengali_SSL_Encoder_RN18.pth')
    parser.add_argument('--batchsize', type=int, default=32)  # change for Testing; default=16
    parser.add_argument('--print_freq', type=int,default=100)
    parser.add_argument('--learning_rate', type=float, default=1.0)
    parser.add_argument('--learning_rate_AE', type=float, default=0.005)
    parser.add_argument('--margin', type=float, default=0.2)
    parser.add_argument('--lambd', type=float, default=1.0)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--round', type=int, choices=[1,2,3,4,5])
    parser.add_argument('--ssl_bkb', type=bool, default=False, help='if backbone is a SSL model or not')
    parser.add_argument("--bkb", type=str) # choices=['PAR', 'SimCLR','Barlow','SimSiam'])
    parser.add_argument('--set', type=str, default='self', choices=['self', 'cross_fulldata'])
    # parser.add_argument('--backbone', type=str, default='projector')
    args = parser.parse_args()

    # bkb = args.load_model.split('/')[2].split('_')[0] if args.ssl_bkb is True else "PAR"
    bkb =  args.bkb    

    if not os.path.exists(args.saved_models):
        os.mkdir(args.saved_models)

    print("\n--------------------------------------------------\n")    
    print(args)

    EXPT = f"{os.path.basename(args.dataset)}_backbone={bkb}_R{args.round}_{os.path.basename(args.dataset)[0]}_{args.set}"

    train_loader, _ = get_dataloader(args)

    print('-'* 50)


    ### setting up tensorboard ###
    # dt_curr = datetime.now(timezone.utc).strftime("%b:%d_%H:%M:%S")
    # tfb_dir = os.path.join(os.getcwd(), f"{args.saved_models}/tboard_logs/BHSig260Bengali_{EXPT}-_{dt_curr}")
    # Vis = Visualizer(tfb_dir)

    model = Triplet_Model(args)
    # model.encoder.load_state_dict(torch.load(args.load_model))
    epoch = 0

    if args.model_path is not None:
        checkpoint = torch.load(args.model_path)
        epoch = checkpoint['epochs']
        print(f">> Resume training from {epoch} epochs")
        model.load_state_dict(checkpoint['model'])

    model = model.to(device)

    logs = {'epoch':[], 'step':[], 'intra_loss':[], 'inter_loss': []}

    # 3. train model
    # for epoch in range(args.max_epochs):
    while epoch != args.max_epochs: ## Retraining from checkpoint ##
        epoch_loss_intra, epoch_loss_inter = 0.0, 0.0
        for ii, batch_train in enumerate(train_loader):
            step_count = (ii+1)+epoch*len(train_loader)
            start = time.time()
            model.train()

            intra_loss, inter_loss = model.train_model(batch=batch_train)
            epoch_loss_intra += intra_loss 
            epoch_loss_inter += inter_loss 
            # Vis.plot_current_losses(losses={'intra_loss' : intra_loss, 'inter_loss' : inter_loss},
            #                         step=step_count,
            #                         individual=False)

            if (ii+1) % args.print_freq == 0:
                logs['epoch'].append(epoch+1)
                logs['step'].append(ii+1)
                logs['intra_loss'].append(epoch_loss_intra/(ii+1))
                logs['inter_loss'].append(epoch_loss_inter/(ii+1))
                pd.DataFrame.from_dict(logs).to_csv(f"logs/{EXPT}.csv", index=False)
                
                # print(f'Epoch: {epoch+1}/{args.max_epochs} |' \
                #       f'Step: {ii+1}/{len(train_loader)} |' \
                #       f'Avg. Intra-class Loss: {(epoch_loss_intra/(ii+1)):.4f} |' \
                #       f'Avg. Inter-class Loss: {(epoch_loss_inter/(ii+1)):.4f} |')
    
        model.scheduler.step()
        epoch += 1

        # save model after every 5 epochs
        if (epoch+1) % 5 == 0:
            torch.save({'model': model.state_dict(), 'epochs': epoch+1},
                        f'{args.saved_models}/DTL_{EXPT}.pt')


    print('Training complete !!')

    # finally, save model backbone
    torch.save({'model': model.state_dict(), 'epochs': args.max_epochs},
                        f'{args.saved_models}/DTL_{EXPT}.pt')
    print('Model backbone saved !! \n\n')





