import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam

from Network_modules import *
from optimizer import *
from scheduler import *
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# torch.manual_seed(1)

class SSL_Model(nn.Module):
    def __init__(self, args):
        super(SSL_Model, self).__init__() 
        self.args = args
        self.encoder = RN18_Encoder(self.args.is_pretrained)
        self.attn_module = Attn2D_Module()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.decoder = UNet_Decoder()
        self.train_params = self.parameters()
        if args.optimizer.upper() == 'SGD':
            self.optimizer = SGD(self.train_params, self.args.learning_rate)
        elif args.optimizer.upper() == 'LARS':
            self.optimizer = LARS(self.train_params, self.args.learning_rate)
        else: # Adam
            self.optimizer = Adam(self.train_params, self.args.learning_rate)
        self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer, self.args.warmup_epochs, self.args.max_epochs)
        
    def train_SSL(self, batch):

        self.train()
        self.optimizer.zero_grad()
        reconstr_loss = 0.0   

        '''
        TODO:   1. iterate through batch and perform: i) Attn ii) Reconstruction
                2. return reconstructed images at epoch intervals
                [N,P,C,H,W] --(+)--> [N,P,512] --> [N,512,4,4] --> [N,512] --> decoder
        '''

        batch['image'] = batch['image'].to(device)      # [N,3,256,256] 
        batch['patches'] = batch['patches'].to(device)  # [N,16,3,64,64]

        image_feature = self.encoder(batch['image'], pool=True) # [N,512]

        patch_attn_feats = []
        patch_attn_maps = []
        
        ### for N = 1 ###
        for patch in batch['patches'][0]: 
        # patch = batch['patches'][0] # [64,3,32,32]
            patch_feature_map = self.encoder(patch.unsqueeze(0), pool=False) # [N,512,1,1]
            attn_map, attn_feature = self.attn_module(image_feature, patch_feature_map) # [N,512]
            patch_attn_feats.append(attn_feature.unsqueeze(1)) # [N,1,512]
            patch_attn_maps.append(attn_map)

        patch_attn_feats = torch.cat(patch_attn_feats, dim=1) # [N,16,512]
        patch_attn_feats = patch_attn_feats.permute(0,2,1) # [N,512,16]
        
        patch_attn = self.avg_pool(patch_attn_feats) # [N,512,1]
        patch_attn = patch_attn.permute(0,2,1).squeeze(1) # [N,512]
        # patch_attn = patch_attn_feats.reshape(-1, patch_attn_feats.shape[1], int(math.sqrt(patch_attn_feats.shape[2])), int(math.sqrt(patch_attn_feats.shape[2]))) # [N,512,4,4]
        
        recons_image = self.decoder(patch_attn) # [N,3,256,256]

        reconstr_loss += F.mse_loss(batch['image'], recons_image)

        reconstr_loss.backward()
        self.optimizer.step()

        image_results = torch.cat([batch['image'].to(device), recons_image.to(device)], dim=0)
        
        return patch_attn_maps, reconstr_loss.item(), image_results


    def get_attn_recons(self, batch):
        '''No gradients are to flow'''
        self.eval()
        with torch.no_grad():
            batch['image'] = batch['image'].to(device)      # [N,3,256,256] 
            batch['patches'] = batch['patches'].to(device)  # [N,16,3,64,64]

            image_feature = self.encoder(batch['image'], pool=True) # [N,512]

            patch_attn_feats = []
            patch_attn_maps = []
            
            ### for N = 1 ###
            for patch in batch['patches'][0]: 
            # patch = batch['patches'][0] # [64,3,32,32]
                patch_feature_map = self.encoder(patch.unsqueeze(0), pool=False) # [N,512,1,1]
                attn_map, attn_feature = self.attn_module(image_feature, patch_feature_map) # [N,512]
                patch_attn_feats.append(attn_feature.unsqueeze(1)) # [N,1,512]
                # patch_attn_maps.append(attn_map)

                # visualise attention maps #
                # print(attn_map.shape)
                attn_map = attn_map.unsqueeze(-1).view(attn_map.shape[0], 1, 2, 2) # [N,1,4] -> [N,1,2,2]
                I_grid = make_grid(batch['image'].to(device), normalize=True, scale_each=True)
                imposed_attn = visualize_attn_softmax(I_grid, attn_map, up_factor=128)
                # print(f"superimposed attn: {imposed_attn.shape}")
                patch_attn_maps.append(imposed_attn.unsqueeze(0)) 

            # print(np.array(patch_attn_maps).shape)
            patch_attn_maps = torch.mean(torch.stack(patch_attn_maps), dim=0)
            # print(f"patch attn map: {patch_attn_maps.shape}")
            patch_attn_maps = torch.cat([patch_attn_maps.to(device)], dim=0)

            patch_attn_feats = torch.cat(patch_attn_feats, dim=1) # [N,16,512]
            patch_attn_feats = patch_attn_feats.permute(0,2,1) # [N,512,16]
            
            patch_attn = self.avg_pool(patch_attn_feats) # [N,512,1]
            patch_attn = patch_attn.permute(0,2,1).squeeze(1) # [N,512]
            
            recons_image = self.decoder(patch_attn) # [N,3,256,256]
        
        return patch_attn_maps, recons_image






