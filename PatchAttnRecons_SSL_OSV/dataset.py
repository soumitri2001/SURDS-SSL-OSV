import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import scipy.ndimage as ndimage
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# np.random.seed(1)
# torch.manual_seed(1)

class Writer_Dataset(Dataset):
    def __init__(self, args, mode):

        self.args = args
        self.mode = mode
        self.ptsz = args.ptsz

        self.basic_transforms = transforms.Compose([transforms.Resize((256,256)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                         std=[0.229, 0.224, 0.225])
                                                    ])

        self.augment_transforms = transforms.Compose([transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.5),
                                                      transforms.RandomChoice([transforms.RandomEqualize(0.5),
                                                                               transforms.RandomInvert(0.5)]),
                                                      transforms.RandomApply([transforms.RandomAffine(
                                                          15, (0.05, 0.05), (1.5, 1.5), 0.5)], p=0.2)
                                                      ])

        # save all image paths with label (0/1)
        data_root = os.path.join(self.args.base_dir, self.args.dataset)  # BHSig260/Bengali

        data_df = pd.DataFrame(columns=['img_path', 'label'])

        for dir in os.listdir(data_root):
            dir_path = os.path.join(data_root, dir)
            if os.path.isdir(dir_path):
                for img in os.listdir(dir_path):
                    img_path = os.path.join(dir_path, img)
                    img_split = img.split('-')
                    label = 1 if img_split[3] == 'G' else 0
                    data_df = data_df.append({'img_path': img_path, 'label': label}, ignore_index=True)

        print(f'{self.args.dataset} comprises total {len(data_df)} images !!')

        self.train_df, self.test_df = train_test_split(data_df, test_size=0.3, shuffle=False, random_state=np.random.randint(0,100))

        # print(f'Training set: {len(self.train_df)} images | Val/Test set: {len(self.test_df)} images')

    def __len__(self):
        if self.mode == 'Train':
            return len(self.train_df)
        elif self.mode == 'Test':
            return len(self.test_df)
    
    def __get_com_cropped__(self, image):
        '''image is a binary PIL image'''
        image = np.asarray(image)
        com = ndimage.measurements.center_of_mass(image)
        com = np.round(com)
        com[0] = np.clip(com[0], 0, image.shape[0])
        com[1] = np.clip(com[1], 0, image.shape[1])
        X_center, Y_center = int(com[0]), int(com[1])
        c_row, c_col = image[X_center, :], image[:, Y_center]

        x_start, x_end, y_start, y_end = -1, -1, -1, -1

        for i, v in enumerate(c_col):
            v = np.sum(image[i, :])
            if v < 255*image.shape[1]: # there exists text pixel
                if x_start == -1:
                    x_start = i
                else:
                    x_end = i

        for j, v in enumerate(c_row):
            v = np.sum(image[:, j])
            if v < 255*image.shape[0]: # there exists text pixel
                if y_start == -1:
                    y_start = j
                else:
                    y_end = j

        crop_rgb = Image.fromarray(np.asarray(image[x_start:x_end, y_start:y_end])).convert('RGB')
        return crop_rgb

    def __getpatches__(self, x_arr):
        patches = []
        # x = Image.fromarray(x).convert('RGB').resize((256,256))
        # x_arr = np.asarray(x)
        C, H, W = x_arr.shape # 3, 256, 256

        ### non-overlapping patches ###
        num_H = H // self.ptsz
        num_W = W // self.ptsz

        for i in range(num_H):
            for j in range(num_W):
                start_x = i*self.ptsz
                end_x = start_x + self.ptsz
                start_y = j*self.ptsz
                end_y = start_y + self.ptsz

                patch = x_arr[:, start_x:end_x, start_y:end_y]
                # print(patch.shape)
                patch_tns = torch.from_numpy(patch)
                patches.append(torch.unsqueeze(patch_tns, 0))

        ### 50% pixel overlapping ###
        # num_H = (H//(self.ptsz//2)) - 1
        # num_W = (W//(self.ptsz//2)) - 1

        # for i in range(num_H):
        #     for j in range(num_W):
        #         start_x = i*(self.ptsz//2)
        #         end_x = start_x + self.ptsz
        #         start_y = j*(self.ptsz//2)
        #         end_y = start_y + self.ptsz

        #         patch = x_arr[:, start_x:end_x, start_y:end_y]
        #         # print(patch.shape)
        #         patch_tns = torch.from_numpy(patch)
        #         patches.append(torch.unsqueeze(patch_tns, 0))

        return torch.cat(patches, dim=0)

    def __getitem__(self, index):
        sample = {}
        if self.mode == 'Train':
            img_path = self.train_df.iloc[index]['img_path']
            sig_image = Image.open(img_path)
            writer_id = img_path.split('/')[-2]
            label = self.train_df.iloc[index]['label']
            cropped_sig = self.__get_com_cropped__(sig_image)
            # sig_image = self.basic_transforms(self.augment_transforms(cropped_sig))
            sig_image = self.basic_transforms(cropped_sig)
            sig_np = sig_image.numpy()
            # print("signp has shape: "+ str(sig_np.shape))
            sig_patches = self.__getpatches__(sig_np)

            sample = {'image' : sig_image, 'patches' : sig_patches, 'label' : label, 
                      'writer_id' : writer_id, 'img_name' : os.path.basename(img_path)}

        elif self.mode == 'Test':
            img_path = self.test_df.iloc[index]['img_path']
            sig_image = Image.open(img_path)
            writer_id = img_path.split('/')[-2]
            label = self.test_df.iloc[index]['label']
            cropped_sig = self.__get_com_cropped__(sig_image)
            sig_image = self.basic_transforms(cropped_sig)
            sig_np = sig_image.numpy()
            # print("signp has shape: "+ str(sig_np.shape))
            sig_patches = self.__getpatches__(sig_np)

            sample = {'image' : sig_image, 'patches' : sig_patches, 'label' : label,
                      'writer_id' : writer_id, 'img_name' : os.path.basename(img_path)}
        
        return sample


def get_dataloader(args):
        train_dset = Writer_Dataset(args, mode='Train')
        train_loader = DataLoader(train_dset, batch_size=args.batchsize, shuffle=True, num_workers=8)
        print('==> Train data loaded')
        
        test_dset = Writer_Dataset(args, mode='Test')
        test_loader = DataLoader(test_dset, batch_size=1, shuffle=False, num_workers=8)
        print('==> Test data loaded')

        return train_loader, test_loader
