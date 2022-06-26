import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import scipy.ndimage as ndimage
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

random.seed(1)
np.random.seed(1)

class Writer_Dataset(Dataset):
    def __init__(self, args, mode):

        self.args = args
        self.mode = mode

        self.basic_transforms = transforms.Compose([transforms.RandomInvert(1.0),
                                                    transforms.Resize((256,256)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                         std=[0.229, 0.224, 0.225])
                                                    ])

        self.augment_transforms = transforms.Compose([transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.5),
                                                      transforms.RandomChoice([transforms.RandomEqualize(1.0),
                                                                               transforms.RandomAffine(15, (0.05, 0.05), (1.5, 1.5), 0.5)]),
                                                      
                                                      ])

        data_root = os.path.join(self.args.base_dir, self.args.dataset)  # ./../BHSig260/Bengali
        data_df = pd.DataFrame(columns=['img_path', 'label', 'writer_id'])

        for dir in os.listdir(data_root):
            dir_path = os.path.join(data_root, dir)
            if os.path.isdir(dir_path):
                for img in os.listdir(dir_path):
                    img_path = os.path.join(dir_path, img)
                    img_split = img.split('-')
                    label = None
                    if img_split[3] == 'G':
                        label = 1
                    elif img_split[3] == 'F':
                        label = 0
                    assert label is not None
                    data_df = data_df.append({'img_path': img_path, 'label': label, 'writer_id': dir}, ignore_index=True)

        print(f'{self.args.dataset} comprises total {len(data_df)} images !!')

        self.train_df, self.test_df = train_test_split(data_df, test_size=0.3, shuffle=False, random_state=1)

        # self.train_df = data_df.copy()

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

    def __getitem__(self, index):
        sample = {}
        if self.mode == 'Train':

            # Anchor
            img_path = self.train_df.iloc[index]['img_path']
            sig_image = Image.open(img_path)
            writer_id = img_path.split('/')[-2]
            label = self.train_df.iloc[index]['label']
            cropped_sig = self.__get_com_cropped__(sig_image)
            anchor_image = self.basic_transforms(cropped_sig)

            positive_path, negative_path_intra, negative_path_inter = None, None, None

            remaining_list = list(range(len(self.train_df)))
            remaining_list.remove(index)

            # positive
            while True:
                p_rand = random.randint(0, len(remaining_list) - 1)
                if p_rand != index and self.train_df.iloc[p_rand]['writer_id'] == self.train_df.iloc[index]['writer_id'] and self.train_df.iloc[p_rand]['label'] == self.train_df.iloc[index]['label']: # same writer+label
                    positive_path = self.train_df.iloc[p_rand]['img_path']
                    remaining_list.remove(p_rand)
                    break
            positive_sig_image = Image.open(positive_path)
            cropped_positive_sig = self.__get_com_cropped__(positive_sig_image)
            positive_image = self.basic_transforms(cropped_positive_sig)

            # intra-class negative 
            while True:
                n_rand = random.randint(0, len(remaining_list) - 1)
                if n_rand != index and self.train_df.iloc[n_rand]['writer_id'] == self.train_df.iloc[index]['writer_id'] and self.train_df.iloc[n_rand]['label'] != self.train_df.iloc[index]['label']: # same writer diff label
                    negative_path_intra = self.train_df.iloc[n_rand]['img_path']
                    negativeintra_label = self.train_df.iloc[n_rand]['label']
                    remaining_list.remove(n_rand)
                    break
            negativeintra_sig_image = Image.open(negative_path_intra)
            cropped_negativeintra_sig = self.__get_com_cropped__(negativeintra_sig_image)
            negativeintra_image = self.basic_transforms(cropped_negativeintra_sig)
            
            # inter-class negative
            while True:
                n_rand = random.randint(0, len(remaining_list) - 1)
                if n_rand != index and self.train_df.iloc[n_rand]['writer_id'] != self.train_df.iloc[index]['writer_id'] and self.train_df.iloc[n_rand]['label'] != self.train_df.iloc[index]['label']: # diff writer diff label
                    negative_path_inter = self.train_df.iloc[n_rand]['img_path']
                    negativeinter_label = self.train_df.iloc[n_rand]['label']
                    remaining_list.remove(n_rand)
                    break
            negativeinter_sig_image = Image.open(negative_path_inter)
            cropped_negativeinter_sig = self.__get_com_cropped__(negativeinter_sig_image)
            negativeinter_image = self.basic_transforms(cropped_negativeinter_sig)


            sample = {'anchor' : anchor_image, 'positive' : positive_image, 'negative_intra' : negativeintra_image, 'negative_inter' : negativeinter_image,
                      'label' : label, 'writer_id' : writer_id, 'img_name' : os.path.basename(img_path)}

        elif self.mode == 'Test':
            img_path = self.test_df.iloc[index]['img_path']
            sig_image = Image.open(img_path)
            writer_id = img_path.split('/')[-2]
            label = self.test_df.iloc[index]['label']
            cropped_sig = self.__get_com_cropped__(sig_image)
            sig_image = self.basic_transforms(cropped_sig)
            sample = {'image' : sig_image, 'label' : label,
                      'writer_id' : writer_id, 'img_name' : os.path.basename(img_path)}

        return sample
        

def get_dataloader(args):
        train_dset = Writer_Dataset(args, mode='Train')
        train_loader = DataLoader(train_dset, batch_size=args.batchsize, shuffle=True, num_workers=4)
        # print('==> Train data loaded')
        
        test_dset = Writer_Dataset(args, mode='Test')
        test_loader = DataLoader(test_dset, batch_size=1, shuffle=False, num_workers=4)
        # print('==> Test data loaded')

        return train_loader , test_loader


