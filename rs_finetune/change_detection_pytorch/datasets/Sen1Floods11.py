import os
import torch
import json
import random
import rasterio
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F

# random.seed(42)  
from tqdm import tqdm
import numpy as np
from storage_paths import datasets_path


STATS = {
    'mean': {
        'B01': 1626.91600224,
        'B02': 1396.03470631,
        'B03': 1364.06118417,
        'B04': 1218.22847919,
        'B05': 1466.07290663,
        'B06': 2386.90297537,
        'B07': 2845.61256277,
        'B08': 2622.95796892,
        'B8A': 3077.48221481,
        'B09': 486.87436782,
        'B10': 63.77861008,
        'B11': 2030.64763024,
        'B12': 1179.16607221,
        'VV': -10.184408,
        'VH': -16.895273,
        },
    'std': {
        'B01': 700.17133846,
        'B02': 739.09452682,
        'B03': 735.2482388,
        'B04': 864.936695,
        'B05': 776.8803358,
        'B06': 921.36834309,
        'B07': 1084.37346097,
        'B08': 1022.63418007,
        'B8A': 1196.44255318,
        'B09': 336.61105431,        
        'B10': 143.99923282,       
        'B11': 980.87061347,
        'B12': 764.60836557,
        'VV': 4.255339,
        'VH': 5.290568,
        }
}


WAVES = {
    "B01": 0.443,
    "B02": 0.493,
    "B03": 0.56,
    "B04": 0.665,
    "B05": 0.704,
    "B06": 0.74,
    "B07": 0.783,
    "B08": 0.842,
    "B8A": 0.865,
    "B09": 0.945,
    "B10": 1.375,
    "B11": 1.61,
    "B12": 2.19,
    'VV': 3.5,
    'VH': 4.0
}


class Sen1Floods11(Dataset):
    def __init__(self,
                 bands,
                 img_size=224,
                 metadata_path=None,
                 root_path=None,
                 split_file_path=None,
                 split = 'train',
                 limited_label=1.0,
                 limited_label_strategy='stratified',
                 fill_zeros=False,
                ):
        if metadata_path is None:
            metadata_path = datasets_path("x-sen1floods11", "v1.1", "catalog")
        if root_path is None:
            root_path = datasets_path("x-sen1floods11")
        if split_file_path is None:
            split_file_path = datasets_path("x-sen1floods11", "v1.1", "splits", "flood_handlabeled")
        
        self.classes = ['Not Water', 'Water']
        self.ignore_index = -1
        
        self.split_file_path = split_file_path
        self.split = split
        self.root_path = root_path
        self.img_size = img_size
        self.metadata_path = metadata_path
        self.bands = bands
        self.num_classes = 2
        self.fill_zeros = fill_zeros

        split_csv_names = {"train": "flood_train_data.csv", "val": "flood_valid_data.csv", "test": "flood_test_data.csv"}
        split_file = os.path.join(self.split_file_path, split_csv_names[split])

        data_root = os.path.join(
            root_path, "v1.1", "data/flood_events/HandLabeled/"
        )
        
        with open(split_file) as f:
            file_list = f.readlines()

        rows = [ln.rstrip().split(",") for ln in file_list if ln.strip()]

        self.s1_image_list = []
        self.s2_image_list = []
        self.target_list = []
        for parts in rows:
            s1_name = parts[0].strip()
            label_name = parts[2].strip() if len(parts) >= 3 else parts[1].strip()
            self.s1_image_list.append(os.path.join(data_root, "S1Hand", s1_name))
            self.s2_image_list.append(
                os.path.join(data_root, "S2Hand", s1_name.replace("S1Hand", "S2Hand"))
            )
            self.target_list.append(os.path.join(data_root, "LabelHand", label_name))


        self.indices = list(range(len(self.s1_image_list)))


    def __len__(self):
        # return len(self.s1_image_list)
        return len(self.indices)



    def __getitem__(self, index):

        with rasterio.open(self.s2_image_list[index]) as src:
            s2_image = src.read()

        with rasterio.open(self.s1_image_list[index]) as src:
            s1_image = src.read()
            s1_image = np.nan_to_num(s1_image)

        with rasterio.open(self.target_list[index]) as src:
            target = src.read(1)
        
        s2_image = torch.from_numpy(s2_image).float()
        s1_image = torch.from_numpy(s1_image).float()

        target = torch.from_numpy(target).long()

        band_index_map = {
            'B01': s2_image[0],
            'B02': s2_image[1],
            'B03': s2_image[2],
            'B04': s2_image[3],
            'B05': s2_image[4],
            'B06': s2_image[5],
            'B07': s2_image[6],
            'B08': s2_image[7],
            'B8A': s2_image[8],
            'B09': s2_image[9],
            'B10': s2_image[10],
            'B11': s2_image[11],
            'B12': s2_image[12],
            'VV': s1_image[0],
            'VH': s1_image[1],
        }

        img = []
        # for b in self.bands:
        #     ch = (band_index_map[b] - STATS['mean'][b]) / STATS['std'][b]
        #     ch = F.resize(ch.unsqueeze(0), [self.img_size, self.img_size],
        #             interpolation=transforms.InterpolationMode.BILINEAR,
        #         )
        #     img.append(ch.squeeze(0))


        if self.split == 'train':
            i, j, h, w = transforms.RandomCrop.get_params(target, (256, 256))
            for b in self.bands:
                ch = (band_index_map[b] - STATS['mean'][b]) / STATS['std'][b]
                ch = np.clip(ch, -3, 3).unsqueeze(0)
                ch = F.crop(ch, i, j, h, w)
                ch = F.resize(ch.unsqueeze(0), [self.img_size, self.img_size],
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    )
                img.append(ch.squeeze(0))
            target = F.crop(target, i, j, h, w)
        else:
            for b in self.bands:
                ch = ((band_index_map[b] - STATS['mean'][b]) / STATS['std'][b])
                ch = np.clip(ch, -3, 3).unsqueeze(0)
                ch = F.resize(ch, [self.img_size, self.img_size],
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    )
                img.append(ch.squeeze(0))

        target = F.resize(target.unsqueeze(0), [self.img_size, self.img_size],
                            interpolation=transforms.InterpolationMode.NEAREST,
                            ).squeeze(0)

        image = torch.stack(img, axis=0) 

        if self.fill_zeros and len(image) < 3:
            zero_band = torch.zeros((self.img_size, self.img_size), dtype=torch.float32)
            image = torch.cat([image, zero_band.unsqueeze(0)], dim=0)
        
        if self.split == 'train':
            if random.random() > 0.5:
                image = F.hflip(image)
                target = F.hflip(target)
            if random.random() > 0.5:
                image = F.vflip(image)
                target = F.vflip(target)

        filename = self.s1_image_list[index].strip().rsplit('/', 1)[-1].rsplit('.', 1)[0]
        base_id = filename.replace("_S1Hand", "")
        meta_json = os.path.join(
            self.metadata_path, "sen1floods11_hand_labeled_source", base_id, f"{base_id}.json"
        )
        with open(meta_json, 'r') as file:
            metadata = json.load(file)
        metadata.update({'waves': [WAVES[b] for b in self.bands if b in self.bands]})
    
        return image, target, filename, metadata