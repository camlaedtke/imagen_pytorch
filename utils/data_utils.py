import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils import data
from torchvision import transforms as T
from utils.transformations import ComposeDouble, FunctionWrapperDouble, select_random_label, select_fixed_label
import webdataset as wds

class CocoDataset(data.Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.info_df = pd.read_pickle(cfg["dataset"]["info_file"])
        self.image_paths = self.info_df["file_path"].values.tolist()
        self.captions = self.info_df["caption"].values.tolist()
        
        self.transform = ComposeDouble([
            FunctionWrapperDouble(T.Resize(cfg["dataset"]["image_size"]), input=True, target=False),
            FunctionWrapperDouble(T.RandomHorizontalFlip(), input=True, target=False),
            FunctionWrapperDouble(T.CenterCrop(cfg["dataset"]["image_size"]), input=True, target=False),
            FunctionWrapperDouble(T.ToTensor(), input=True, target=False),
            FunctionWrapperDouble(select_random_label, input=False, target=True),
        ])

   
    def __len__(self):
        return len(self.image_paths)

    
    def __getitem__(self, index):
        """Modify to return tuple ('images', 'text_embeds', 'text_masks')"""
        image = Image.open(self.image_paths[index]).convert("RGB")
        captions = self.captions[index]
        return self.transform(image, captions)
    
    
    
class CC3MDataset(data.Dataset):
    # TODO: MAKE SURE IMAGES ARE IN FLOAT32!!
    def __init__(self, cfg):
        super().__init__()
        
        self.transform = T.Compose([
            T.Resize(cfg["dataset"]["image_size"]),
            T.RandomHorizontalFlip(),
            T.CenterCrop(cfg["dataset"]["image_size"]),
            T.ToTensor()
        ])
        
        self.dataset = (
            wds.WebDataset(cfg["dataset"]["data_path"])
            .shuffle(10000)
            .decode("pil")
            .rename(image="jpg;png", caption="txt")
            .map_dict(image=self.transform)
            .to_tuple("image", "caption")
        )

   
    # def __len__(self):
    #     return len(self.image_paths)

    
    def __getitem__(self, index):
        image, caption = next(iter(self.dataset))
        
        return image, caption