import yaml
import wandb
import torch
import warnings
import numpy as np
from pathlib import Path
from flatdict import FlatDict
from torchvision import datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader
from imagen_pytorch import Unet, Imagen, ImagenTrainer, ElucidatedImagen
from utils.transformations import ComposeDouble, FunctionWrapperDouble, select_random_label
from utils.train_utils import run_train_loop
from utils.data_utils import CocoDataset

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    
    cfg = yaml.safe_load(Path("configs\\imagen-medium-config.yaml").read_text())
    cfg_flat = dict(FlatDict(cfg, delimiter='.'))
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    
    wandb.init(project="imagen", entity="camlaedtke", config=cfg_flat)
    
    ##### INPUT PIPELINE #####
    coco_dataset = CocoDataset(cfg)
    
    coco_dataloader = DataLoader(
        dataset = coco_dataset, 
        batch_size = cfg["train"]["batch_size"], 
        shuffle = True,
        drop_last = True,
        num_workers = 4,
        prefetch_factor = 8,
        pin_memory = True
    )

    ##### MODEL #####
    BaseUnet = Unet(
        dim = cfg["model"]["base_unet"]["dim"],
        text_embed_dim = cfg["model"]["text_embed_dim"],
        cond_dim = cfg["model"]["base_unet"]["cond_dim"],
        dim_mults = cfg["model"]["base_unet"]['dim_mults'], 
        num_resnet_blocks = cfg["model"]["base_unet"]["num_resnet_blocks"],
        layer_attns = cfg["model"]["base_unet"]['layer_attns'], 
        layer_cross_attns = cfg["model"]["base_unet"]['layer_cross_attns'], 
        attn_heads = cfg["model"]["base_unet"]["attn_heads"],
        ff_mult = cfg["model"]["base_unet"]["ff_mult"],
        memory_efficient = cfg["model"]["base_unet"]["memory_efficient"],
        dropout = cfg["model"]["base_unet"]["dropout"]
    )


    SRUnet = Unet(
        dim = cfg["model"]["sr_unet1"]["dim"],
        text_embed_dim = cfg["model"]["text_embed_dim"],
        cond_dim = cfg["model"]["sr_unet1"]["cond_dim"],
        dim_mults = cfg["model"]["sr_unet1"]["dim_mults"], 
        num_resnet_blocks = cfg["model"]["sr_unet1"]["num_resnet_blocks"], 
        layer_attns = cfg["model"]["sr_unet1"]["layer_attns"],
        layer_cross_attns = cfg["model"]["sr_unet1"]["layer_cross_attns"], 
        attn_heads = cfg["model"]["sr_unet1"]["attn_heads"],
        ff_mult = cfg["model"]["sr_unet1"]["ff_mult"],
        memory_efficient = cfg["model"]["sr_unet1"]["memory_efficient"],
        dropout = cfg["model"]["sr_unet1"]["dropout"]
    )

    imagen = ElucidatedImagen(
        unets = (BaseUnet, SRUnet),
        image_sizes = cfg["model"]["image_sizes"],
        text_embed_dim = cfg["model"]["text_embed_dim"],
        text_encoder_name = cfg["model"]["text_encoder_name"],
        cond_drop_prob = cfg["model"]["cond_drop_prob"],
        num_sample_steps = cfg["model"]["num_sample_steps"], 
        sigma_min = cfg["model"]["sigma_min"],          
        sigma_max = cfg["model"]["sigma_max"],       
        sigma_data = cfg["model"]["sigma_delta"],            
        rho = cfg["model"]["rho"],                     
        P_mean = cfg["model"]["P_mean"],               
        P_std = cfg["model"]["P_std"],                 
        S_churn = cfg["model"]["S_churn"],                
        S_tmin = cfg["model"]["S_tmin"],
        S_tmax = cfg["model"]["S_tmax"],
        S_noise = cfg["model"]["S_noise"],
    ).cuda()
    

    ##### TRAINING #####
    trainer = ImagenTrainer(
        imagen, 
        lr = cfg["train"]["lr"],
        amp = cfg["train"]["amp"],
        use_ema = cfg["train"]["use_ema"],
        warmup_steps = eval(cfg["train"]["warmup_steps"]),
        cosine_decay_max_steps = eval(cfg["train"]["cosine_decay_max_steps"]),
    )
    
#     try:
#         trainer.load(cfg["train"]["checkpoint_path"], strict=False, only_model=True)
#         print("Loaded checkpoint")
#     except: 
#         pass
    
    # torch.backends.cudnn.benchmark = True
    
    run_train_loop(cfg, trainer, coco_dataloader, device)
    
