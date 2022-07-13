import yaml
import math
import wandb
import torch
import warnings
import numpy as np
from time import time
from PIL import Image
from pathlib import Path
from flatdict import FlatDict
from torchvision import datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader
from imagen_pytorch.t5 import t5_encode_text
from imagen_pytorch import Unet, Imagen, ImagenTrainer
# from utils.train_utils import run_train_loop
from utils.data_utils import CocoDataset

warnings.filterwarnings("ignore")


def get_emb_tensor(cfg, targets, device):
    text_embeds, text_masks = t5_encode_text(targets, name=cfg["model"]["text_encoder_name"], return_attn_mask=True)
    text_embeds, text_masks = map(lambda t: t.to(device), (text_embeds, text_masks))
    return text_embeds, text_masks


def format_images(display_list):
    image_list = []
    for i in range(len(display_list)):
        img = display_list[i].cpu().numpy() * 255
        img = np.swapaxes(img,0,2).astype(np.uint8)
        image_list.append(img)
    return image_list


def save_checkpoint(cfg, step, loss, trainer):
    if step % cfg["train"]["checkpoint_rate"] == 0 and step !=0 and not math.isnan(loss): 
        trainer.save(cfg["train"]["checkpoint_path"])
           
        
def print_epoch_stats(e_time, loss_arr):
    print(f"  Time: {e_time:.0f} min, Train Loss: {np.mean(loss_arr, where=np.isnan(loss_arr)==False):.4f}")
    
    
def train(cfg, dataloader, trainer, epoch, i, device):
    loss_arr = []
    for step, batch in enumerate(dataloader): 
        images, texts = batch
        images = images.to(device)
       
        text_embeds, text_masks = get_emb_tensor(cfg, texts, device)
       
        loss = trainer(
            images, 
            text_embeds = text_embeds, 
            text_masks = text_masks, 
            unet_number = i, 
            max_batch_size = cfg["train"]["base_unet_max_batch_size"] if i == 1 else cfg["train"]["sr_unet1_max_batch_size"]
        )
        loss_arr.append(loss)
        trainer.update(unet_number = i)
        save_checkpoint(cfg, step, loss, trainer)

        curr_step = int(len(dataloader)*(epoch-1) + step)
        wandb.log({f"Train Loss {i}": loss, f"Train {i} Step": curr_step})
        print(f"\rTrain Step {step+1}/{len(dataloader)} --- Loss: {loss:.4f} ", end='')
    
    return trainer, loss_arr


def run_train_loop(cfg, trainer, dataloader, device):
        
    for epoch in range(1, cfg["train"]["epochs"]+1):
        print(f"\nEpoch {epoch}/{cfg['train']['epochs']}")
        
        for i in (1,2):
            print(f"--- Unet {i} ---")

            start = time()

            trainer, loss_arr = train(cfg, dataloader, trainer, epoch, i, device)

            end = time()
            e_time = (end-start)/60 

            print_epoch_stats(e_time, loss_arr)
            if not math.isnan(loss_arr[-1]): 
                trainer.save(cfg["train"]["checkpoint_path"])
            
        texts = [
            'dog',
            'cheeseburger',
            'blue car',
            'red flowers in a white vase',
            'a puppy looking anxiously at a giant donut on the table',
            'the milky way galaxy in the style of monet'
        ]
        sampled_images = trainer.sample(texts, cond_scale = cfg["train"]["cond_scale"])
        image_list = format_images(sampled_images)
        images_pil = [Image.fromarray(image) for image in image_list]
        wandb.log({"Samples": [wandb.Image(image, caption=caption) for image, caption in zip(images_pil, texts)], "Epoch": epoch})
        
        


if __name__ == "__main__":
    
    cfg = yaml.safe_load(Path("configs\\imagen-large-config.yaml").read_text())
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


    imagen = Imagen(
        unets = (BaseUnet, SRUnet),
        text_encoder_name = cfg["model"]["text_encoder_name"], 
        image_sizes = cfg["model"]["image_sizes"], 
        cond_drop_prob = cfg["model"]["cond_drop_prob"],
        timesteps = cfg["model"]["timesteps"],
    ).cuda()
    
    trainer = ImagenTrainer(
        imagen, 
        lr = cfg["train"]["lr"],
        amp = cfg["train"]["amp"],
        use_ema = cfg["train"]["use_ema"],
        max_grad_norm = eval(cfg["train"]["max_grad_norm"]),
        warmup_steps = eval(cfg["train"]["warmup_steps"]),
        cosine_decay_max_steps = eval(cfg["train"]["cosine_decay_max_steps"]),
    )

    ##### TRAINING #####
    # torch.backends.cudnn.benchmark = True
    
    if cfg["train"]["load_checkpoint"]:
        try:
            trainer.load(cfg["train"]["load_checkpoint_path"], strict=False, only_model=False)
            print("Loaded checkpoint")
        except: 
            pass
    
    run_train_loop(cfg, trainer, coco_dataloader, device)
    
