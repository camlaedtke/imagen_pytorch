import yaml
import math
import wandb
import warnings
import numpy as np
from time import time
from PIL import Image
from pathlib import Path
from flatdict import FlatDict

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from imagen_pytorch.t5 import t5_encode_text
from imagen_pytorch import Unet, Imagen, ImagenTrainer
import webdataset as wds

warnings.filterwarnings("ignore")


def padding_tensor(sequences):
    num = len(sequences)
    max_len = max([s.size(1) for s in sequences])
    out_dims = (num, max_len, 2048)
    out_tensor = sequences[0].data.new(*out_dims).fill_(0)
    for i, tensor in enumerate(sequences):
        length = tensor.size(1)
        out_tensor[i, :length, :] = tensor[:,:,0].permute(1,0)
    return out_tensor


def my_collate(batch):
    imgs = [item[0] for item in batch]
    embeds = [item[1] for item in batch]
    embeds = padding_tensor(embeds)
    return [imgs, embeds]


def format_images(display_list):
    image_list = []
    for i in range(len(display_list)):
        img = display_list[i].cpu().permute(1,2,0).numpy() * 255
        img = img.astype(np.uint8)
        image_list.append(img)
    return image_list
    
    
def log_sample_images(cfg, trainer, i):
    texts = ['dog', 'cheeseburger', 'blue car', 'red flowers in a white vase',
                 'a puppy looking anxiously at a giant donut on the table', 'the milky way galaxy in the style of monet']
    sampled_images = trainer.sample(texts, cond_scale = cfg["train"]["cond_scale"], stop_at_unet_number=i)
    image_list = format_images(sampled_images)
    images_pil = [Image.fromarray(image) for image in image_list]
    wandb.log({"Samples": [wandb.Image(image, caption=caption) for image, caption in zip(images_pil, texts)]})
    
    
def train(cfg, dataloader, trainer, epoch, i, device):
    loss_arr = []
    # cfg["dataset"]["num_images"]
    n_batches = (848950 // cfg["train"]["batch_size"])
    step_start = time()
    for step, batch in enumerate(dataloader): 
        curr_step = int(n_batches*(epoch-1) + step)
        # step_start = time()
        
        fetch_start = time()
        images, text_embeds = batch
        images = torch.stack(images, dim=0)
        images = images.to(device)
        fetch_end = time()
        fetch_time = fetch_end-fetch_start
       
        embed_start = time()
        text_embeds = text_embeds.to(device)
        embed_end = time()
        embed_time = embed_end-embed_start
       
        loss_start = time()
        loss = trainer(
            images, 
            text_embeds = text_embeds, 
            unet_number = i, 
            max_batch_size = cfg["train"]["base_unet_max_batch_size"] if i == 1 else cfg["train"]["sr_unet1_max_batch_size"]
        )
        loss_arr.append(loss)
        loss_end = time()
        loss_time = loss_end-loss_start
        
        
        update_start = time()
        trainer.update(unet_number = i)
        update_end = time()
        update_time = update_end-update_start
        
        step_end = time()
        step_time = step_end-step_start
        
        dead_time = step_time - fetch_time - embed_time - loss_time - update_time


        if step % cfg["train"]["checkpoint_rate"] == 0 and step !=0 and not math.isnan(loss): 
            log_sample_images(cfg, trainer, i)
            trainer.save(cfg["train"]["checkpoint_path"])
            print()
            
        wandb.log({
            f"Train {i} Step": curr_step,
            f"Train Loss {i}": loss, 
            f"Step time {i}": step_time, 
            f"Img load time {i}": fetch_time, 
            f"Embed load time {i}": embed_time,
            f"Loss time {i}": loss_time,
            f"Update time {i}": update_time,
            f"Dead time {i}": dead_time
        })
        print(f"\rTrain Step {step+1}/{n_batches} --- Loss: {loss:.4f} ", end='')
        step_start = time()

    return trainer, loss_arr


def run_train_loop(cfg, trainer, dataloader, device, i=1):
    for epoch in range(1, cfg["train"]["epochs"]+1):
        print(f"\nEpoch {epoch}/{cfg['train']['epochs']}")
        print(f"--- Unet {i} ---")
        start = time()
        trainer, loss_arr = train(cfg, dataloader, trainer, epoch, i, device)
        end = time()
        print(f"  Time: {(end-start)/60:.0f} min")

        
        
if __name__ == "__main__":
    
    cfg = yaml.safe_load(Path("configs\\imagen-medium-config.yaml").read_text())
    cfg_flat = dict(FlatDict(cfg, delimiter='.'))
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    
    wandb.init(project="imagen", entity="camlaedtke", config=cfg_flat)
    
    ##### INPUT PIPELINE #####    
    preproc = T.Compose([
        T.Resize(cfg["dataset"]["image_size"]),
        T.RandomHorizontalFlip(),
        T.CenterCrop(cfg["dataset"]["image_size"]),
        T.ToTensor()
    ])

    # Benchmark ...
    # Times: Step: 7.3140s, Img load: 0.0083s, Embed: 0.6073s, Loss: 6.5603s, Update: 0.1382s
    """
    cc_dataset = (
        wds.WebDataset("cc12m/{00000..00500}.tar")
        .shuffle(4800)
        .decode("pilrgb")
        .rename(image="jpg;png", caption="txt")
        .map_dict(image=preproc)
        .to_tuple("image", "caption")
    )
    
    cc_dataloader = DataLoader(
        dataset = cc_dataset, 
        batch_size = cfg["train"]["batch_size"], 
        drop_last = True,
        num_workers = 4,
        prefetch_factor = 8,
        pin_memory = True
    )
    """
    
    # Benchmark ...
    """
    batch size: 240 (24/12)
    checkpoint_rate: 100
    | NW  | PF  | Step             | Img              | Embed            | Loss             | Update           |
    | --- | --- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
    |  1  |  2  | 3.587 +/- 0.406s | 0.081 +/- 0.015s | 0.021 +/- 0.024s | 3.346 +/- 0.378s | 0.139 +/- 0.048s |
    |  1  |  4  | 3.578 +/- 0.426s | 0.077 +/- 0.004s | 0.010 +/- 0.002s | 3.356 +/- 0.403s | 0.134 +/- 0.043s |
    |  1  |  6  | 3.566 +/- 0.484s | 0.078 +/- 0.004s | 0.010 +/- 0.003s | 3.345 +/- 0.455s | 0.133 +/- 0.045s |
    |  2  |  6  | 3.598 +/- 0.385s | 0.079 +/- 0.009s | 0.012 +/- 0.012s | 3.373 +/- 0.351s | 0.133 +/- 0.042s |
    |  4  |  4  | 3.612 +/- 0.459s | 0.086 +/- 0.022s | 0.020 +/- 0.024s | 3.369 +/- 0.416s | 0.138 +/- 0.049s |
    |  4  |  2  | 3.694 +/- 1.332s | 0.082 +/- 0.014s | 0.021 +/- 0.027s | 3.462 +/- 1.296s | 0.130 +/- 0.043s |
    
    
    batch size: 256 (36/16)
    checkpoint_rate: 250
    
    | NW  | PF  | Step             | Img              | Embed            | Loss             | Update           |
    | --- | --- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
    |  1  |  4  | 3.681 +/- 0.826s | 0.083 +/- 0.007s | 0.011 +/- 0.006s | 3.422 +/- 0.803s | 0.165 +/- 0.045s |
    |  2  |  4  |  |  |  |  |  |
    
    """
    
    cc_dataset = (
        wds.WebDataset("file:E:/datasets/cc12m/{00000..00099}.tar") 
        .shuffle(cfg["dataset"]["shuffle_size"])
        .decode("pilrgb")
        .rename(image="png", embedding="emb.pyd")
        .map_dict(image=preproc)
        .to_tuple("image", "embedding")
    )
    
    cc_dataloader = DataLoader(
        dataset = cc_dataset, 
        batch_size = cfg["train"]["batch_size"], 
        drop_last = True,
        num_workers = cfg["dataset"]["num_workers"],
        prefetch_factor = cfg["dataset"]["prefetch_factor"],
        pin_memory = True,
        collate_fn = my_collate
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
    
    UNET_NUMBER = 1
    
    trainer = ImagenTrainer(
        imagen, 
        lr = cfg["train"]["lr"],
        fp16 = cfg["train"]["amp"],
        use_ema = cfg["train"]["use_ema"],
        max_grad_norm = eval(cfg["train"]["max_grad_norm"]),
        warmup_steps = eval(cfg["train"]["warmup_steps"]),
        cosine_decay_max_steps = eval(cfg["train"]["cosine_decay_max_steps"]),
        only_train_unet_number = UNET_NUMBER
    )

    ##### TRAINING #####
    # torch.backends.cudnn.benchmark = True
    
    if cfg["train"]["load_checkpoint"]:
        try:
            trainer.load(cfg["train"]["load_checkpoint_path"], strict=False, only_model=False)
            print("Loaded checkpoint")
        except: 
            pass
    
    run_train_loop(cfg, trainer, cc_dataloader, device, i=UNET_NUMBER)
    
