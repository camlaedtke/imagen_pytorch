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


def get_emb_tensor(cfg, texts, device):
    text_embeds = t5_encode_text(texts, name=cfg["model"]["text_encoder_name"], return_attn_mask=False)
    if cfg["train"]["embedding_non_blocking"]:
        return text_embeds.to(device, non_blocking=True)
    else:
        return text_embeds


def pad_tensor(sequences):
    num = len(sequences)
    max_len = max([s.size(1) for s in sequences])
    out_dims = (num, max_len, 2048)
    out_tensor = sequences[0].data.new(*out_dims).fill_(0)
    for i, tensor in enumerate(sequences):
        length = tensor.size(1)
        out_tensor[i, :length, :] = tensor[:,:,0].permute(1,0)
    return out_tensor


def pad_embeddings(batch):
    imgs = [item[0] for item in batch]
    embeds = [item[1] for item in batch]
    embeds = pad_tensor(embeds)
    return [imgs, embeds]
    
    
def format_images(display_list):
    image_list = []
    for i in range(len(display_list)):
        img = display_list[i].cpu().permute(1,2,0).numpy() * 255
        img = img.astype(np.uint8)
        image_list.append(img)
    return image_list
    
    
def get_sample_images(cfg, trainer, i):
    texts = cfg["train"]["sample_texts"]
    sampled_images = trainer.sample(texts, cond_scale = cfg["train"]["cond_scale"], stop_at_unet_number=i)
    image_list = format_images(sampled_images)
    images_pil = [Image.fromarray(image) for image in image_list]
    return images_pil, texts
    
    
def train(cfg, dataloader, trainer, epoch, i, device):
    n_batches = cfg["dataset"]["num_images"] // cfg["train"]["batch_size"]
    step_start = time()
    for step, batch in enumerate(dataloader): 
        curr_step = int(n_batches*(epoch-1) + step)
        
        fetch_start = time()
        images, texts = batch
        if cfg["dataset"]["precomputed_embeddings"]:
            images = torch.stack(images, dim=0).to(device, non_blocking=cfg["train"]["image_non_blocking"])
        else:
            images = images.to(device, non_blocking=cfg["train"]["image_non_blocking"])
        fetch_end = time()
        
        embed_start = time()
        if cfg["dataset"]["precomputed_embeddings"]:
            text_embeds = texts.to(device, non_blocking=cfg["train"]["embedding_non_blocking"])
        else:
            text_embeds = get_emb_tensor(cfg, texts, device)
        embed_end = time()
        
        loss_start = time()
        loss = trainer(
            images, 
            text_embeds = text_embeds, 
            unet_number = i, 
            max_batch_size = cfg["train"]["unet1_max_batch_size"] if i == 1 else cfg["train"]["unet2_max_batch_size"]
        )
        loss_end = time()
        
        update_start = time()
        trainer.update(unet_number = i)
        update_end = time()
        
        step_end = time()
        
        fetch_time = fetch_end-fetch_start
        embed_time = embed_end-embed_start
        loss_time = loss_end-loss_start
        update_time = update_end-update_start
        step_time = step_end-step_start
        
        dead_time = step_time - fetch_time - embed_time - loss_time - update_time
        
        if step % cfg["train"]["checkpoint_rate"] == 0 and step !=0 and not math.isnan(loss): 
            images_pil, texts = get_sample_images(cfg, trainer, i)
            wandb.log({
                f"Train {i} Step": curr_step, 
                f"Train Loss {i}": loss, 
                f"Step time {i}": step_time, 
                f"Img load time {i}": fetch_time, 
                f"Embed load time {i}": embed_time, 
                f"Loss time {i}": loss_time,
                f"Update time {i}": update_time, 
                f"Dead time {i}": dead_time,
                "Samples": [wandb.Image(image, caption=caption) for image, caption in zip(images_pil, texts)]
            })
            trainer.save(cfg["train"]["checkpoint_path"])
            print()
        else:
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
        

def run_train_loop(cfg, trainer, dataloader, device, i=1):
    for epoch in range(1, cfg["train"]["epochs"]+1):
        print(f"\nEpoch {epoch}/{cfg['train']['epochs']}")
        print(f"--- Unet {i} ---")
        start = time()
        train(cfg, dataloader, trainer, epoch, i, device)
        end = time()
        print(f"  \n\nTime: {(end-start)/3600:.2f} hours")
        
          
if __name__ == "__main__":
    
    cfg = yaml.safe_load(Path("configs\\imagen-small-config.yaml").read_text())
    cfg_flat = dict(FlatDict(cfg, delimiter='.'))
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    wandb.init(project="imagen", entity="camlaedtke", config=cfg_flat, resume=True, id="2ja6akkc")
    # wandb.init(project="imagen", entity="camlaedtke", config=cfg_flat)
    
    ##### INPUT PIPELINE #####    
    preproc = T.Compose([
        T.Resize(cfg["dataset"]["image_size"]),
        T.RandomHorizontalFlip(),
        T.CenterCrop(cfg["dataset"]["image_size"]),
        T.ToTensor()
    ])
    
    if cfg["dataset"]["precomputed_embeddings"]:
        dataset = (
            wds.WebDataset(cfg["dataset"]["dataset_path"], shardshuffle=cfg["dataset"]["shard_shuffle"]) 
            .shuffle(cfg["dataset"]["shuffle_size"], initial=cfg["dataset"]["shuffle_initial"])
            .decode("pilrgb")
            .rename(image="png", embedding="emb.pyd")
            .map_dict(image=preproc)
            .to_tuple("image", "embedding")
        )

        loader = DataLoader(
            dataset = dataset, 
            batch_size = cfg["train"]["batch_size"], 
            drop_last = cfg["dataset"]["drop_last"],
            num_workers = cfg["dataset"]["num_workers"],
            prefetch_factor = cfg["dataset"]["prefetch_factor"],
            pin_memory = cfg["dataset"]["pin_memory"],
            collate_fn = pad_embeddings
        )
    else:
        dataset = (
            wds.WebDataset(cfg["dataset"]["dataset_path"], shardshuffle=True)
            .shuffle(cfg["dataset"]["shuffle_size"])
            .decode("pilrgb")
            .rename(image="jpg;png", caption="txt")
            .map_dict(image=preproc)
            .to_tuple("image", "caption")
        )
        loader = DataLoader(
            dataset = dataset, 
            batch_size = cfg["train"]["batch_size"], 
            drop_last = cfg["dataset"]["drop_last"],
            num_workers = cfg["dataset"]["num_workers"],
            prefetch_factor = cfg["dataset"]["prefetch_factor"],
            pin_memory = cfg["dataset"]["pin_memory"],
        )
        
    
    
    ##### MODEL #####
    unet1 = Unet(
        dim = cfg["model"]["unet1"]["dim"],
        cond_dim = cfg["model"]["unet1"]["cond_dim"],
        dim_mults = cfg["model"]["unet1"]['dim_mults'], 
        num_resnet_blocks = cfg["model"]["unet1"]["num_resnet_blocks"],
        layer_attns = cfg["model"]["unet1"]['layer_attns'], 
        layer_cross_attns = cfg["model"]["unet1"]['layer_cross_attns'], 
        attn_heads = cfg["model"]["unet1"]["attn_heads"],
        ff_mult = cfg["model"]["unet1"]["ff_mult"],
        memory_efficient = cfg["model"]["unet1"]["memory_efficient"],
        dropout = cfg["model"]["unet1"]["dropout"],
        cosine_sim_attn = cfg["model"]["unet1"]["cosine_sim_attn"],
        use_linear_attn = cfg["model"]["unet1"]["use_linear_attn"]
    )

    unet2 = Unet(
        dim = cfg["model"]["unet2"]["dim"],
        cond_dim = cfg["model"]["unet2"]["cond_dim"],
        dim_mults = cfg["model"]["unet2"]["dim_mults"], 
        num_resnet_blocks = cfg["model"]["unet2"]["num_resnet_blocks"], 
        layer_attns = cfg["model"]["unet2"]["layer_attns"],
        layer_cross_attns = cfg["model"]["unet2"]["layer_cross_attns"], 
        attn_heads = cfg["model"]["unet2"]["attn_heads"],
        ff_mult = cfg["model"]["unet2"]["ff_mult"],
        memory_efficient = cfg["model"]["unet2"]["memory_efficient"],
        dropout = cfg["model"]["unet2"]["dropout"],
        cosine_sim_attn = cfg["model"]["unet2"]["cosine_sim_attn"],
        use_linear_attn = cfg["model"]["unet2"]["use_linear_attn"]
    )

    imagen = Imagen(
        unets = (unet1, unet2),
        text_encoder_name = cfg["model"]["text_encoder_name"], 
        image_sizes = cfg["model"]["image_sizes"], 
        cond_drop_prob = cfg["model"]["cond_drop_prob"],
        timesteps = cfg["model"]["timesteps"],
    ).cuda()
    
    trainer = ImagenTrainer(
        imagen, 
        lr = cfg["train"]["lr"],
        fp16 = cfg["train"]["amp"],
        use_ema = cfg["train"]["use_ema"],
        max_grad_norm = eval(cfg["train"]["max_grad_norm"]),
        warmup_steps = eval(cfg["train"]["warmup_steps"]),
        cosine_decay_max_steps = eval(cfg["train"]["cosine_decay_max_steps"]),
        only_train_unet_number = cfg["train"]["unet_number"]
    )
    

    ##### TRAINING #####
    if cfg["train"]["cudnn_benchmark"]:
        torch.backends.cudnn.benchmark = True
    
    if cfg["train"]["load_checkpoint"]:
        trainer.load(
            cfg["train"]["load_checkpoint_path"], 
            strict=cfg["train"]["checkpoint_strict"], 
            only_model=cfg["train"]["checkpoint_model_only"],
            noop_if_not_exist=True
        )
        
    run_train_loop(cfg, trainer, loader, device, i=cfg["train"]["unet_number"])
    
