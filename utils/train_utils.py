import math
import wandb
import torch
import numpy as np
from time import time
from PIL import Image
import matplotlib.pyplot as plt
from imagen_pytorch.t5 import t5_encode_text



def get_emb_tensor(cfg, targets, device):
    text_embeds, text_masks = t5_encode_text(targets, name = cfg["model"]["text_encoder_name"])
    text_embeds, text_masks = map(lambda t: t.to(device), (text_embeds, text_masks))
    return text_embeds, text_masks



def display_images(display_list):
    image_list = []
    plt.figure(figsize=(10, 10), dpi=150)
    for i in range(len(display_list)):
        img = display_list[i].cpu().numpy() * 255
        img = np.swapaxes(img,0,2).astype(np.uint8)
        image_list.append(img)
        plt.subplot(1, len(display_list), i+1)
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    return image_list


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
        
        
# def print_epoch_stats(e_time, train_loss_arr, valid_loss_arr):
#     print(f"   Time: {e_time:.0f} min, "\
#           f"Train Loss: {np.mean(train_loss_arr, where=np.isnan(train_loss_arr)==False):.4f}, "\
#           f"Valid Loss: {np.mean(valid_loss_arr, where=np.isnan(valid_loss_arr)==False):.4f}")

def print_epoch_stats(e_time, train_loss_arr):
    print(f"   Time: {e_time:.0f} min, "\
          f"Train Loss: {np.mean(train_loss_arr, where=np.isnan(train_loss_arr)==False):.4f}")
    
    
def train(cfg, train_dataloader, trainer, epoch, i, device):
    train_loss_arr = []
    fetch_times = []; embed_times = []; loss_times = []; update_times = []; step_times = []
    for step, batch in enumerate(train_dataloader):
        step_start = time()
        
        fetch_start = time()
        images, texts = batch
        images = images.to(device)
        fetch_end = time()
        fetch_times.append(fetch_end-fetch_start)
        
        embed_start = time()
        text_embeds, text_masks = get_emb_tensor(cfg, texts, device)
        embed_end = time()
        embed_times.append(embed_end-embed_start)

        loss_start = time()
        loss = trainer(
            images, 
            text_embeds = text_embeds, 
            text_masks = text_masks, 
            unet_number = i, 
            max_batch_size=cfg["train"]["base_unet_max_batch_size"] if i==1 else cfg["train"]["sr_unet1_max_batch_size"]
        )
        loss_end = time()
        loss_times.append(loss_end-loss_start)
        
        update_start = time()
        trainer.update(unet_number = i)
        update_end = time()
        update_times.append(update_end-update_start)
        
        step_end = time()
        step_times.append(step_end-step_start)
        
        
        train_loss_arr.append(loss)
        save_checkpoint(cfg, step, loss, trainer)
        
        curr_step = int(len(train_dataloader)*(epoch-1) + step)
        wandb.log({f"Train Loss {i}": loss, f"Train {i} Step": curr_step})
        print(f"\r   Train Step {step+1}/{len(train_dataloader)}, Train Loss: {loss:.4f}", end='')
        
    
    step_time = np.mean(step_times)
    fetch_time = np.mean(fetch_times)
    embed_time = np.mean(embed_times)
    loss_time = np.mean(loss_times)
    update_time = np.mean(update_times)
    print()
    print(f"      Step: {step_time:.4f}s, Img load: {fetch_time:.4f}s, Embed: {embed_time:.4f}s, "\
          f"Loss: {loss_time:.4f}s, Update: {update_time:.4f}s")
    return trainer, train_loss_arr


def validate(cfg, valid_dataloader, trainer, epoch, i, device):
    valid_loss_arr = []
    for step, batch in enumerate(valid_dataloader):
        images, texts = batch
        images = images.to(device)
        text_embeds, text_masks = get_emb_tensor(cfg, texts, device)

        loss = trainer(
            images, 
            text_embeds = text_embeds, 
            text_masks = text_masks, 
            unet_number = i, 
            max_batch_size=cfg["train"]["base_unet_max_batch_size"] if i==1 else cfg["train"]["sr_unet1_max_batch_size"]
        )
        valid_loss_arr.append(loss)
        
        curr_step = int(len(valid_dataloader)*(epoch-1) + step)
        wandb.log({f"Validation Loss {i}": loss, f"Valid {i} Step": curr_step})
        print(f"\r   Valid Step {step+1}/{len(valid_dataloader)}, Valid Loss: {loss:.4f}", end='')
    print()
    return valid_loss_arr



def run_train_loop(cfg, trainer, train_dataloader, device):
    
    for epoch in range(1, cfg["train"]["epochs"]+1):
        print(f"\nEpoch {epoch}/{cfg['train']['epochs']}")
        
        for i in (1,2):
            
            print(f"--- Unet {i} ---")
            start = time()

            trainer, train_loss_arr = train(cfg, train_dataloader, trainer, epoch, i, device)

            end = time()
            e_time = (end-start)/60 

            print_epoch_stats(e_time, train_loss_arr)
            if not math.isnan(train_loss_arr[-1]): 
                trainer.save(cfg["train"]["checkpoint_path"])
            
        texts = [
            'red flowers in a white vase',
            'a puppy looking anxiously at a giant donut on the table',
            'the milky way galaxy in the style of monet'
        ]
        sampled_images = trainer.sample(texts, cond_scale = cfg["train"]["cond_scale"])
        # image_list = display_images(sampled_images)
        image_list = format_images(sampled_images)
        images_pil = [Image.fromarray(image) for image in image_list]
        wandb.log({"Samples": [wandb.Image(image) for image in images_pil], "Epoch": epoch})