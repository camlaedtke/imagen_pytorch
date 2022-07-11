# imagen_pytorch

Training pipeline for Imagen, Google's Text-to-Image Neural Network, on the CocoCaptions dataset. Using @lucidrains excellent [repo](https://github.com/lucidrains/imagen-pytorch). 

Training runs are logged in Wandb: https://wandb.ai/camlaedtke/imagen?workspace=user-camlaedtke

Some running notes
- Batch size of 64-128 works. 
- Setting `max_grad_norm = 1.25` makes training more stable, but appears to considerably slow convergence and hurt performance.
- Best results have been attained with a learning rate of 1e-5. The larger the SR unet, the smaller the learning rate should be. 

Remaining questions
- Set `cond_drop_prob=0.1` or `cond_drop_prob=0.5`?
- Benefits to setting `warmup_steps` and/or `cosine_decay_max_steps`?
- Would different learning rates for the unets be better? Potentially a smaller lr for the second unet? 
