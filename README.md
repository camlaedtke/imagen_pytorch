# imagen_pytorch

Training pipeline for Imagen, Google's Text-to-Image Neural Network, on the CocoCaptions dataset. Using @lucidrains excellent [repo](https://github.com/lucidrains/imagen-pytorch). 

Training runs are logged in Wandb: https://wandb.ai/camlaedtke/imagen?workspace=user-camlaedtke

Some running notes
- Batch size of 192 seems larger than necessary. 128 should be fine. Not sure about 64. 
- Learning rate between 2e-5 and 5e-5 seems good. 
- Setting `max_grad_norm = 1.25` helps a lot

Remaining questions
- Set `cond_drop_prob = 0.1` or `cond_drop_prob=0.5`?
- Benefits to setting `warmup_steps` and/or `cosine_decay_max_steps`?
- Set `dropout` to default of 0.1, or change to 0.2?
