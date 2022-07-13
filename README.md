# imagen_pytorch

Training pipeline for Imagen, Google's Text-to-Image Neural Network, on the CocoCaptions dataset. Using @lucidrains excellent [repo](https://github.com/lucidrains/imagen-pytorch). Using `imagen-pytorch` version 0.21.1 because recent changes have made single GPU training a bit more difficult. 

Training runs are logged in Wandb: https://wandb.ai/camlaedtke/imagen?workspace=user-camlaedtke

#### Dataset notes
- The CocoCaptions dataset contains five different captions for each image. 
- The dataloader in `utils/data_utils.py` randomly selects one of the five captions during training to artificially increase the size of our dataset. 
- The script `create_dataset_info.ipynb` creates a pandas dataframe of image/caption information that helps simplify the data input pipeline. 

#### Some running notes
- Batch size of 64-128 is good. 
- Setting `max_grad_norm = 1.25` makes training more stable, but appears to considerably slow convergence and hurt performance.
- Best results have been attained with a learning rate of 1e-5. The larger the SR unet, the smaller the learning rate should be. And a smaller learning rate for the base unet may be needed for very large text embedding models like google/t5-v1_1-xl. 
