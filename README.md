# imagen_pytorch

Training pipeline for Imagen, Google's Text-to-Image Neural Network, on the CocoCaptions, Conceptual 3M, and Conceptual 12M datasets. Using @lucidrains excellent [repo](https://github.com/lucidrains/imagen-pytorch). 

Training runs are logged in Wandb: https://wandb.ai/camlaedtke/imagen?workspace=user-camlaedtke

#### CocoCaptions Dataset notes
- The CocoCaptions dataset contains five different captions for each image. 
- The dataloader in `utils/data_utils.py` randomly selects one of the five captions during training to artificially increase the size of our dataset. 
- The script `create_dataset_info.ipynb` creates a pandas dataframe of image/caption information that helps simplify the data input pipeline. 

#### Conceptual 12M
- Downloaded with [img2dataset](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc12m.md)

`curl.exe --output cc12m.tsv --url https://storage.googleapis.com/conceptual_12m/cc12m.tsv`

`sed -i "1s/^/url\tcaption\n/" cc12m.tsv`

`img2dataset --url_list cc12m.tsv --input_format "tsv"\
         --url_col "url" --caption_col "caption" --output_format webdataset\
           --output_folder cc12m --processes_count 8 --thread_count 32 --image_size 256\
             --enable_wandb True`


#### Some running notes
- Batch size of 64-128 is good. 
- Setting `max_grad_norm = 1.25` makes training more stable, but appears to considerably slow convergence and hurt performance.
- Best results have been attained with a learning rate of 1e-5. The larger the SR unet, the smaller the learning rate should be. And a smaller learning rate for the base unet may be needed for very large text embedding models like google/t5-v1_1-xl. 