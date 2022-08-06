# imagen_pytorch

Training pipeline for Imagen, Google's Text-to-Image Neural Network, on the Conceptual 12M dataset. Using Phil Wang's excellent [repo](https://github.com/lucidrains/imagen-pytorch). 

Training runs are logged in Wandb: https://wandb.ai/camlaedtke/imagen?workspace=user-camlaedtke
 

#### Conceptual 12M

Downloaded with [img2dataset](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc12m.md)

On Windows ...
```bash
curl.exe --output cc12m.tsv --url https://storage.googleapis.com/conceptual_12m/cc12m.tsv
```
```bash
sed -i "1s/^/url\tcaption\n/" cc12m.tsv
```
```bash
img2dataset --url_list cc12m.tsv --input_format "tsv"\
         --url_col "url" --caption_col "caption" --output_format webdataset\
           --output_folder cc12m --processes_count 8 --thread_count 32 --image_size 256\
             --enable_wandb True
```


#### Some running notes
- Batch size of 64-512 seems to be good.
- Setting `max_grad_norm = 1.25` makes training more stable, but appears to considerably slow convergence and hurt performance.
- Best results have been attained with a learning rate of around 1.5e-5 when combined with batch size of 256.