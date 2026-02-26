

MODEL=intermediate-model/vae

./create_imgcache_sdxl.py --vae --model $MODEL --writepreview --cpu --data_root testimgs
