CUDA_LAUNCH_BLOCKING=1
#python3 main.py --train electronics --save_dir ./data/electronics

# path-to-model: ./data/[DATASET-NAME]/[N-LAYERS]_[HIDDEN-SIZE]_[BATCH-SIZE]
python3 main.py --test electronics --save_dir ./data/electronics --model ./data/electronics/model/2_512_1024/405_aspect_planning.tar
