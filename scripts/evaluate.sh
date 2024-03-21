CUDA_VISIBLE_DEVICES=2,3 \
OMP_NUM_THREADS=4,5 \
python -m torch.distributed.launch --master_port 1235 --nproc_per_node=2 main_distributed.py \
    --teacher regnety_160 \
    --student-model deit_ti_distilled \
    --eval-student \
    --student-ckpt /path/to/deit_tiny/latest.pth.tar \
    --batch-size 64 \
    --data-path /path/to/imagenet2012 \
    --world_size 2