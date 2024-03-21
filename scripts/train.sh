CUDA_VISIBLE_DEVICES=1,2 \
OMP_NUM_THREADS=4 \
python -m torch.distributed.launch --master_port 1235 --nproc_per_node=2 main_distributed.py \
    --teacher regnety_160 \
    --student-model deit_ti_distilled \
    --batch-size 512 \
    --teacher-path ckpts/regnety_160-a5fe301d.pth \
    --data-path /path/to/imagenet2012 \
    --output_dir output_deit_ti/