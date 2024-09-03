CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
       --master_port 1240 \
       --nproc_per_node=8 \
       --use_env main.py \
       --method vidt \
       --backbone_name swin_nano \
       --epochs 50 \
       --lr 1e-4 \
       --min-lr 1e-7 \
       --batch_size 2 \
       --num_workers 2 \
       --aux_loss True \
       --with_box_refine True \
       --distil_model vidt_small \
       --distil_model_path vkd/vidt/ckpts/vidt_small_150.pth \
       --coco_path /home/coco \
       --output_dir swin_nano