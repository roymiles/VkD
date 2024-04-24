# VkD : Improving Knowledge Distillation using Orthogonal Projections
This is the official implementation of CVPR24 paper "VkD : Improving Knowledge Distillation using Orthogonal Projections"
Code for the CVPR24 paper:

```text
"VkD : Improving Knowledge Distillation using Orthogonal Projections".
Roy Miles, Ismail Elezi, Jiankang Deng. CVPR 2024.
```
[[Paper on arxiv](https://arxiv.org/abs/2403.06213)]

## Abstract

Knowledge distillation is an effective method for training small and efficient deep learning models. However, the efficacy of a single method can degenerate when transferring to other tasks, modalities, or even other architectures. To address this limitation, we propose a novel constrained feature distillation method. This method is derived from a small set of core principles, which results in two emerging components: an orthogonal projection and a task-specific normalisation. Equipped with both of these components, our transformer models can outperform all previous methods on ImageNet and reach up to a 4.4% relative improvement over the previous state-of-the-art methods. To further demonstrate the generality of our method, we apply it to object detection and image generation, whereby we obtain consistent and substantial performance improvements over state-of-the-art.

## Structure

We provide the training code and weights for the two main sets of experiments corresponding for DeIT and ViDT. These are based on the repositories [co-advise](https://github.com/OliverRensu/co-advise) and [ViDT](https://github.com/naver-ai/vidt/tree/main) respectively.

```
.
├── scripts/               # Example scripts for training and evaluation.
├── vidt/                  # Object detection with ViDT models.       
├── main_distributed.py    # Model, optimizer, data loader setup.
├── engine.py              # Train and evaluation loops.
```

## Image Classification
Training data-efficient image transformers using orthogonal knowledge distillation. 
All models are trained for 300 epochs with a CNN based teacher.
The dataset should be downloaded and copied to `/data/imagenet2012`.
The pre-trained regnet-y teacher weights (original DeiT) can be found [here](https://drive.google.com/drive/folders/1vtNwxbHHvJnbFMwjD8oyjjW0lyHqfuRy?usp=sharing). Currently the code is setup to use automatically download the weights from Hugging Face for conveniance. However, we did find the original DeiT teacher weights to be much more effective (+0.5%). We have recently re-ran the experiments for DeiT-Ti and improved the performance to 79.2%. This is with a single GPU and a batch size of 512. Multi-GPU training will likely require changing some learning rate/scheduling hyperparameters to be the most effective.

### Pretrained Models

We provide the pre-distilled model weights and logs for the DeIT experiments.

| Model | Top-1 Acc | Top-5 Acc | Params | Weights / Log |
| --- | --- | --- | --- | --- |
| tiny | 78.3 | 94.1 | 6M | [link](https://drive.google.com/drive/folders/1G8jv2If3lpFlnfnmGUFrrh6Lxhzdc6y-?usp=drive_link) |
| tiny (single gpu) | 79.2 | 94.6 | 6M | [link](https://drive.google.com/drive/folders/1L4tgMthQVdRv1SD9DV_9xyLhAHBPMtuk) |
| small | 82.3 | 96.0 | 22M| [link](https://drive.google.com/drive/folders/1aBYO8BhJMCKij4GxZXXaxZkaT6VVL1-0?usp=sharing) |
| small (single-gpu) | 82.9 | 96.2 | 22M| [link](https://drive.google.com/drive/folders/1up5w3V4PFwobkiQs_4tVnDw7_2RdvXr2)

### Evaluation

<details>
<summary>Run this command to evaluate the <code>DeiT-Ti</code> model in the paper :</summary>
<pre><code>CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
python -m torch.distributed.launch \
       --master_port 1234 \
       --nproc_per_node=1 \
       main_distributed.py \
       --teacher regnety_160 \
       --student-model deit_ti_distilled \
       --eval-student \
       --student-ckpt /ckpts/deit_tiny/latest.pth.tar \
       --batch-size 512 \
       --data-path /data/imagenet2012</code></pre>
</details>

<details>
<summary>Run this command to evaluate the <code>DeiT-S</code> model in the paper :</summary>
<pre><code>CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
python -m torch.distributed.launch \
       --master_port 1235 \
       --nproc_per_node=1 \
       main_distributed.py \
       --teacher regnety_160 \
       --student-model deit_s_distilled \
       --eval-student \
       --student-ckpt /ckpts/deit_small/latest.pth.tar \
       --batch-size 512 \
       --data-path /data/imagenet2012</code></pre>
</details>


### Training
We have tested training with 1 and 2 GPUs using effective batches sizes between 256 and 1024. Using larger batch sizes, or more GPUs, will require modifying the distributed training and/or the learning rates.

<details>
<summary>Run this command to train the <code>DeiT-Ti</code> model in the paper :</summary>
<pre><code>CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
python -m torch.distributed.launch \
       --master_port 1236 \
       --nproc_per_node=1 \
       main_distributed.py \
       --teacher regnety_160 \
       --student-model deit_ti_distilled \
       --batch-size 512 \
       --data-path /data/imagenet2012 \
       --output_dir output/</code></pre>
</details>

<details>
<summary>Run this command to train the <code>DeiT-S</code> model in the paper :</summary>
<pre><code>CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
python -m torch.distributed.launch \
       --master_port 1237 \
       --nproc_per_node=1 \
       main_distributed.py \
       --teacher regnety_160 \
       --student-model deit_s_distilled \
       --batch-size 512 \
       --data-path /data/imagenet2012 \
       --output_dir output/</code></pre>
</details>

## Object Detection
Training a more efficient and effective transformer-based object detector.
All models are trained for 300 epochs with any larger pre-trained ViDT as the teacher.
The MSCOCO dataset should be downloaded and copied to `/data/coco`.
See the `vidt/` README for any additional requirements.

### Pretrained Models

We provide the pre-distilled model weights and full logs for the ViDT experiments. All the teacher weights can be found [here](https://github.com/naver-ai/vidt/tree/main#a-vit-backbone-used-for-vidt). We only provide the mid-training checkpoints, but are currently re-training the models to release the final checkpoints too.

| Backbone | Epochs | AP | AP50 | AP75 | AP_S | AP_M | AP_L | Params | Weights / Log |
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | 
| `Swin-nano` | 50 | 43.0 | 62.3 | 46.2 | 24.8 | 45.3 | 60.1 | 16M | [link](https://drive.google.com/drive/folders/1hgbdWYd8mb5CFlUE4bqnRoaTfTvlA7Fv?usp=sharing)|
| `Swin-tiny` | 50 | 46.9 | 66.6 | 50.9 | 27.8 | 49.8 | 64.6 | 38M | [link](https://drive.google.com/drive/folders/1QvUDcI2Va6QLK5jxdL6DAur1-jlWuPe_?usp=sharing)|

### Evaluation

<details>
<summary>Run this command to evaluate the <code>ViDT (Swin-nano)</code> model in the paper :</summary>
<pre><code>python -m torch.distributed.launch \
       --master_port 1238 \
       --nproc_per_node=8 \
       --nnodes=1 \
       --use_env main.py \
       --method vidt \
       --backbone_name swin_nano \
       --batch_size 2 \
       --num_workers 2 \
       --aux_loss True \
       --with_box_refine True \
       --coco_path /data/coco \
       --resume /ckpts/vidt_nano/checkpoint.pth \
       --pre_trained none \
       --eval True</code></pre>
</details>

<details>
<summary>Run this command to evaluate the <code>ViDT (Swin-tiny)</code> model in the paper :</summary>
<pre><code>python -m torch.distributed.launch \
       --master_port 1239 \
       --nproc_per_node=8 \
       --nnodes=1 \
       --use_env main.py \
       --method vidt \
       --backbone_name swin_tiny \
       --batch_size 2 \
       --num_workers 2 \
       --aux_loss True \
       --with_box_refine True \
       --coco_path /data/coco \
       --resume /ckpts/vidt_tiny/checkpoint.pth \
       --pre_trained none \
       --eval True</code></pre>
</details>

### Training

<details>
<summary>Run this command to train the <code>ViDT (Swin-nano)</code> model in the paper :</summary>
<pre><code>CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch \
       --master_port 1240 \
       --nproc_per_node=6 --nnodes=1 \
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
       --distil_model vidt_base \
       --distil_model_path /ckpts/vidt_base_150.pth \
       --coco_path /home/data/coco \
       --output_dir swin_nano</code></pre>
</details>

<details>
<summary>Run this command to train the <code>ViDT (Swin-tiny)</code> model in the paper :</summary>
<pre><code>CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch \
       --master_port 1241 \
       --nproc_per_node=6 --nnodes=1 \
       --use_env main.py \
       --method vidt \
       --backbone_name swin_small \
       --epochs 50 \
       --lr 1e-4 \
       --min-lr 1e-7 \
       --batch_size 2 \
       --num_workers 2 \
       --aux_loss True \
       --with_box_refine True \
       --distil_model vidt_base \
       --distil_model_path /ckpts/vidt_base_150.pth \
       --coco_path /data/coco \
       --output_dir swin_tiny</code></pre>
</details>
</details>

## Citation
```
@InProceedings{miles2024orthogonalKD,
      title      = {Understanding the Role of the Projector in Knowledge Distillation}, 
      author     = {Roy Miles, Ismail Elezi and Jiankang Deng},
      booktitle  = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      year       = {2024},
      month      = {March}
}
```

If you have any questions, feel free to email me! 

Please consider **citing our paper and staring the repo** if you find this repo useful.
