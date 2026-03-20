# M2SVid: End-to-End Inpainting and Refinement for Monocular-to-Stereo Video Conversion


[![Project Page](https://img.shields.io/badge/Project%20Page-m2svid.github.io-blue?style=flat&logo=googlechrome)](https://m2svid.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2505.16565-b31b1b.svg?style=flat&logo=arxiv)](https://arxiv.org/abs/2505.16565)


by [**Nina Shvetsova**](https://ninatu.github.io/), [**Goutam Bhat**](https://goutamgmb.github.io/), [**Prune Truong**](https://prunetruong.com/), [**Hilde Kuehne**](https://hildekuehne.github.io/), [**Federico Tombari**](https://federicotombari.github.io/)

**Accepted to 3DV 2026!**

<p align="center">
  <img src="teaser.gif" width="600">
</p>

**Update:** [March 20, 2026] We have released the pre-trained M2SVid weights! 

---

*This is not an officially supported Google product. This project is not
eligible for the [Google Open Source Software Vulnerability Rewards
Program](https://bughunters.google.com/open-source-security).*

---

</div>

## 📄 Abstract

We tackle the problem of monocular-to-stereo video conversion and propose a novel architecture for inpainting and refinement of the warped right view obtained by depth-based reprojection of the input left view.
We extend the Stable Video Diffusion (SVD) model to utilize the input left video, the warped right video, and the disocclusion masks as conditioning input to generate a high-quality right camera view. In order to effectively exploit information from neighboring frames for inpainting, we modify the attention layers in SVD to compute full attention for discoccluded pixels. Our model is trained to generate the right view video in an end-to-end manner without iterative diffusion steps by minimizing image space losses to ensure high-quality generation.
**Our approach outperforms previous state-of-the-art methods, being ranked best 2.6× more often than the second-place method in a user study, while being 6× faster.**



## 🛠️ Get started

### Weights

1. Download `ckpts.zip` from [Hi3D repo](https://github.com/yanghb22-fdu/Hi3D-Official) and unzip (follow step "2. Download checkpoints here and unzip."). Our model follows Hi3D implementation and uses the same openclip model.

2. Download the [M2SVid weights (8.5Gb)](https://storage.googleapis.com/gresearch/m2svid/m2svid_weights.zip) and extract them into the `ckpts` folder: `unzip m2svid_weights.zip -d ckpts/`. We provide two model variants: one with **full attention** for disoccluded tokens ([m2svid_weights.pt, 4.64Gb](https://storage.googleapis.com/gresearch/m2svid/m2svid_weights.pt)) and one **without full attention** ([m2svid_no_full_atten_weights.pt, 4.6Gb](https://storage.googleapis.com/gresearch/m2svid/m2svid_no_full_atten_weights.pt)). 

3. Optional (for training only): download [stable-video-diffusion-img2vid-xt checkpoint](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) and put it in `ckpts/`.


### Environment

1. Create conda env `depthcrafter` following [DepthCrafter instructions](https://github.com/Tencent/DepthCrafter)
2. Create conda env `sgm`. We used cuda 11.8, `python=3.10.6`, `torch==2.0.1 torchvision==0.15.2`. We tested our model training/inference on GPUs A100 and H100.

```bash
conda env create -f environment.yml -n sgm
```


## ⚙️ Inference

Run inference on demo video:

```bash
bash inference.sh
```

See examples outputs in `demo` folder.

**Note 1:** The width/hight of the video should be divisible by 64.

**Note 2:** The model was trained on a resolution of 512x512. For inference of higher resolution videos, please follow the tiling approach described in the [StereoCrafter paper](https://stereocrafter.github.io/). Our released models support temporal and spatial stitching.

### Inference Steps:
1. **Depth prediction and depth-based warping**

```bash
source /opt/conda/bin/activate ""
conda activate depthcrafter
PYTHONPATH="third_party/DepthCrafter/::${PYTHONPATH}" python third_party/DepthCrafter/run.py  \
        --video-path demo/input.mp4 --save_folder outputs/depthcrafter --save_npz True --num_inference_steps 25 --max_res 1024

PYTHONPATH="./:./third_party/Hi3D-Official/:./third_party/pytorch-msssim/:${PYTHONPATH}" python warping.py  \
        --video_path demo/input.mp4 \
        --depth_path outputs/depthcrafter/input.npz \
        --output_path_reprojected outputs/reprojected/input_reprojected.mp4  \
        --output_path_mask outputs/reprojected/input_reprojected_mask.mp4 \
        --disparity_perc 0.05
```

2. **Inpainting and refinement with M2SVid**

```bash
source /opt/conda/bin/activate ""
conda activate sgm
PYTHONPATH="./:./third_party/Hi3D-Official/:./third_party/pytorch-msssim/:${PYTHONPATH}" python inpaint_and_refine.py  \
        --mask_antialias 0 \
        --model_config configs/m2svid.yaml \
        --ckpt ckpts/m2svid_weights.pt \
        --video_path demo/input.mp4  \
        --reprojected_path outputs/reprojected/input_reprojected.mp4 \
        --reprojected_mask_path outputs/reprojected/input_reprojected_mask.mp4\
        --output_folder outputs/m2svid \
```

**Note:** If you are using the version without full attention, ensure you use the `m2svid_no_full_atten.yaml` config instead:


```bash
source /opt/conda/bin/activate ""
conda activate sgm
PYTHONPATH="./:./third_party/Hi3D-Official/:./third_party/pytorch-msssim/:${PYTHONPATH}" python inpaint_and_refine.py  \
        --mask_antialias 0 \
        --model_config configs/m2svid_no_fullatten.yaml \
        --ckpt ckpts/m2svid_no_full_atten_weights.pt \
        --video_path demo/input.mp4  \
        --reprojected_path outputs/reprojected/input_reprojected.mp4 \
        --reprojected_mask_path outputs/reprojected/input_reprojected_mask.mp4\
        --output_folder outputs/m2svid_no_full_atten \
```



## 🏋️ Training and Quantitative Evaluation

### Datasets

We used the [Ego4D](https://ego4d-data.org/) and [Stereo4D](https://stereo4d.github.io/) datasets for model training and evaluation.

1. Download and preprocess the Stereo4D dataset into the folder `datasets/stereo4d` by following the [official instructions](https://github.com/Stereo4d/stereo4d-code). You only need to perform the rectification and stereo matching steps. Then, you can warp all videos using our `warping.py` script. At the end, you should have the following folders: `left_rectified`, `right_rectified`, `reprojected`, and `reprojected_mask`. We provide the train/val split in `datasets/stereo4d/subsets`.

2.  For Ego4D, we use only videos with the attribute `is_stereo=True`, resulting in 263 videos in total.  Download videos into `datasets/ego4d` by following the [official instructions](https://ego4d-data.org/). We rectify the videos, split them into 150-frames clips, and apply the BiDAStereo model to estimate disparities. Check the [**ego4d preprocessing README**](data_preprocess/) for more details.   At the end, you should have the following folders: `cropped_videos` (side by side rectified and cropped left and right videos), `reprojected`, and `reprojected_mask`. We provide the train/val split in `datasets/ego4d/subsets`.

### Training 

1. Download [stable-video-diffusion-img2vid-xt checkpoint](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) and put it to ckpts. 

2. Run `make_m2svid_init.py` to modify SVD models weights for ours M2SVid model configuration with left view, warped view and mask conditioning. 

```bash
source /opt/conda/bin/activate ""
conda activate sgm
PYTHONPATH="./:./third_party/Hi3D-Official/:./third_party/pytorch-msssim/:${PYTHONPATH}" python make_m2svid_init.py
```

3. Run training 
```bash
source /opt/conda/bin/activate ""
conda activate sgm
PYTHONPATH="./:./third_party/Hi3D-Official/:./third_party/pytorch-msssim/:${PYTHONPATH}" python third_party/Hi3D-Official/train_test_updated.py \
    --base configs/training/m2svid_train.yaml \
    --no-test True \
    --train True \
    --logdir outputs/training/m2svid
```

### Evaluation

Evaluation on stereo4d: 

```bash
source /opt/conda/bin/activate ""
conda activate sgm
PYTHONPATH="./:./third_party/Hi3D-Official/:./third_party/pytorch-msssim/:${PYTHONPATH}" python third_party/Hi3D-Official/train_test_updated.py \
    --base configs/training/m2svid_train.yaml \
    --dataset_base configs/testing/stereo4d.yaml \
    --no-test False \
    --train False \
    --logdir outputs/training/m2svid \
    --resume /home/jupyter/outputs_m2svid/training/m2svid/checkpoints/epoch=000120.ckpt
```

Evaluation on ego4d:

```bash
source /opt/conda/bin/activate ""
conda activate sgm
PYTHONPATH="./:./third_party/Hi3D-Official/:./third_party/pytorch-msssim/:${PYTHONPATH}" python third_party/Hi3D-Official/train_test_updated.py \
    --base configs/training/m2svid_train.yaml \
    --dataset_base configs/testing/ego4d.yaml \
    --no-test False \
    --train False \d
    --logdir outputs/training/m2svid \
    --resume /home/jupyter/outputs_m2svid/training/m2svid/checkpoints/epoch=000000.ckpt
```

### Evaluation of Released Models

To reproduce the paper's results on **Stereo4D** and **Ego4D** using our released weights:

```bash
source /opt/conda/bin/activate ""
conda activate sgm

# Evaluate on Stereo4D
PYTHONPATH="./:./third_party/Hi3D-Official/:./third_party/pytorch-msssim/:${PYTHONPATH}" python third_party/Hi3D-Official/train_test_updated.py \
    --base configs/testing/pretrained_m2svid.yaml \
    --dataset_base configs/testing/stereo4d.yaml \
    --no-test False \
    --train False \
    --logdir outputs/training/m2svid 

# Evaluate on Ego4D
PYTHONPATH="./:./third_party/Hi3D-Official/:./third_party/pytorch-msssim/:${PYTHONPATH}" python third_party/Hi3D-Official/train_test_updated.py \
    --base configs/training/pretrained_m2svid.yaml \
    --dataset_base configs/testing/stereo4d.yaml \
    --no-test False \
    --train False \
    --logdir outputs/training/m2svid 
```

## 🎓 Citation

```bibtex
@article{shvetsova2026m2svid,
  title={M2SVid: End-to-End Inpainting and Refinement for Monocular-to-Stereo Video Conversion},
  author={Shvetsova, Nina and Bhat, Goutam and Truong, Prune and Kuehne, Hilde and Tombari, Federico},
  journal={3DV},
  year={2026}
}
```
