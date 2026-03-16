# EGO4D Preprocessing

## Enviroment

```bash
conda create -y -n bidavideo python=3.8 && \
    conda activate bidavideo && \
    pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118) && \
    pip install "git+[https://github.com/facebookresearch/pytorch3d.git@stable](https://github.com/facebookresearch/pytorch3d.git@stable)" && \
    pip install pip==24.0.0 && \
    conda install ipykernel -y && ipython kernel install --user --name=bidavideo && \
    pip install decord && \
    pip install ffmpeg-python && pip install kornia && pip install kornia-rs && pip install kornia_moons
``` 

Then install requirments for [bidavideo](https://github.com/TomTomTommi/bidavideo) via `pip install -r requirements.txt`
    
## Preprocessing

1. We provide rectification parameters in `datasets/ego4d/recrification_params_loftr_ransac`, but we provide script to generate them `compute_rectification_params.py`

2. Run rectification
```bash
PYTHONPATH="../:../third_party/Hi3D-Official/:../third_party/pytorch-msssim/:${PYTHONPATH}" \
    python rectify_videos.py
```

3. Split into 150-frames clips
```bash
PYTHONPATH="../:../third_party/Hi3D-Official/:../third_party/pytorch-msssim/:${PYTHONPATH}" \
    python split_ego4d_into_clips.py
```

4. Download [bidavideo](https://github.com/TomTomTommi/bidavideo#) checkpoints into the `checkpoints` directory. Compute disparities and warp the videos.  
   For each video, we also find a shift that makes the smallest disparity close to 1; therefore, we crop the video from the sides.

```bash
PYTHONPATH="../third_party/:../:../third_party/Hi3D-Official/:../third_party/pytorch-msssim/:${PYTHONPATH}" \
    python run_stereo_and_reproject.py
```