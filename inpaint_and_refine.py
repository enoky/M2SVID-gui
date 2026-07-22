import random
import argparse
import json
import sys
import gc
from pytorch_lightning import seed_everything
import os
import subprocess
import ffmpeg
from torchvision import transforms
import torch
import numpy as np
import einops
from omegaconf import OmegaConf
from sgm.util import instantiate_from_config
import warnings
import torch.nn.functional as F
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

from m2svid.utils.video_utils import open_ffmpeg_process, get_video_fps
from m2svid.data.utils import get_video_frames, apply_closing, apply_dilation

parser = argparse.ArgumentParser()
parser.add_argument("--model_config", type=str)
parser.add_argument("--ckpt", type=str)
parser.add_argument("--video_path", type=str)
parser.add_argument("--grid_video_path", type=str, help="Grid video where left half is mask and right half is wrapped video")
parser.add_argument("--output_folder", type=str)
parser.add_argument("--reprojected_closing_holes_kernel", type=int, default=11)
parser.add_argument("--mask_antialias", type=int, default=False)
parser.add_argument("--spatial_tile_size", type=int, default=512)
parser.add_argument("--spatial_tile_overlap", type=int, default=256)
parser.add_argument("--decode_window", type=int, default=6, help="Frames decoded together at full resolution (lower if decoding OOMs)")
parser.add_argument("--decode_temporal_overlap", type=int, default=2, help="Frames cross-faded between consecutive decode windows")
# New temporal chunking arguments
parser.add_argument("--chunk_size", type=int, default=25, help="Total frames per model forward pass")
parser.add_argument("--overlap", type=int, default=3, help="Number of frames to overlap and cross-fade")
parser.add_argument("--original_input_blend_strength", type=float, default=0.0, help="Weight of original input for conditioning")
parser.add_argument("--dry_run", action="store_true", help="Print chunk schedule and exit without running the model")
parser.add_argument("--steps", type=int, default=None, help="Number of inference steps (default is from model config)")
# Worker mode for batch processing (R1 optimization)
parser.add_argument("--worker", action="store_true", help="Run as persistent worker: load model once, process clips from stdin JSON lines")
parser.add_argument("--compile", action="store_true", help="torch.compile the UNet and VAE decoder (~15-20%% faster steady-state; one-time warmup on the first clip)")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Chunk schedule builder
# ---------------------------------------------------------------------------
def build_chunk_schedule(total_frames, chunk_size, overlap):
    """Build a list of chunks describing which source frames to use and how to handle overlaps.
    
    Each entry is a dict:
        source_indices: list[int] - contiguous indices into the source video
        actual_len: int - number of valid frames generated before padding
        overlap: int - frames overlapping with previous chunk (to be cross-faded)
        tail_overlap: int - frames overlapping with next chunk (to be cached)
        abs_start: int - the absolute starting frame index
        padded: int - how many pad frames were appended
    """
    if total_frames <= 0:
        return []

    stride = max(1, chunk_size - overlap)
    schedule = []
    
    for i in range(0, total_frames, stride):
        end_idx = min(i + chunk_size, total_frames)
        actual_len = end_idx - i
        
        # Skip useless tail chunks that contribute almost no new frames
        if i > 0 and overlap > 0 and actual_len <= overlap:
            break
            
        src_indices = list(range(i, end_idx))
        padded = 0
        while len(src_indices) < chunk_size:
            src_indices.append(src_indices[-1])
            padded += 1
            
        is_last = (i + stride >= total_frames) or (total_frames - (i + stride) <= overlap)
        
        schedule.append({
            'source_indices': src_indices,
            'actual_len': actual_len,
            'padded': padded,
            'overlap': overlap if i > 0 else 0,
            'tail_overlap': overlap if not is_last else 0,
            'abs_start': i,
        })
        
    return schedule


# ---------------------------------------------------------------------------
# Spatial tiling helpers (unchanged from original)
# ---------------------------------------------------------------------------
def get_spatial_bounds(length, size, stride):
    bounds = []
    for start in range(0, length, stride):
        end = min(start + size, length)
        if end - start < size and length >= size:
            start = end - size
        bounds.append((start, end))
        if end == length:
            break
    return bounds


# ---------------------------------------------------------------------------
# Model loading (extracted for reuse in worker mode)
# ---------------------------------------------------------------------------
def load_model(model_config_path, ckpt_path, steps=None, use_compile=False):
    """Load and prepare the denoising model. Returns the model in fp16 eval mode on CPU."""
    config = OmegaConf.load(model_config_path)
    if steps is not None and steps > 0:
        config.model.params.sampler_config.params.num_steps = steps
    denoising_model = instantiate_from_config(config.model).cpu()
    denoising_model.init_from_ckpt(ckpt_path)
    denoising_model = denoising_model.half().eval()

    if use_compile:
        try:
            import torch._dynamo as dynamo
            # The tiled/chunked pipeline produces a handful of shape variants; keep them all compiled.
            for knob in ("recompile_limit", "cache_size_limit"):
                if hasattr(dynamo.config, knob):
                    setattr(dynamo.config, knob, 64)
            # Degrade gracefully to eager (e.g. missing Python dev headers) instead of failing the job.
            dynamo.config.suppress_errors = True
            denoising_model.model.diffusion_model.compile(dynamic=True)
            denoising_model.first_stage_model.decoder.compile(dynamic=True)
            print("torch.compile ENABLED for UNet and VAE decoder (first clip includes one-time compile warmup).")
        except Exception as e:
            print(f"Warning: could not enable torch.compile ({e}). Continuing in eager mode.")

    return denoising_model


# ---------------------------------------------------------------------------
# Per-clip processing (extracted for reuse in worker mode)
# ---------------------------------------------------------------------------
def process_clip(denoising_model, job):
    """Process a single clip using the already-loaded model.
    
    Args:
        denoising_model: The loaded VideoLDM model (fp16, eval, on CPU).
        job: Dict with keys matching the CLI args for a single clip:
            video_path, grid_video_path, output_folder,
            reprojected_closing_holes_kernel, mask_antialias,
            spatial_tile_size, spatial_tile_overlap,
            chunk_size, overlap, decode_window, decode_temporal_overlap,
            original_input_blend_strength, dry_run
    """
    seed = random.randint(0, 65535)
    seed_everything(seed)

    # Extract job parameters with defaults
    video_path = job["video_path"]
    grid_video_path = job["grid_video_path"]
    output_folder = job["output_folder"]
    reprojected_closing_holes_kernel = job.get("reprojected_closing_holes_kernel", 11)
    mask_antialias = job.get("mask_antialias", 0)
    spatial_tile_size = job.get("spatial_tile_size", 512)
    spatial_tile_overlap = job.get("spatial_tile_overlap", 256)
    chunk_size = job.get("chunk_size", 25)
    overlap = job.get("overlap", 3)
    decode_window = max(1, job.get("decode_window", 6))
    decode_temporal_overlap = max(0, min(job.get("decode_temporal_overlap", 2), decode_window - 1))
    original_input_blend_strength = job.get("original_input_blend_strength", 0.0)
    dry_run = job.get("dry_run", False)

    # Clamp chunk_size to model maximum
    max_temporal_size = getattr(denoising_model, "num_samples", chunk_size)
    if chunk_size > max_temporal_size:
        print(f"Warning: chunk_size ({chunk_size}) exceeds model's max num_samples ({max_temporal_size}). Capping to {max_temporal_size}.")
        chunk_size = max_temporal_size

    # Load and preprocess videos
    input_video = get_video_frames(video_path, normalize=False)
    grid_video = get_video_frames(grid_video_path, normalize=False)
    W_half = grid_video.shape[3] // 2
    reprojected_mask = grid_video[:, 0:1, :, :W_half]
    reprojected = grid_video[:, :, :, W_half:]
    fps = get_video_fps(video_path, ffmpeg.probe(video_path))

    reprojected_mask = apply_closing(reprojected_mask, reprojected_closing_holes_kernel)
    reprojected[reprojected_mask.repeat(1, 3, 1, 1) > 0.5] = 0
    reprojected_mask = apply_dilation(reprojected_mask, 3)
    # reprojected_mask = reprojected_mask.repeat(1, 3, 1, 1)  # Keep as 1-channel to save RAM

    input_video = input_video.permute(1, 0, 2, 3)       # [c,t,h,w], uint8 0-255
    reprojected = reprojected.permute(1, 0, 2, 3)       # [c,t,h,w], uint8 0-255
    reprojected_mask = reprojected_mask.permute(1, 0, 2, 3)  # [c,t,h,w], uint8 0/1

    # Match dimensions - Grid resolution is authoritative
    H_iv, W_iv = input_video.shape[2:]
    H_rp, W_rp = reprojected.shape[2:]

    # Ensure Grid resolution is a multiple of 8 to avoid VAE size mismatch (e.g. 802 -> 800)
    target_H = (H_rp // 8) * 8
    target_W = (W_rp // 8) * 8

    # Resize reprojected and mask if they aren't multiples of 8 (required for VAE consistency)
    if H_rp != target_H or W_rp != target_W:
        print(f"Warning: Grid resolution ({W_rp}x{H_rp}) is not a multiple of 8. Resizing to {target_W}x{target_H} to avoid model mismatch.")
        reprojected = F.interpolate(reprojected.float(), size=(target_H, target_W), mode='bilinear', align_corners=False).to(reprojected.dtype).clamp(0, 255)
        reprojected_mask = F.interpolate(reprojected_mask.float(), size=(target_H, target_W), mode='bilinear', align_corners=False).to(reprojected_mask.dtype).clamp(-1, 1)

    # Resize input_video to match target resolution exactly (no crops, irrespective of aspect ratio)
    if H_iv != target_H or W_iv != target_W:
        input_video = F.interpolate(input_video.float(), size=(target_H, target_W), mode='bilinear', align_corners=False).to(input_video.dtype).clamp(0, 255)

    c, T, H, W = reprojected_mask.shape
    downsampled_resolution = [int(H / 8), int(W / 8)]

    # Perform resizing in chunks to avoid large float32 allocations (Peak RAM reduction)
    print(f"Resizing mask chunks for latent space ({T} frames)...")
    resized_masks = []
    chunk_size_resize = 100 
    for i in range(0, T, chunk_size_resize):
        # Convert chunk to float for resizing, but keep channel count at 1
        m_chunk = reprojected_mask[:, i:i+chunk_size_resize].permute(1, 0, 2, 3).float()
        # Normalize/clamp to [0, 1] before resizing if needed, but here they are 0/1
        m_chunk = transforms.Resize(
            downsampled_resolution, 
            antialias=mask_antialias
        )(m_chunk).clamp(0, 1)
        resized_masks.append(m_chunk.half()) # Store as half precision [0, 1]

    reprojected_mask = torch.cat(resized_masks, dim=0) # [T, 1, LH, LW]
    reprojected_mask = reprojected_mask.permute(1, 0, 2, 3) # [1, T, LH, LW]
    reprojected_mask = reprojected_mask * 2.0 - 1.0  # Scale to [-1, 1] for model input

    latent_H, latent_W = H // 8, W // 8

    # Spatial tiling setup
    h_stride = max(1, spatial_tile_size - spatial_tile_overlap)
    w_stride = max(1, spatial_tile_size - spatial_tile_overlap)
    h_size = spatial_tile_size
    w_size = spatial_tile_size
    h_bounds = get_spatial_bounds(H, h_size, h_stride)
    w_bounds = get_spatial_bounds(W, w_size, w_stride)

    # Build chunk schedule
    chunk_schedule = build_chunk_schedule(T, chunk_size, overlap)

    print(f"\n=== Chunk Schedule (total_frames={T}, chunk_size={chunk_size}, overlap={overlap}) ===")
    for ci, ch in enumerate(chunk_schedule):
        src = ch['source_indices']
        print(f"  Chunk {ci}: source[{src[0]}..{src[-1]}] "
              f"(pad={ch['padded']}) -> output [{ch['abs_start']}..{ch['abs_start'] + ch['actual_len'] - 1}] "
              f"({ch['actual_len']} active frames) "
              f"[overlap={ch['overlap']}, tail_overlap={ch['tail_overlap']}]")

    total_output = sum(ch['actual_len'] for ch in chunk_schedule) - sum(ch['overlap'] for ch in chunk_schedule)
    print(f"  Total output frames: {total_output} (expected: {T})")
    assert total_output == T, f"Output frame count mismatch: {total_output} != {T}"

    if dry_run:
        print("\n--dry_run: exiting without running the model.")
        return

    print(f"\nSpatial tiles: {len(h_bounds)} height x {len(w_bounds)} width")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, f'{video_name}_generated.mp4')
    print(f"Streaming generated chunks into {out_path} as they complete...")

    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-loglevel', 'error',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{W}x{H}',
        '-pix_fmt', 'rgb24',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-x264opts', 'rc-lookahead=10',
        '-crf', '14',
        '-profile:v', 'high10',
        '-pix_fmt', 'yuv420p10le',
        out_path
    ]

    ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    first_stage_model = denoising_model.first_stage_model

    overlap_buffer = []  # To store decoded pixel frames from the tail of the previous chunk

    for ci, chunk_info in enumerate(tqdm(chunk_schedule, desc="Temporal Chunks")):
        src_indices = chunk_info['source_indices']
        c_overlap = chunk_info['overlap']
        c_tail_overlap = chunk_info['tail_overlap']
        abs_start = chunk_info['abs_start']
        actual_len = chunk_info['actual_len']
        n_gen = len(src_indices)  # == chunk_size (after padding)

        # Gather source frames in reordered order  [c, chunk_size, h, w]
        iv_chunk = torch.stack([input_video[:, idx] for idx in src_indices], dim=1)
        rp_chunk = torch.stack([reprojected[:, idx] for idx in src_indices], dim=1)
        
        # Optional original input conditioning blend for overlap region
        if ci > 0 and c_overlap > 0 and overlap_buffer and original_input_blend_strength > 0:
            blend_strength = original_input_blend_strength
            for f_rel in range(min(c_overlap, len(overlap_buffer))):
                w = (f_rel / (c_overlap - 1)) if c_overlap > 1 else 0.5
                w = w * blend_strength
                orig_frame_tensor = rp_chunk[:, f_rel]
                prev_gen_np = overlap_buffer[f_rel]
                prev_gen_tensor = torch.tensor(prev_gen_np).permute(2, 0, 1).float().to(orig_frame_tensor.device)
                rp_chunk[:, f_rel] = ((1.0 - w) * prev_gen_tensor + w * orig_frame_tensor).to(rp_chunk.dtype)

        # Mask is in latent space [c, t, lh, lw] — gather similarly
        rm_chunk = torch.stack([reprojected_mask[:, idx] for idx in src_indices], dim=1)

        # Allocate per-chunk latent accumulator (for spatial blending) ON GPU
        chunk_latent_H, chunk_latent_W = latent_H, latent_W
        chunk_latents = torch.zeros((1, 4, n_gen, chunk_latent_H, chunk_latent_W), dtype=torch.float16, device="cuda")
        chunk_weights = torch.zeros((1, 1, n_gen, chunk_latent_H, chunk_latent_W), dtype=torch.float16, device="cuda")

        # Move full chunks to GPU before the loop to eliminate CPU->GPU transfers per tile
        iv_chunk_gpu_full = (iv_chunk.cuda().half() / 255.0) * 2.0 - 1.0
        rp_chunk_gpu_full = (rp_chunk.cuda().half() / 255.0) * 2.0 - 1.0
        rm_chunk_gpu_full = rm_chunk.cuda()

        # R2 Two-Pass Loop
        # ----------------------------------------------------
        # PASS 1: CONDITIONING
        # ----------------------------------------------------
        cond_results = []
        denoising_model.conditioner.to('cuda')
        
        for h_s, h_e in h_bounds:
            row_conds = []
            for w_s, w_e in w_bounds:
                # Slice spatial region DIRECTLY ON GPU
                iv_slice_gpu = iv_chunk_gpu_full[:, :, h_s:h_e, w_s:w_e]
                rp_slice_gpu = rp_chunk_gpu_full[:, :, h_s:h_e, w_s:w_e]
                lh_s, lh_e = h_s // 8, h_e // 8
                lw_s, lw_e = w_s // 8, w_e // 8
                rm_slice_gpu = rm_chunk_gpu_full[:, :, lh_s:lh_e, lw_s:lw_e]

                input_batch = {
                    'video': iv_slice_gpu[None],
                    'video_2nd_view': iv_slice_gpu[None],
                    'reprojected_video': rp_slice_gpu[None],
                    'reprojected_mask': rm_slice_gpu[None],
                    'fps_id': torch.tensor([fps]).cuda(),
                    'caption': [""],
                    "motion_bucket_id": torch.tensor([127]).cuda()
                }

                with torch.inference_mode():
                    with torch.autocast("cuda", dtype=torch.float16):
                        # Extract conditioning logic natively
                        batch = denoising_model.add_custom_cond(input_batch, infer=True)
                        frames = denoising_model.get_input(batch)
                        conditioner_input_keys = [e.input_key for e in denoising_model.conditioner.embedders]
                        c, uc = denoising_model.conditioner.get_unconditional_conditioning(
                            batch,
                            force_uc_zero_embeddings=conditioner_input_keys if len(denoising_model.conditioner.embedders) > 0 else [],
                        )
                        row_conds.append({
                            "frames": frames,
                            "c": c,
                            "uc": uc,
                            "num_video_frames": batch["num_video_frames"],
                            "inpainting_mask": batch.get("inpainting_mask", None)
                        })
            cond_results.append(row_conds)

        denoising_model.conditioner.to('cpu')
        torch.cuda.empty_cache()

        # ----------------------------------------------------
        # PASS 2: SAMPLING
        # ----------------------------------------------------
        if hasattr(denoising_model, 'model'):
            denoising_model.model.to('cuda')

        spatial_pbar = tqdm(total=len(h_bounds) * len(w_bounds),
                            desc=f"Spatial Tiles for chunk {ci} (frames {abs_start}-{abs_start+actual_len-1})",
                            leave=False)

        for i, (h_s, h_e) in enumerate(h_bounds):
            for j, (w_s, w_e) in enumerate(w_bounds):
                cond_data = cond_results[i][j]
                frames = cond_data["frames"]
                c = cond_data["c"]
                uc = cond_data["uc"]
                num_video_frames = cond_data["num_video_frames"]
                inpainting_mask = cond_data["inpainting_mask"]
                
                N = len(frames)
                x = einops.rearrange(frames, 'b c t h w -> (b t) c h w').to('cuda')
                
                additional_model_inputs = {}
                additional_model_inputs["image_only_indicator"] = torch.zeros(N * 2, num_video_frames).to('cuda')
                additional_model_inputs["num_video_frames"] = num_video_frames

                if denoising_model.cond_reprojected_video and inpainting_mask is not None:
                    inp_mask = torch.concat([inpainting_mask, inpainting_mask], dim=0)
                    additional_model_inputs["inpainting_mask"] = inp_mask

                def denoiser(input, sigma, c):
                    return denoising_model.denoiser(denoising_model.model, input, sigma, c, **additional_model_inputs)

                with torch.inference_mode():
                    with denoising_model.ema_scope("Plotting"):
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            shape = (x.shape[0], 4, int(x.shape[2] // 8), int(x.shape[3] // 8))
                            randn = torch.randn(shape, device='cuda')
                            samples = denoising_model.sampler(denoiser, randn, cond=c, uc=uc, num_video_frames=num_video_frames)
                            
                            generated_latent = einops.rearrange(
                                samples, '(b t) c h w -> b c t h w', b=1, t=n_gen
                            )

                lh_s, lh_e = h_s // 8, h_e // 8
                lw_s, lw_e = w_s // 8, w_e // 8

                # Spatial blending weight ON GPU
                weight = torch.ones((1, 1, n_gen, lh_e - lh_s, lw_e - lw_s), dtype=torch.float16, device="cuda")
                latent_h_ovr = spatial_tile_overlap // 8
                latent_w_ovr = spatial_tile_overlap // 8
                if h_s > 0:
                    ramp = torch.linspace(0, 1, latent_h_ovr, dtype=torch.float16, device="cuda").view(1, 1, 1, -1, 1)
                    weight[:, :, :, :latent_h_ovr, :] *= ramp
                if h_e < H:
                    ramp = torch.linspace(1, 0, latent_h_ovr, dtype=torch.float16, device="cuda").view(1, 1, 1, -1, 1)
                    weight[:, :, :, -latent_h_ovr:, :] *= ramp
                if w_s > 0:
                    ramp = torch.linspace(0, 1, latent_w_ovr, dtype=torch.float16, device="cuda").view(1, 1, 1, 1, -1)
                    weight[:, :, :, :, :latent_w_ovr] *= ramp
                if w_e < W:
                    ramp = torch.linspace(1, 0, latent_w_ovr, dtype=torch.float16, device="cuda").view(1, 1, 1, 1, -1)
                    weight[:, :, :, :, -latent_w_ovr:] *= ramp

                chunk_latents[:, :, :, lh_s:lh_e, lw_s:lw_e] += generated_latent * weight
                chunk_weights[:, :, :, lh_s:lh_e, lw_s:lw_e] += weight

                del samples, randn, x, additional_model_inputs, frames, c, uc, cond_data
                cond_results[i][j] = None
                spatial_pbar.update(1)

        spatial_pbar.close()
        
        if hasattr(denoising_model, 'model'):
            denoising_model.model.to('cpu')

        # Resolve spatial blending on GPU, then transfer back to CPU for temporal blending
        resolved = (chunk_latents / chunk_weights).cpu()  # (1, 4, n_gen, lH, lW)
        del chunk_latents, chunk_weights, iv_chunk_gpu_full, rp_chunk_gpu_full, rm_chunk_gpu_full

        # Flush residual CUDA cache from spatial tile inference to prevent hang/OOM
        # when loading first_stage_model onto GPU
        torch.cuda.empty_cache()

        first_stage_model.decoder.to('cuda')
        
        # Decode at FULL spatial resolution. Spatially tiled VAE decoding gives
        # each tile slightly different per-frame normalization statistics, which
        # shows up as tile-shaped color shifts that flicker over time. VRAM is
        # bounded instead by decoding in overlapping temporal windows that are
        # cross-faded in pixel space, so the VideoDecoder's temporal layers
        # still get a multi-frame window to stabilize per-frame color.
        t_stride = max(1, decode_window - decode_temporal_overlap)
        t_bounds = get_spatial_bounds(n_gen, decode_window, t_stride)

        # Accumulators for temporal cross-fading of decode windows (CPU)
        pixel_accum = torch.zeros((1, 3, n_gen, H, W), dtype=torch.float32)
        weight_accum = torch.zeros((1, 1, n_gen, 1, 1), dtype=torch.float32)

        decode_pbar = tqdm(total=len(t_bounds),
                           desc=f"Decoding Spatial chunk {ci}",
                           leave=False)

        for t_s, t_e in t_bounds:
            win_len = t_e - t_s
            latent_win = resolved[:, :, t_s:t_e].cuda()
            latent_win_flat = einops.rearrange(latent_win, 'b c t h w -> (b t) c h w')
            with torch.inference_mode():
                with torch.autocast("cuda", dtype=torch.float16):
                    decoded_flat = denoising_model.decode_first_stage(latent_win_flat, num_video_frames=win_len)
            decoded_win = einops.rearrange(decoded_flat, '(b t) c h w -> b c t h w', b=1, t=win_len).cpu().float()
            del latent_win, latent_win_flat, decoded_flat
            torch.cuda.empty_cache()

            # Temporal cross-fade ramps at window edges. Ramp endpoints are
            # excluded so aligned head/tail ramps sum to 1 and no frame ever
            # gets zero total weight.
            w_t = torch.ones(win_len, dtype=torch.float32)
            ramp_len = min(decode_temporal_overlap, win_len)
            if t_s > 0 and ramp_len > 0:
                w_t[:ramp_len] = torch.linspace(0, 1, ramp_len + 2)[1:-1]
            if t_e < n_gen and ramp_len > 0:
                w_t[-ramp_len:] = torch.linspace(1, 0, ramp_len + 2)[1:-1]
            w_t = w_t.view(1, 1, -1, 1, 1)

            pixel_accum[:, :, t_s:t_e] += decoded_win * w_t
            weight_accum[:, :, t_s:t_e] += w_t

            del decoded_win
            decode_pbar.update(1)

        decode_pbar.close()

        # Resolve weighted average and convert back to half precision
        resolved_pixels = (pixel_accum / weight_accum).half()
        del pixel_accum, weight_accum

        new_overlap_buffer = []

        # Now stream the resolved pixels to ffmpeg
        for f_rel in range(0, actual_len):
            frame_tensor = resolved_pixels[0, :, f_rel, :, :]
            frame_numpy = frame_tensor.float().numpy().transpose(1, 2, 0)
            frame_numpy_uint8 = (((frame_numpy + 1) / 2).clip(0, 1) * 255).astype(np.uint8)
            
            abs_idx = abs_start + f_rel
            final_frame = frame_numpy_uint8
            
            # Crossfade overlapping head frames with previous tail
            if ci > 0 and f_rel < c_overlap and f_rel < len(overlap_buffer):
                w = f_rel / (c_overlap - 1) if c_overlap > 1 else 0.5
                prev_frame = overlap_buffer[f_rel]
                blended = (1.0 - w) * prev_frame.astype(np.float32) + w * final_frame.astype(np.float32)
                final_frame = blended.clip(0, 255).astype(np.uint8)

            # Output to ffmpeg unless this is a tail frame of a non-last chunk
            is_tail = (f_rel >= actual_len - c_tail_overlap)
            if not is_tail:
                ffmpeg_process.stdin.write(np.ascontiguousarray(final_frame).tobytes())
            else:
                new_overlap_buffer.append(final_frame)

            # Feed this generated frame back as conditioning for future chunks
            if abs_idx < T:
                reprojected[:, abs_idx] = torch.tensor(final_frame).permute(2, 0, 1)
                reprojected_mask[:, abs_idx] = -1.0  # mark as unmasked

        first_stage_model.decoder.to('cpu')
        torch.cuda.empty_cache()
        ffmpeg_process.stdin.flush()
        del resolved, resolved_pixels
        overlap_buffer = new_overlap_buffer

    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
    print("Done processing!")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if args.worker:
        # ---------------------------------------------------------------
        # WORKER MODE (R1): Load model once, process clips from stdin
        # ---------------------------------------------------------------
        print("Loading model (worker mode)...")
        denoising_model = load_model(args.model_config, args.ckpt, args.steps, use_compile=args.compile)
        
        # Signal readiness to parent process
        sys.stdout.write("###WORKER_READY###\n")
        sys.stdout.flush()

        # Process jobs from stdin (one JSON line per clip)
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            if line == "###EXIT###":
                break
            
            try:
                job = json.loads(line)
            except json.JSONDecodeError as e:
                sys.stdout.write(f"###JOB_FAILED###{e}###\n")
                sys.stdout.flush()
                continue

            try:
                process_clip(denoising_model, job)
                sys.stdout.write("###JOB_COMPLETE###\n")
                sys.stdout.flush()
            except Exception as e:
                import traceback
                traceback.print_exc()
                sys.stdout.write(f"###JOB_FAILED###{e}###\n")
                sys.stdout.flush()
            
            # Cleanup between clips to prevent VRAM/RAM leaks
            torch.cuda.empty_cache()
            gc.collect()

        print("Worker shutting down.")
    else:
        # ---------------------------------------------------------------
        # LEGACY MODE: Single-clip processing (backward compatible)
        # ---------------------------------------------------------------
        denoising_model = load_model(args.model_config, args.ckpt, args.steps, use_compile=args.compile)
        job = {
            "video_path": args.video_path,
            "grid_video_path": args.grid_video_path,
            "output_folder": args.output_folder,
            "reprojected_closing_holes_kernel": args.reprojected_closing_holes_kernel,
            "mask_antialias": args.mask_antialias,
            "spatial_tile_size": args.spatial_tile_size,
            "spatial_tile_overlap": args.spatial_tile_overlap,
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
            "decode_window": args.decode_window,
            "decode_temporal_overlap": args.decode_temporal_overlap,
            "original_input_blend_strength": args.original_input_blend_strength,
            "dry_run": args.dry_run,
        }
        process_clip(denoising_model, job)
