import random

from goal_diffusion import GoalGaussianDiffusion, Trainer
from unet import UnetMW as Unet
from transformers import CLIPTextModel, CLIPTokenizer
from datasets import SequentialDatasetRoboTwin
from torch.utils.data import Subset
import argparse
import os
import huggingface_hub
import torch
import imageio
from tqdm import tqdm
import numpy as np

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def generate_long_sequence(trainer, initial_frame, text, total_steps, sample_per_seq, guidance_weight=0):
    """
    Generate sequences longer than sample_per_seq by predicting multiple frames per step,
    while using only the latest frame for conditioning.

    Args:
        trainer: Trainer instance
        initial_frame: Starting frame tensor [1, 3, H, W]
        text: Text prompt
        total_steps: Total frames to generate (including initial frame)
        sample_per_seq: Model's native sequence length
        guidance_weight: CFG weight

    Returns:
        Tensor of frames [total_steps, 3, H, W]
    """
    current_frame = initial_frame  # [1, 3, H, W]
    all_frames = [initial_frame.squeeze(0)]  # Remove batch dim for first frame

    with torch.no_grad():
        while len(all_frames) < total_steps:
            # Number of frames to generate in this step (up to sample_per_seq - 1)
            num_to_generate = min(sample_per_seq - 1, total_steps - len(all_frames))

            # Generate multiple frames at once
            pred = trainer.sample(current_frame, [text], 1, guidance_weight)  # [1, 3*(T-1), H, W]

            # Split into individual frames and add to sequence
            new_frames = pred[0].split(3, dim=0)  # List of [3, H, W] tensors
            all_frames.extend(new_frames[:num_to_generate])

            # Update current_frame to the last generated frame
            current_frame = new_frames[-1].unsqueeze(0)  # [1, 3, H, W]

    return torch.stack(all_frames[:total_steps])  # [total_steps, 3, H, W]


def main(args):
    valid_n = 1
    sample_per_seq = 8
    target_size = (128, 128)

    # Create temporary dataset to get total length
    temp_dataset = SequentialDatasetRoboTwin(
        sample_per_seq=sample_per_seq,
        path="../datasets/robotwin",
        target_size=target_size,
        randomcrop=False
    )

    # Split indices for train and validation
    total_indices = list(range(len(temp_dataset)))
    random.shuffle(total_indices)
    train_ratio = 0.8
    train_size = int(train_ratio * len(total_indices))
    train_indices = total_indices[:train_size]
    val_indices = total_indices[train_size:]

    # Create train and validation datasets
    train_set = SequentialDatasetRoboTwin(
        sample_per_seq=sample_per_seq,
        path="../datasets/robotwin",
        target_size=target_size,
        randomcrop=False,
        indices=train_indices
    )

    valid_set = SequentialDatasetRoboTwin(
        sample_per_seq=sample_per_seq,
        path="../datasets/robotwin",
        target_size=target_size,
        randomcrop=False,
        indices=val_indices
    )

    unet = Unet()

    pretrained_model = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    diffusion = GoalGaussianDiffusion(
        channels=3 * (sample_per_seq - 1),
        model=unet,
        image_size=target_size,
        timesteps=100,
        sampling_timesteps=args.sample_steps,
        loss_type='l2',
        objective='pred_v',
        beta_schedule='cosine',
        min_snr_loss_weight=True,
    )

    trainer = Trainer(
        diffusion_model=diffusion,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        train_set=train_set,
        valid_set=valid_set,
        train_lr=1e-4,
        train_num_steps=60000,
        save_and_sample_every=2500,
        ema_update_every=10,
        ema_decay=0.999,
        train_batch_size=16,
        valid_batch_size=32,
        gradient_accumulate_every=1,
        num_samples=valid_n,
        results_folder='../results/robotwin',
        fp16=True,
        amp=True,
    )

    if args.checkpoint_num is not None:
        trainer.load(args.checkpoint_num)

    if args.mode == 'train':
        trainer.train()

    # Inference on validation set
    device = trainer.device
    for idx in range(len(valid_set)):
        # Get validation sample
        x_future, x_cond, task = valid_set[idx]
        x_future, x_cond = x_future.to(device), x_cond.to(device)
        episode_name = valid_set.get_episode_name(idx)  # Get episode folder name
        x_cond = x_cond.unsqueeze(0)  # [1, 3, H, W]
        # Generate long sequence
        total_steps = args.generate_frames if args.generate_frames > sample_per_seq else sample_per_seq
        frames = generate_long_sequence(
            trainer,
            x_cond,
            task,
            total_steps=total_steps,
            sample_per_seq=sample_per_seq,
            guidance_weight=args.guidance_weight
        )

        # Convert to numpy and save as GIF
        frames_np = frames.cpu().numpy().transpose(0, 2, 3, 1)  # [T, H, W, 3]
        frames_np = (frames_np.clip(0, 1) * 255).astype('uint8')

        output_path = f"../results/robotwin/{task}_{episode_name}_steps_{total_steps}.gif"
        imageio.mimsave(output_path, frames_np, duration=200, loop=1000)
        print(f"Saved generated sequence to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='train',
                        choices=['train', 'inference'])
    parser.add_argument('-c', '--checkpoint_num', type=int,
                        default=None)  # set to checkpoint number to resume training or generate samples
    parser.add_argument('-p', '--inference_path', type=str, default=None)  # set to path to generate samples
    parser.add_argument('-t', '--text', type=str, default=None)  # set to text to generate samples
    parser.add_argument('-n', '--sample_steps', type=int, default=100)  # set to number of steps to sample
    parser.add_argument('-g', '--guidance_weight', type=int, default=0)  # set to positive to use guidance
    parser.add_argument('-f', '--generate_frames', type=int, default=10)  # set to number of steps to sample
    args = parser.parse_args()
    if args.mode == 'inference':
        assert args.checkpoint_num is not None
        assert args.sample_steps <= 100
    main(args)
