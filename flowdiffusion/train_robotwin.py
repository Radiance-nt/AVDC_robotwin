import random
from goal_diffusion import GoalGaussianDiffusion, Trainer, calculate_ssim, calculate_psnr
from unet import UnetMW as Unet
from transformers import CLIPTextModel, CLIPTokenizer
from datasets import SequentialDatasetRoboTwin
from torch.utils.data import Subset
import argparse
import os
import torch
import imageio
from tqdm import tqdm
import wandb


def generate_long_sequence(trainer, initial_frame, text, total_steps, sample_per_seq, guidance_weight=0):
    current_frame = initial_frame
    all_frames = [initial_frame.squeeze(0)]
    with torch.no_grad():
        while len(all_frames) < total_steps:
            num_to_generate = min(sample_per_seq - 1, total_steps - len(all_frames))
            pred = trainer.sample(current_frame, [text], 1, guidance_weight)
            new_frames = pred[0].split(3, dim=0)
            all_frames.extend(new_frames[:num_to_generate])
            current_frame = new_frames[-1].unsqueeze(0)
    return torch.stack(all_frames[:total_steps])


def main(args):
    valid_n = 1
    sample_per_seq = 8
    validation_sample_per_seq = args.generate_frames + 1
    target_size = (128, 128)
    temp_dataset = SequentialDatasetRoboTwin(
        sample_per_seq=sample_per_seq,
        path="../datasets/robotwin",
        target_size=target_size,
        randomcrop=False
    )
    total_indices = list(range(len(temp_dataset)))
    random.shuffle(total_indices)
    train_ratio = 0.9
    train_size = int(train_ratio * len(total_indices))
    train_indices = total_indices[:train_size]
    val_indices = total_indices[train_size:]
    if args.use_wandb:
        wandb.config.update({
            "train_indices": train_indices,
            "val_indices": val_indices,
            "train_ratio": train_ratio
        })
    train_set = SequentialDatasetRoboTwin(
        sample_per_seq=sample_per_seq,
        path="../datasets/robotwin",
        target_size=target_size,
        randomcrop=False,
        indices=train_indices
    )
    valid_set = SequentialDatasetRoboTwin(
        sample_per_seq=validation_sample_per_seq,
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
        wandb=args.use_wandb,
    )
    if args.checkpoint_num is not None:
        trainer.load(args.checkpoint_num)
    if args.mode == 'train':
        trainer.train()
    device = trainer.device
    for idx in range(len(valid_set)):
        x_future, x_cond, task = valid_set[idx]
        x_future, x_cond = x_future.to(device), x_cond.to(device)
        full_sequence = torch.cat((x_cond, x_future), dim=0).reshape(-1, 3, *target_size)

        episode_name = valid_set.get_episode_name(idx)
        x_cond = x_cond.unsqueeze(0)
        total_steps = (args.generate_frames + 1) if args.generate_frames >= sample_per_seq else sample_per_seq
        frames = generate_long_sequence(
            trainer,
            x_cond,
            task,
            total_steps=total_steps,
            sample_per_seq=sample_per_seq,
            guidance_weight=args.guidance_weight
        )
        # Calculate PSNR and SSIM for generated sequence
        psnr_values, ssim_values = [], []
        for t in range(1, min(total_steps, full_sequence.shape[0])):
            gt_frame = full_sequence[t]
            pred_frame = frames[t]
            psnr = calculate_psnr(pred_frame, gt_frame)
            ssim = calculate_ssim(pred_frame.unsqueeze(0), gt_frame.unsqueeze(0), device)
            psnr_values.append(psnr.item())
            ssim_values.append(ssim.item())
        avg_psnr = sum(psnr_values) / len(psnr_values)
        avg_ssim = sum(ssim_values) / len(ssim_values)
        print(f'Average PSNR for generated sequence: {avg_psnr:.4f}, Average SSIM: {avg_ssim:.4f}')
        if args.use_wandb:
            wandb.log({'inference_avg_psnr': avg_psnr, 'inference_avg_ssim': avg_ssim})
        frames_np = frames.cpu().numpy().transpose(0, 2, 3, 1)
        frames_np = (frames_np.clip(0, 1) * 255).astype('uint8')
        output_path = f"../results/robotwin/{task}_{episode_name}_steps_{total_steps}.gif"
        imageio.mimsave(output_path, frames_np, duration=200, loop=1000)
        print(f"Saved generated sequence to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='train', choices=['train', 'inference'])
    parser.add_argument('-c', '--checkpoint_num', type=int, default=None)
    parser.add_argument('-p', '--inference_path', type=str, default=None)
    parser.add_argument('-t', '--text', type=str, default=None)
    parser.add_argument('-n', '--sample_steps', type=int, default=100)
    parser.add_argument('-g', '--guidance_weight', type=int, default=0)
    parser.add_argument('-f', '--generate_frames', type=int, default=7)
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--run_name', type=str, default=None)
    args = parser.parse_args()
    if args.use_wandb:
        wandb.init(project="robotwin", name=args.run_name)
    if args.mode == 'inference':
        assert args.checkpoint_num is not None
        assert args.sample_steps <= 100
    main(args)
