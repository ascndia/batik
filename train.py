import torch
import torch.nn.functional as F
from accelerate import Accelerator
import argparse
import logging
from tqdm import tqdm
from omegaconf import OmegaConf
from utils import build_model, array2grid, load_encoders, requires_grad, sample_posterior, preprocess_raw_image, update_ema
from meanflow import MeanFlow
from dataset import CustomDataset
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup, set_seed
import itertools
from accelerate.logging import get_logger
import wandb
import os
import copy
from copy import deepcopy
from diffusers.models import AutoencoderKL
from sample import meanflow_sampler

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger

def main(args=None):
    config = OmegaConf.load(args.train_config)
    
    accelerator = Accelerator(
        log_with=config.misc.report_to,
        project_dir=config.misc.output_dir,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        mixed_precision=config.train.mixed_precision
    )

    if config.train.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if accelerator.is_main_process:
        tracker_config = OmegaConf.to_container(config, resolve=True)
        accelerator.init_trackers(
            project_name=config.misc.wandb_project, 
            config=tracker_config,
            init_kwargs={
                "wandb": {
                    "name": config.misc.experiment_name 
                }
            },
        )
        os.makedirs(config.misc.output_dir, exist_ok=True)
        save_dir = os.path.join(config.misc.output_dir, config.misc.experiment_name)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        # save yaml config
        OmegaConf.save(config, os.path.join(save_dir, "config.yaml"))

        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")

    device = accelerator.device

    train_cfg = config.train
    model_cfg = config.model
    optim_cfg = config.optimizer
    scheduler_cfg = config.scheduler

    assert train_cfg.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = train_cfg.resolution // 8

    if model_cfg.enc_type != None:
        encoders, encoder_types, architectures = load_encoders(
            model_cfg.enc_type, device, train_cfg.resolution
            )
        encoders = [e.eval() for e in encoders]
    else:
        raise NotImplementedError("Currently, we require encoders for training.")
    
    z_dims = [encoder.embed_dim for encoder in encoders] if model_cfg.enc_type != 'None' else [0]

    model = build_model(model_cfg, z_dims= z_dims)
    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)
    requires_grad(ema, False)

    train_dataset = CustomDataset(data_dir=config.misc.data_dir)
    local_batch_size = int(train_cfg.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=local_batch_size, 
        shuffle=True, 
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({config.misc.data_dir})")
    
    if optim_cfg.use_8bit:
        from bitsandbytes.optim import Adam8bit
        if accelerator.is_main_process:
            print("Using 8-bit AdamW optimizer.")
        optimizer_cls = Adam8bit
    else:
        from torch.optim import AdamW
        if accelerator.is_main_process:
            print("Using standard AdamW optimizer.")
        optimizer_cls = AdamW

    optimizer = optimizer_cls(
        model.parameters(), 
        lr=optim_cfg.lr,
        weight_decay=optim_cfg.weight_decay,
        betas=(optim_cfg.betas[0], optim_cfg.betas[1]),
        eps=optim_cfg.eps
    )

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=scheduler_cfg.warmup_steps,
        num_training_steps=train_cfg.max_steps
    )

    # 4. Prepare everything
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, lr_scheduler
    )

    mf_config = config.meanflow
    mf = MeanFlow(
        path_type=mf_config.path_type,
        weighting=mf_config.weighting,
        time_sampler=mf_config.time_sampler,
        time_mu=mf_config.time_mu,
        time_sigma=mf_config.time_sigma,
        ratio_r_not_equal_t=mf_config.ratio_r_not_equal_t,
        adaptive_p=mf_config.adaptive_p,
        label_dropout_prob=mf_config.label_dropout_prob,
        proj_coeff=mf_config.proj_coeff,
    )

    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    set_seed(train_cfg.seed)
    model.train()
    global_step = 0
    infinite_dataloader = itertools.cycle(train_dataloader)

    progress_bar = tqdm(
        total=train_cfg.max_steps,
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # sd-vae latents scaling factors
    latents_scale = torch.tensor(
        [0.18215, 0.18215, 0.18215, 0.18215]
        ).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor(
        [0., 0., 0., 0.]
        ).view(1, 4, 1, 1).to(device)
    
    sample_batch_size = 16 // accelerator.num_processes
    gt_batch = next(iter(train_dataloader))
    gt_raw_images = gt_batch[0][:sample_batch_size].to(device)
    gt_xs_moments = gt_batch[1][:sample_batch_size].squeeze(1).to(device)
    gt_xs = sample_posterior(gt_xs_moments, latents_scale=latents_scale, latents_bias=latents_bias)
    ys_sample = torch.randint(model_cfg.num_classes, size=(sample_batch_size,), device=device)
    xT_sample = torch.randn((sample_batch_size, 4, latent_size, latent_size), device=device)
    
    print("--- Starting Training ---")
    while global_step < train_cfg.max_steps:
        
        with accelerator.accumulate(model):

            batch = next(infinite_dataloader)
            raw_image = batch[0].to(device)
            x_moments = batch[1].squeeze(dim=1).to(device)
            y = batch[2].to(device)
            
            with torch.no_grad():            
                x = sample_posterior(x_moments, latents_scale=latents_scale, latents_bias=latents_bias)
                zs_target = []
                with accelerator.autocast():
                    for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
                        raw_image_ = preprocess_raw_image(raw_image, encoder_type)
                        z = encoder.forward_features(raw_image_)
                        if 'dinov2' in encoder_type: z = z['x_norm_patchtokens']                                                  
                        zs_target.append(z)

            model_kwargs = {
                "y": y,
                "zs_target": zs_target
            }

            loss, mean_flow_loss, proj_loss = mf.compute_loss(model, x, model_kwargs)

            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                params_to_clip = model.parameters()
                grad_norm = accelerator.clip_grad_norm_(params_to_clip, train_cfg.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if accelerator.sync_gradients:

            update_ema(accelerator.unwrap_model(model), ema)

            global_step += 1
            progress_bar.update(1)

            if global_step % train_cfg.log_interval == 0:
                gathered_loss = accelerator.gather(loss).mean().item()
                gathered_mf_loss = accelerator.gather(mean_flow_loss).mean().item()
                gathered_proj_loss = accelerator.gather(proj_loss).mean().item()
                gathered_grad_norm = accelerator.gather(grad_norm).mean().item()
                lr = lr_scheduler.get_last_lr()[0]
                logs = {
                    "total_loss": gathered_loss,
                    "meanflow_loss": gathered_mf_loss,
                    "repa_loss": gathered_proj_loss,
                    "lr": lr,
                    "grad_norm": gathered_grad_norm
                }
                accelerator.log(logs, step=global_step)

                progress_bar.set_postfix(loss=f"{gathered_loss:.4f}", mf_loss=f"{gathered_mf_loss:.4f}", proj_loss=f"{gathered_proj_loss:.4f}", grad_norm=f"{gathered_grad_norm:.4f}", lr=f"{lr:.2e}")
                # if accelerator.is_main_process:
                #     logger.info(f"Step {global_step}: {logs}")

            if global_step % train_cfg.save_interval == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    checkpoint = {
                        "model": unwrapped_model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "config": config,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            if (global_step == 1 or (global_step % train_cfg.sample_interval == 0)):
                if accelerator.is_main_process:
                    logger.info("Generating samples...")
                
                with torch.no_grad():
                    # Use the EMA model for sampling
                    unwrapped_ema = accelerator.unwrap_model(ema) 
                    unwrapped_ema.eval() # Make sure EMA is in eval mode

                    # --- THIS IS THE CHANGED PART ---
                    samples = meanflow_sampler(
                        unwrapped_ema, 
                        xT_sample, 
                        y=ys_sample,
                        cfg_scale=4.0,  # Using the 4.0 CFG scale from your REPA script
                        num_steps=50    # Using the 50 steps from your REPA script
                    ).to(torch.float32)
                    # --- END OF CHANGED PART ---

                    # The rest of the logic is the same
                    samples = vae.decode((samples - latents_bias) / latents_scale).sample
                    samples = (samples + 1) / 2.
                    out_samples = accelerator.gather(samples)

                if accelerator.is_main_process:
                    gt_samples_decoded = vae.decode((gt_xs - latents_bias) / latents_scale).sample
                    gt_samples_decoded = (gt_samples_decoded + 1) / 2.
                    
                    accelerator.log({
                        "samples": wandb.Image(array2grid(out_samples.cpu())),
                        "gt_samples": wandb.Image(array2grid(gt_samples_decoded.cpu()))
                    }, step=global_step)
                    logger.info("Sampling complete.")
                
                model.train()

    progress_bar.close() 
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        save_path = f"{save_dir}/final_model.pt"
        torch.save(unwrapped_model.state_dict(), save_path)
        print(f"Final model saved to {save_path}")

    print("--- Training Complete ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-GPU Training Script")
    parser.add_argument(
        '--train_config', 
        type=str, 
        required=True, 
        help="Path to the training config.yaml file."
    )
    args = parser.parse_args()
    main(args)