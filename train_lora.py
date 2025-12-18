"""
Train Stable Diffusion LoRA - FULLY AUTOMATED
Optimized for RTX 4050 (8GB VRAM)

Usage:
1. Create a folder named 'dataset' in this directory.
2. Put your images in it (e.g., image1.jpg, image2.png).
3. For each image, create a text file with the same name containing the prompt (e.g., image1.txt).
   Example: 
     image1.jpg -> "a photo of a cat sitting on a chair"
     image1.txt -> "a photo of a cat sitting on a chair"
4. Run this script!
"""

import os
import argparse
import itertools
import math
import random
import json
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
from torchvision import transforms

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from transformers import CLIPTextModel, CLIPTokenizer
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from peft import LoraConfig, get_peft_model, PeftModel

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "stable-diffusion-v1-5/stable-diffusion-v1-5"
OUTPUT_DIR = "lora_output"
DATASET_DIR = "dataset"
RESOLUTION = 512
TRAIN_BATCH_SIZE = 1  # Keep low for 8GB VRAM
GRADIENT_ACCUMULATION_STEPS = 4  # Increase effective batch size
LEARNING_RATE = 8e-5
LR_SCHEDULER = "constant"
LR_WARMUP_STEPS = 0
MAX_TRAIN_STEPS = 450 # Optimized for 7 images
CHECKPOINTING_STEPS = 225
SEED = 42

# Memory optimization
MIXED_PRECISION = "fp16" # "no", "fp16", "bf16"
USE_8BIT_ADAM = False # Requires bitsandbytes, set to False for Windows compatibility by default

def load_image_dataset(dataset_dir, size=512):
    """
    Load images and corresponding text files from a directory.
    Supports two formats:
    1. Simple: image.jpg + image.txt
    2. CSV: prompts.csv (imgId, prompt) + images/ folder
    """
    dataset = []
    path = Path(dataset_dir)
    
    # Check for case-insensitive dataset directory if not found
    if not path.exists():
        # Try to find 'dataset' or 'Dataset'
        for p in [Path("dataset"), Path("Dataset")]:
            if p.exists():
                path = p
                print(f"📂 Found dataset at: {path}")
                break
    
    if not path.exists():
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' not found. Please create it and add images/captions.")

    # Check for CSV format (prompts.csv)
    csv_file = path / "prompts.csv"
    images_dir = path / "images"
    
    if csv_file.exists() and images_dir.exists():
        print("📄 Found prompts.csv and images/ directory. Loading from CSV...")
        import csv
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Normalize column names just in case
            fieldnames = [f.lower() for f in reader.fieldnames]
            
            # Map known column names
            id_col = next((c for c in fieldnames if c in ['imgid', 'id', 'filename', 'image']), None)
            prompt_col = next((c for c in fieldnames if c in ['prompt', 'caption', 'text']), None)
            
            if not id_col or not prompt_col:
                print(f"⚠️  Could not identify columns in CSV. Found: {reader.fieldnames}")
                print("   Expected: imgId/id/filename and prompt/caption/text")
            else:
                 # Re-read with original fieldnames to access data
                 f.seek(0)
                 reader = csv.DictReader(f)
                 for row in reader:
                     # Find the column keys (case sensitive matching from header)
                     # We need to find the actual key that corresponds to our identified 'id_col'
                     row_keys = list(row.keys())
                     actual_id_key = next(k for k in row_keys if k.lower() == id_col)
                     actual_prompt_key = next(k for k in row_keys if k.lower() == prompt_col)
                     
                     img_id = row[actual_id_key]
                     prompt = row[actual_prompt_key]
                     
                     # Try to find the image file (check png, jpg, etc)
                     found_img = None
                     for ext in [".png", ".jpg", ".jpeg", ".webp"]:
                         img_candidate = images_dir / f"{img_id}{ext}"
                         if img_candidate.exists():
                             found_img = img_candidate
                             break
                     
                     if found_img:
                         dataset.append({"image": str(found_img), "text": prompt})
                     else:
                         print(f"⚠️  Image not found for ID: {img_id}")
            
        if len(dataset) > 0:
            print(f"✅ Loaded {len(dataset)} pairs from CSV.")
            return dataset

    # Fallback to Simple Format (flat directory)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = list(path.iterdir())
    image_files = [f for f in files if f.suffix.lower() in image_extensions]
    
    if not image_files:
        # Check if they are in 'images' subdir even without CSV
        if (path / "images").exists():
            files = list((path / "images").iterdir())
            image_files = [f for f in files if f.suffix.lower() in image_extensions]

    if not image_files:
        raise ValueError(f"No images found in '{path}'. Supported formats: {image_extensions}")
        
    print(f"📂 Found {len(image_files)} images in '{path}'")

    for img_path in image_files:
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists():
            with open(txt_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()
        else:
            # If no text file, use the filename as caption (cleaned up)
            caption = img_path.stem.replace("_", " ").replace("-", " ")
            print(f"⚠️  No text file for {img_path.name}, using filename as prompt: '{caption}'")
            
        dataset.append({"image": str(img_path), "text": caption})
        
    return dataset

class LocalDataset(Dataset):
    def __init__(self, data, tokenizer, size=512):
        self.data = data
        self.tokenizer = tokenizer
        self.size = size
        
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item["image"]
        text = item["text"]
        
        try:
            image = Image.open(image_path)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            
            pixel_values = self.image_transforms(image)
            
            # Tokenize text
            inputs = self.tokenizer(
                text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            )
            input_ids = inputs.input_ids[0]
            
            return {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
            }
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return a random other item to avoid crashing
            return self.__getitem__(random.randint(0, len(self.data) - 1))

def main():
    print("="*70)
    print("🚂 LOW-RANK ADAPTATION (LoRA) TRAINING")
    print("="*70)
    
    # accelerator setup
    accelerator = Accelerator(
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        mixed_precision=MIXED_PRECISION,
    )
    
    set_seed(SEED)
    
    # Load models
    print("\n📥 Loading models...")
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_NAME, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet")

    # Freeze frozen parameters
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    
    # Set up LoRA configuration
    print("🔧 Configuring LoRA adapters...")
    lora_config = LoraConfig(
        r=4,
        lora_alpha=16,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    
    # Add LoRA to UNet
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    
    # Optimizer
    optimizer_cls = torch.optim.AdamW
    
    optimizer = optimizer_cls(
        unet.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )
    
    # Dataset
    print("\n📚 Preparing dataset...")
    try:
        raw_data = load_image_dataset(DATASET_DIR, size=RESOLUTION)
    except Exception as e:
        print(f"\n❌ Dataset error: {e}")
        print(f"Please create a folder named '{DATASET_DIR}' and put your images there.")
        return

    train_dataset = LocalDataset(raw_data, tokenizer, size=RESOLUTION)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0, # Windows safe
    )
    
    # Prepare with accelerator
    unet, optimizer, train_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader
    )
    
    # Move other models to device and cast to dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    
    # Training Loop
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / GRADIENT_ACCUMULATION_STEPS)
    num_train_epochs = math.ceil(MAX_TRAIN_STEPS / num_update_steps_per_epoch)
    
    print(f"\n🚀 Starting training!")
    print(f"   Epochs: {num_train_epochs}")
    print(f"   Max Steps: {MAX_TRAIN_STEPS}")
    print(f"   Batch Size: {TRAIN_BATCH_SIZE}")
    print(f"   Gradient Accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    
    global_step = 0
    progress_bar = tqdm(range(MAX_TRAIN_STEPS), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    
    for epoch in range(num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                
                # Add noise to the latents according to the noise magnitude at each timestep
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                
                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                if global_step % CHECKPOINTING_STEPS == 0:
                    save_path = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    print(f"\n💾 Saved checkpoint to {save_path}")
            
            logs = {"loss": loss.detach().item(), "step": global_step}
            progress_bar.set_postfix(**logs)
            
            if global_step >= MAX_TRAIN_STEPS:
                break
        
        if global_step >= MAX_TRAIN_STEPS:
            break

    # Save final model
    print("\n" + "=" * 70)
    print("💾 Saving final LoRA weights...")
    print("=" * 70)
    
    accelerator.wait_for_everyone()
    unet = accelerator.unwrap_model(unet)
    unet.save_pretrained(OUTPUT_DIR)
    
    print(f"✅ Training complete! Model saved to '{OUTPUT_DIR}'")
    print("\nTo use your trained model:")
    print("1. In your generation script, add:")
    print(f'   pipe.load_lora_weights("{OUTPUT_DIR}")')
    print("2. Use the prompt/keyword you trained with!")

if __name__ == "__main__":
    main()
