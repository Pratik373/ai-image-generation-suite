"""
Stable Diffusion Image Generator - FULLY AUTOMATED
Optimized for RTX 4050 (8GB VRAM)

Just run this script and follow the prompts!
"""

import os
import sys
import torch
import subprocess
from datetime import datetime
from pathlib import Path
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from huggingface_hub import login, HfFolder

# ============================================================================
# CONFIGURATION
# ============================================================================

# Output configuration
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Generation parameters (optimized for RTX 4050)
NUM_INFERENCE_STEPS = 30  # Higher = better quality but slower (20-50 recommended)
GUIDANCE_SCALE = 7.5      # How closely to follow prompt (7-12 recommended)
IMAGE_HEIGHT = 512        # Must be divisible by 8
IMAGE_WIDTH = 512         # Must be divisible by 8

# ============================================================================
# HUGGING FACE AUTHENTICATION
# ============================================================================

def check_and_login():
    """
    Check if user is logged into Hugging Face, if not, ask for token.
    """
    print("=" * 70)
    print("🔐 HUGGING FACE AUTHENTICATION")
    print("=" * 70)
    
    # Check if already logged in
    token = HfFolder.get_token()
    
    if token:
        print("✅ Already logged in to Hugging Face!")
        try:
            # Verify token works
            from huggingface_hub import whoami
            user_info = whoami(token)
            print(f"   Logged in as: {user_info['name']}")
            return True
        except:
            print("⚠️  Stored token is invalid, need to re-login")
            token = None
    
    if not token:
        print("\n📝 Please provide your Hugging Face token")
        print("   Get it from: https://huggingface.co/settings/tokens")
        print()
        
        token = input("🔑 Paste your Hugging Face token here: ").strip()
        
        if not token:
            print("❌ No token provided. Exiting.")
            return False
        
        try:
            # Login with the token
            login(token=token, add_to_git_credential=True)
            print("✅ Successfully logged in!")
            return True
        except Exception as e:
            print(f"❌ Login failed: {e}")
            print("   Please check your token and try again.")
            return False
    
    return True

# ============================================================================
# GET USER INPUT
# ============================================================================

def get_user_prompt():
    """
    Ask user what image they want to generate.
    """
    print("\n" + "=" * 70)
    print("🎨 IMAGE GENERATION")
    print("=" * 70)
    print("\nWhat image would you like to generate?")
    print("(Be descriptive for best results)")
    print()
    print("Examples:")
    print("  - a beautiful sunset over mountains, highly detailed, 8k")
    print("  - a cute cat wearing sunglasses, professional photo")
    print("  - futuristic cyberpunk city at night, neon lights, cinematic")
    print()
    
    prompt = input("✏️  Your prompt: ").strip()
    
    if not prompt:
        print("⚠️  No prompt provided, using default...")
        prompt = "a beautiful landscape with mountains and a lake, highly detailed, 8k, photorealistic"
    
    print(f"\n✅ Prompt: {prompt}")
    
    # Ask for negative prompt (optional)
    print("\n(Optional) What should the image NOT contain?")
    print("(Press Enter to skip)")
    negative_prompt = input("🚫 Negative prompt: ").strip()
    
    if not negative_prompt:
        negative_prompt = "blurry, low quality, distorted, ugly, deformed"
        print(f"   Using default: {negative_prompt}")
    
    return prompt, negative_prompt

# ============================================================================
# DEVICE SETUP
# ============================================================================

def setup_device():
    """
    Configure the compute device (GPU/CPU).
    Returns the device and data type for model loading.
    """
    print("\n" + "=" * 70)
    print("🖥️  SYSTEM CHECK")
    print("=" * 70)
    
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16  # FP16 uses half the VRAM of FP32
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = "cpu"
        dtype = torch.float32
        print("⚠️  No GPU detected, using CPU (will be VERY slow)")
        print("   Consider installing CUDA-enabled PyTorch")
    
    return device, dtype

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(device, dtype):
    """
    Load the Stable Diffusion model with optimizations.
    
    Optimizations applied:
    - FP16 precision (half memory)
    - Attention slicing (reduces VRAM during generation)
    - Efficient scheduler (faster inference)
    """
    print("\n" + "=" * 70)
    print("📥 LOADING MODEL")
    print("=" * 70)
    
    # Ask user which model to use
    print("\nWhich Stable Diffusion model would you like to use?")
    print()
    print("1. Stable Diffusion v1.5 (Recommended, faster, 4GB)")
    print("2. Stable Diffusion XL Base 1.0 (Better quality, slower, 7GB)")
    print()
    
    choice = input("Enter choice (1 or 2) [default: 1]: ").strip()
    
    if choice == "2":
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        print(f"\n✅ Selected: SDXL Base 1.0")
    else:
        model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        print(f"\n✅ Selected: SD v1.5")
    
    print(f"\n📦 Loading {model_id}...")
    print("   (First run will download the model, then it's cached locally)")
    print("   This may take 5-10 minutes on first run...")
    
    try:
        # Load the pipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
            use_safetensors=True,
        )
        
        # Move model to GPU
        pipe = pipe.to(device)
        
        # ====================================================================
        # MEMORY OPTIMIZATIONS
        # ====================================================================
        
        # Enable attention slicing - reduces memory usage during generation
        pipe.enable_attention_slicing()
        
        # Enable VAE slicing - prevents OOM when decoding large images
        pipe.enable_vae_slicing()
        
        # Use a faster scheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
        print("✅ Model loaded successfully!")
        
        return pipe
        
    except Exception as e:
        print(f"\n❌ Failed to load model: {e}")
        print("\n🔧 Possible solutions:")
        print("1. Check your internet connection")
        print("2. Verify you're logged in to Hugging Face")
        print("3. Ensure you have enough disk space (~7GB)")
        raise

# ============================================================================
# IMAGE GENERATION
# ============================================================================

def generate_image(pipe, device, prompt, negative_prompt):
    """
    Generate an image from the text prompt.
    
    Returns:
        PIL.Image: Generated image
    """
    print("\n" + "=" * 70)
    print("🎨 GENERATING IMAGE")
    print("=" * 70)
    print(f"\n📝 Prompt: {prompt}")
    print(f"🚫 Negative: {negative_prompt}")
    print(f"📐 Resolution: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print(f"🔢 Steps: {NUM_INFERENCE_STEPS}")
    print(f"🎯 Guidance Scale: {GUIDANCE_SCALE}")
    print("\n⏳ Generating... (this will take 15-30 seconds)")
    
    # Generate with random seed each time
    generator = torch.Generator(device=device).manual_seed(torch.randint(0, 1000000, (1,)).item())
    
    # Generate the image
    with torch.inference_mode():
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            height=IMAGE_HEIGHT,
            width=IMAGE_WIDTH,
            generator=generator,
        )
    
    image = output.images[0]
    
    print("✅ Image generated successfully!")
    
    return image

# ============================================================================
# SAVE OUTPUT
# ============================================================================

def save_image(image, prompt):
    """
    Save the generated image with a timestamp.
    
    Returns:
        Path: Path to saved image
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"generated_{timestamp}.png"
    filepath = OUTPUT_DIR / filename
    
    # Save the image
    image.save(filepath)
    
    # Also save the prompt as a text file
    prompt_file = OUTPUT_DIR / f"generated_{timestamp}_prompt.txt"
    with open(prompt_file, 'w', encoding='utf-8') as f:
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Timestamp: {timestamp}\n")
    
    print(f"\n💾 Image saved: {filepath}")
    print(f"📝 Prompt saved: {prompt_file}")
    
    # Try to open the image
    try:
        os.startfile(filepath)  # Windows
        print("🖼️  Opening image...")
    except:
        print("   (Open the outputs folder to view)")
    
    return filepath

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution flow - fully automated!
    """
    print("\n")
    print("=" * 70)
    print("🚀 STABLE DIFFUSION IMAGE GENERATOR - AUTOMATED")
    print("=" * 70)
    print("\nOptimized for RTX 4050 (8GB VRAM)")
    print()
    
    try:
        # Step 1: Login to Hugging Face
        if not check_and_login():
            return
        
        # Step 2: Get user's desired image
        prompt, negative_prompt = get_user_prompt()
        
        # Step 3: Setup device
        device, dtype = setup_device()
        
        # Step 4: Load model
        pipe = load_model(device, dtype)
        
        # Step 5: Generate image
        image = generate_image(pipe, device, prompt, negative_prompt)
        
        # Step 6: Save output
        filepath = save_image(image, prompt)
        
        # Cleanup
        if device == "cuda":
            torch.cuda.empty_cache()
        
        print("\n" + "=" * 70)
        print("✨ SUCCESS! Your image is ready!")
        print("=" * 70)
        print(f"\n📂 Location: {filepath.absolute()}")
        print("\n🎉 Want to generate another? Just run this script again!")
        
        # Ask if user wants to generate another
        print("\n" + "=" * 70)
        another = input("\n🔄 Generate another image? (y/n): ").strip().lower()
        if another == 'y':
            print("\n" * 2)
            main()  # Recursive call
        
    except torch.cuda.OutOfMemoryError:
        print("\n" + "=" * 70)
        print("❌ CUDA OUT OF MEMORY ERROR")
        print("=" * 70)
        print("\n🔧 Your GPU ran out of memory. Try these solutions:")
        print("\n1. Close other applications using GPU")
        print("2. Reduce resolution to 384x384 (edit IMAGE_HEIGHT/WIDTH in script)")
        print("3. Choose SD v1.5 instead of SDXL")
        print("\nThen run the script again.")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Cancelled by user")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify Hugging Face token is valid")
        print("3. Ensure CUDA is installed: python -c 'import torch; print(torch.cuda.is_available())'")
        import traceback
        traceback.print_exc()

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
