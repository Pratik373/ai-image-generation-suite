"""
Simple Image-to-Image Generator using Stable Diffusion
Optimized for RTX 4050 (8GB VRAM)

This is a simpler, faster alternative to Kandinsky for image modifications.
"""

import os
import torch
from datetime import datetime
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from huggingface_hub import HfFolder

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = Path("img2img_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Generation parameters
NUM_INFERENCE_STEPS = 25  # Reduced for memory
GUIDANCE_SCALE = 7.5
STRENGTH = 0.75  # How much to modify (0.0 = no change, 1.0 = complete change)
MAX_IMAGE_SIZE = 512  # Maximum dimension to prevent OOM

# ============================================================================
# DEVICE SETUP
# ============================================================================

def setup_device():
    """Configure GPU/CPU."""
    print("\n" + "=" * 70)
    print("🖥️  SYSTEM CHECK")
    print("=" * 70)
    
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = "cpu"
        dtype = torch.float32
        print("⚠️  Using CPU (will be slow)")
    
    return device, dtype

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(device, dtype):
    """Load Stable Diffusion img2img model."""
    print("\n" + "=" * 70)
    print("📥 LOADING IMAGE-TO-IMAGE MODEL")
    print("=" * 70)
    print("\n📦 Loading Stable Diffusion v1.5 img2img...")
    print("   (First run downloads ~4GB, then cached)")
    
    try:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
            use_safetensors=True,
        )
        
        # AGGRESSIVE MEMORY OPTIMIZATIONS for 6GB VRAM
        print("\n⚙️  Applying memory optimizations...")
        
        # Enable CPU offloading (moves models to CPU when not in use)
        pipe.enable_model_cpu_offload()
        
        # Enable attention slicing
        pipe.enable_attention_slicing(1)
        
        # Enable VAE slicing
        if hasattr(pipe, 'vae'):
            pipe.vae.enable_slicing()
        
        print("✅ Model loaded with CPU offloading!")
        print("   (This uses less VRAM but is slightly slower)")
        return pipe
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise

# ============================================================================
# USER INPUT
# ============================================================================

def get_source_image():
    """Get source image from user."""
    print("\n" + "=" * 70)
    print("🖼️  SOURCE IMAGE")
    print("=" * 70)
    print("\nDrag and drop your image here, or paste the full path:")
    print("(Example: C:\\Users\\YourName\\Pictures\\photo.jpg)")
    print()
    
    image_path = input("📁 Image path: ").strip().strip('"')
    
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return None
    
    try:
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        
        # Resize if too large (to prevent OOM)
        max_dim = max(image.size)
        if max_dim > MAX_IMAGE_SIZE:
            ratio = MAX_IMAGE_SIZE / max_dim
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            print(f"✅ Image loaded and resized: {original_size} → {image.size}")
            print(f"   (Resized to fit {MAX_IMAGE_SIZE}px for memory efficiency)")
        else:
            print(f"✅ Image loaded: {image.size[0]}x{image.size[1]}")
        
        return image
    except Exception as e:
        print(f"❌ Failed to load image: {e}")
        return None

def get_prompt():
    """Get modification prompt."""
    print("\n" + "=" * 70)
    print("✏️  MODIFICATION PROMPT")
    print("=" * 70)
    print("\nHow should the image be modified?")
    print()
    print("Examples:")
    print("  - make it look like an oil painting")
    print("  - add sunset lighting, golden hour")
    print("  - transform into anime style")
    print("  - make it winter scene with snow")
    print()
    
    prompt = input("✏️  Your prompt: ").strip()
    
    if not prompt:
        prompt = "enhance quality, professional photography"
        print(f"⚠️  Using default: {prompt}")
    
    print(f"\n✅ Prompt: {prompt}")
    
    # Negative prompt
    print("\n(Optional) What to avoid?")
    print("(Press Enter to use default)")
    negative_prompt = input("🚫 Negative: ").strip()
    
    if not negative_prompt:
        negative_prompt = "blurry, low quality, distorted, ugly, deformed"
        print(f"   Using default: {negative_prompt}")
    
    return prompt, negative_prompt

def get_strength():
    """Get modification strength."""
    print("\n" + "=" * 70)
    print("⚙️  MODIFICATION STRENGTH")
    print("=" * 70)
    print("\nHow much should the image change?")
    print("  0.3 = Subtle (lighting, colors)")
    print("  0.5 = Moderate (add/remove elements)")
    print("  0.7 = Strong (style transformation)")
    print("  0.9 = Extreme (almost new image)")
    print()
    
    strength_input = input("Enter strength (0.0-1.0) [default: 0.75]: ").strip()
    
    try:
        strength = float(strength_input) if strength_input else 0.75
        strength = max(0.0, min(1.0, strength))
        print(f"✅ Strength: {strength}")
        return strength
    except:
        print("⚠️  Invalid, using default: 0.75")
        return 0.75

# ============================================================================
# IMAGE GENERATION
# ============================================================================

def modify_image(pipe, device, image, prompt, negative_prompt, strength):
    """Modify image based on prompt."""
    print("\n" + "=" * 70)
    print("🎨 MODIFYING IMAGE")
    print("=" * 70)
    print(f"\n📝 Prompt: {prompt}")
    print(f"🚫 Negative: {negative_prompt}")
    print(f"💪 Strength: {strength}")
    print(f"🔢 Steps: {NUM_INFERENCE_STEPS}")
    print("\n⏳ Processing... (15-30 seconds)")
    
    try:
        # Generate
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                strength=strength,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
            )
        
        output_image = result.images[0]
        print("✅ Image modified successfully!")
        return output_image
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        raise

# ============================================================================
# SAVE OUTPUT
# ============================================================================

def save_image(image, prompt):
    """Save modified image."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"img2img_{timestamp}.png"
    filepath = OUTPUT_DIR / filename
    
    # Save image
    image.save(filepath)
    
    # Save prompt
    prompt_file = OUTPUT_DIR / f"img2img_{timestamp}_prompt.txt"
    with open(prompt_file, 'w', encoding='utf-8') as f:
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Timestamp: {timestamp}\n")
    
    print(f"\n💾 Image saved: {filepath}")
    print(f"📝 Prompt saved: {prompt_file}")
    
    # Try to open
    try:
        os.startfile(filepath)
        print("🖼️  Opening image...")
    except:
        print("   (Check outputs folder)")
    
    return filepath

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution."""
    print("\n")
    print("=" * 70)
    print("🎨 IMAGE-TO-IMAGE GENERATOR (Stable Diffusion)")
    print("=" * 70)
    print("\nFast & Reliable - Optimized for RTX 4050")
    print()
    
    try:
        # Check login
        if not HfFolder.get_token():
            print("⚠️  Not logged in to Hugging Face")
            print("   Run: huggingface-cli login")
            return
        
        # Setup
        device, dtype = setup_device()
        
        # Load model
        pipe = load_model(device, dtype)
        
        # Get inputs
        source_image = get_source_image()
        if source_image is None:
            return
        
        prompt, negative_prompt = get_prompt()
        strength = get_strength()
        
        # Generate
        output_image = modify_image(
            pipe, device, source_image,
            prompt, negative_prompt, strength
        )
        
        # Save
        filepath = save_image(output_image, prompt)
        
        # Cleanup
        if device == "cuda":
            torch.cuda.empty_cache()
        
        print("\n" + "=" * 70)
        print("✨ SUCCESS!")
        print("=" * 70)
        print(f"\n📂 Location: {filepath.absolute()}")
        
        # Ask for another
        print("\n" + "=" * 70)
        another = input("\n🔄 Modify another image? (y/n): ").strip().lower()
        if another == 'y':
            print("\n" * 2)
            main()
        
    except torch.cuda.OutOfMemoryError:
        print("\n" + "=" * 70)
        print("❌ OUT OF MEMORY")
        print("=" * 70)
        print("\n🔧 Solutions:")
        print("1. Close other applications")
        print("2. Reduce strength to 0.5")
        print("3. Use smaller source image")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Cancelled")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
