"""
Kandinsky 2.2 Image-to-Image Generator
Optimized for RTX 4050 (8GB VRAM)

Features:
- Text-to-Image generation
- Image-to-Image modification
- Interactive prompts
- Automatic model download and caching
"""

import os
import torch
from datetime import datetime
from pathlib import Path
from PIL import Image
from diffusers import KandinskyV22Pipeline, KandinskyV22Img2ImgPipeline, KandinskyV22PriorPipeline
from huggingface_hub import login, HfFolder

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Generation parameters
NUM_INFERENCE_STEPS = 50  # Kandinsky works best with 50-100 steps
GUIDANCE_SCALE = 4.0      # Kandinsky uses lower guidance (4-7)
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
STRENGTH = 0.75           # For img2img: 0.0 (no change) to 1.0 (complete change)

# ============================================================================
# AUTHENTICATION
# ============================================================================

def check_and_login():
    """Check if user is logged into Hugging Face."""
    print("=" * 70)
    print("🔐 HUGGING FACE AUTHENTICATION")
    print("=" * 70)
    
    token = HfFolder.get_token()
    
    if token:
        print("✅ Already logged in to Hugging Face!")
        try:
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
            login(token=token, add_to_git_credential=True)
            print("✅ Successfully logged in!")
            return True
        except Exception as e:
            print(f"❌ Login failed: {e}")
            return False
    
    return True

# ============================================================================
# DEVICE SETUP
# ============================================================================

def setup_device():
    """Configure the compute device (GPU/CPU)."""
    print("\n" + "=" * 70)
    print("🖥️  SYSTEM CHECK")
    print("=" * 70)
    
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = "cpu"
        dtype = torch.float32
        print("⚠️  No GPU detected, using CPU (will be VERY slow)")
    
    return device, dtype

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_kandinsky_models(device, dtype):
    """
    Load Kandinsky 2.2 models.
    Kandinsky uses two models: Prior (text understanding) and Decoder (image generation)
    """
    print("\n" + "=" * 70)
    print("📥 LOADING KANDINSKY 2.2 MODELS")
    print("=" * 70)
    print("\n📦 Loading models...")
    print("   (First run will download ~10GB total, then cached)")
    print("   This may take 10-15 minutes on first run...")
    
    try:
        # Load the prior model (converts text to image embeddings)
        print("\n1️⃣  Loading Prior model (text encoder)...")
        prior = KandinskyV22PriorPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior",
            torch_dtype=dtype
        ).to(device)
        
        # Load the decoder model (generates images)
        print("2️⃣  Loading Decoder model (image generator)...")
        pipe_txt2img = KandinskyV22Pipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder",
            torch_dtype=dtype
        ).to(device)
        
        # Load img2img pipeline
        print("3️⃣  Loading Image-to-Image model...")
        pipe_img2img = KandinskyV22Img2ImgPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder",
            torch_dtype=dtype
        ).to(device)
        
        print("\n✅ All models loaded successfully!")
        
        return prior, pipe_txt2img, pipe_img2img
        
    except Exception as e:
        print(f"\n❌ Failed to load models: {e}")
        print("\n🔧 Possible solutions:")
        print("1. Check your internet connection")
        print("2. Verify you're logged in to Hugging Face")
        print("3. Ensure you have enough disk space (~10GB)")
        raise

# ============================================================================
# USER INPUT
# ============================================================================

def get_generation_mode():
    """Ask user what they want to do."""
    print("\n" + "=" * 70)
    print("🎨 GENERATION MODE")
    print("=" * 70)
    print("\nWhat would you like to do?")
    print()
    print("1. Text-to-Image (Generate new image from text)")
    print("2. Image-to-Image (Modify an existing image)")
    print()
    
    choice = input("Enter choice (1 or 2): ").strip()
    return choice

def get_text_prompt():
    """Get text prompt from user."""
    print("\n" + "=" * 70)
    print("✏️  TEXT PROMPT")
    print("=" * 70)
    print("\nDescribe the image you want to generate:")
    print("(Be descriptive for best results)")
    print()
    print("Examples:")
    print("  - a beautiful landscape with mountains and lake, oil painting style")
    print("  - futuristic city at night with neon lights, cyberpunk")
    print("  - cute cat wearing a wizard hat, fantasy art")
    print()
    
    prompt = input("✏️  Your prompt: ").strip()
    
    if not prompt:
        prompt = "a beautiful landscape with mountains and a lake, highly detailed"
        print(f"⚠️  Using default: {prompt}")
    
    print(f"\n✅ Prompt: {prompt}")
    
    # Negative prompt
    print("\n(Optional) What should the image NOT contain?")
    print("(Press Enter to use default)")
    negative_prompt = input("🚫 Negative prompt: ").strip()
    
    if not negative_prompt:
        negative_prompt = "low quality, bad anatomy, blurry, pixelated, ugly"
        print(f"   Using default: {negative_prompt}")
    
    return prompt, negative_prompt

def get_source_image():
    """Get source image path from user."""
    print("\n" + "=" * 70)
    print("🖼️  SOURCE IMAGE")
    print("=" * 70)
    print("\nEnter the path to your source image:")
    print("(You can drag and drop the file here)")
    print()
    
    image_path = input("📁 Image path: ").strip().strip('"')
    
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return None
    
    try:
        image = Image.open(image_path).convert("RGB")
        # Resize to target dimensions
        image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.LANCZOS)
        print(f"✅ Image loaded: {image_path}")
        return image
    except Exception as e:
        print(f"❌ Failed to load image: {e}")
        return None

def get_modification_strength():
    """Get how much to modify the image."""
    print("\n" + "=" * 70)
    print("⚙️  MODIFICATION STRENGTH")
    print("=" * 70)
    print("\nHow much should the image be modified?")
    print("  0.3 = Subtle changes (keep most of original)")
    print("  0.5 = Moderate changes")
    print("  0.7 = Strong changes (default)")
    print("  0.9 = Almost completely new image")
    print()
    
    strength_input = input("Enter strength (0.0-1.0) [default: 0.75]: ").strip()
    
    try:
        strength = float(strength_input) if strength_input else 0.75
        strength = max(0.0, min(1.0, strength))  # Clamp between 0 and 1
        print(f"✅ Strength: {strength}")
        return strength
    except:
        print("⚠️  Invalid input, using default: 0.75")
        return 0.75

# ============================================================================
# IMAGE GENERATION
# ============================================================================

def generate_text2img(prior, pipe, device, prompt, negative_prompt):
    """Generate image from text."""
    print("\n" + "=" * 70)
    print("🎨 GENERATING IMAGE FROM TEXT")
    print("=" * 70)
    print(f"\n📝 Prompt: {prompt}")
    print(f"🚫 Negative: {negative_prompt}")
    print(f"📐 Resolution: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print(f"🔢 Steps: {NUM_INFERENCE_STEPS}")
    print("\n⏳ Generating... (this will take 30-60 seconds)")
    
    # Step 1: Generate image embeddings from text using prior
    print("\n1️⃣  Encoding text prompt...")
    image_embeds, negative_image_embeds = prior(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=25,
        guidance_scale=4.0
    ).to_tuple()
    
    # Step 2: Generate image from embeddings
    print("2️⃣  Generating image...")
    image = pipe(
        image_embeds=image_embeds,
        negative_image_embeds=negative_image_embeds,
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE
    ).images[0]
    
    print("✅ Image generated successfully!")
    return image

def generate_img2img(prior, pipe, device, prompt, negative_prompt, source_image, strength):
    """Modify existing image based on text."""
    print("\n" + "=" * 70)
    print("🎨 MODIFYING IMAGE")
    print("=" * 70)
    print(f"\n📝 Prompt: {prompt}")
    print(f"🚫 Negative: {negative_prompt}")
    print(f"💪 Strength: {strength}")
    print(f"📐 Resolution: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print("\n⏳ Generating... (this will take 30-60 seconds)")
    
    # Step 1: Generate image embeddings from text
    print("\n1️⃣  Encoding text prompt...")
    image_embeds, negative_image_embeds = prior(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=25,
        guidance_scale=4.0
    ).to_tuple()
    
    # Step 2: Modify image
    print("2️⃣  Modifying image...")
    image = pipe(
        image=source_image,
        image_embeds=image_embeds,
        negative_image_embeds=negative_image_embeds,
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        strength=strength,
        guidance_scale=GUIDANCE_SCALE
    ).images[0]
    
    print("✅ Image modified successfully!")
    return image

# ============================================================================
# SAVE OUTPUT
# ============================================================================

def save_image(image, prompt, mode="txt2img"):
    """Save the generated image."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"kandinsky_{mode}_{timestamp}.png"
    filepath = OUTPUT_DIR / filename
    
    # Save image
    image.save(filepath)
    
    # Save prompt
    prompt_file = OUTPUT_DIR / f"kandinsky_{mode}_{timestamp}_prompt.txt"
    with open(prompt_file, 'w', encoding='utf-8') as f:
        f.write(f"Mode: {mode}\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Timestamp: {timestamp}\n")
    
    print(f"\n💾 Image saved: {filepath}")
    print(f"📝 Prompt saved: {prompt_file}")
    
    # Try to open the image
    try:
        os.startfile(filepath)
        print("🖼️  Opening image...")
    except:
        print("   (Open the outputs folder to view)")
    
    return filepath

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution flow."""
    print("\n")
    print("=" * 70)
    print("🎨 KANDINSKY 2.2 - IMAGE GENERATOR & MODIFIER")
    print("=" * 70)
    print("\nOptimized for RTX 4050 (8GB VRAM)")
    print()
    
    try:
        # Step 1: Authentication
        if not check_and_login():
            return
        
        # Step 2: Setup device
        device, dtype = setup_device()
        
        # Step 3: Load models
        prior, pipe_txt2img, pipe_img2img = load_kandinsky_models(device, dtype)
        
        # Step 4: Get generation mode
        mode = get_generation_mode()
        
        if mode == "2":
            # Image-to-Image mode
            prompt, negative_prompt = get_text_prompt()
            source_image = get_source_image()
            
            if source_image is None:
                print("❌ Failed to load source image. Exiting.")
                return
            
            strength = get_modification_strength()
            
            # Generate
            image = generate_img2img(
                prior, pipe_img2img, device, 
                prompt, negative_prompt, 
                source_image, strength
            )
            
            filepath = save_image(image, prompt, mode="img2img")
            
        else:
            # Text-to-Image mode (default)
            prompt, negative_prompt = get_text_prompt()
            
            # Generate
            image = generate_text2img(
                prior, pipe_txt2img, device,
                prompt, negative_prompt
            )
            
            filepath = save_image(image, prompt, mode="txt2img")
        
        # Cleanup
        if device == "cuda":
            torch.cuda.empty_cache()
        
        print("\n" + "=" * 70)
        print("✨ SUCCESS! Your image is ready!")
        print("=" * 70)
        print(f"\n📂 Location: {filepath.absolute()}")
        
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
        print("\n🔧 Your GPU ran out of memory. Try:")
        print("1. Close other applications using GPU")
        print("2. Reduce resolution to 384x384 (edit IMAGE_HEIGHT/WIDTH)")
        print("3. Reduce NUM_INFERENCE_STEPS to 30")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Cancelled by user")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check internet connection")
        print("2. Verify Hugging Face token is valid")
        print("3. Ensure CUDA is working")
        import traceback
        traceback.print_exc()

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
