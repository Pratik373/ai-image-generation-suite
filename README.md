# 🎨 AI Image Generation Suite

> **Production-ready AI image generation toolkit featuring Stable Diffusion and Kandinsky 2.2 models, optimized for RTX 4050 (8GB VRAM)**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-orange)](https://huggingface.co/)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Models Included](#-models-included)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Performance](#-performance)
- [Examples](#-examples)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌟 Overview

This project provides a **beginner-friendly yet production-quality** toolkit for AI image generation. It includes multiple state-of-the-art models with both **text-to-image** and **image-to-image** capabilities, all optimized for consumer-grade GPUs.

### Why This Project?

- ✅ **Fully Interactive** - No code editing required, just run and follow prompts
- ✅ **Production Optimized** - Memory-efficient, handles 8GB VRAM smoothly
- ✅ **Multiple Models** - Choose the best model for your use case
- ✅ **Well Documented** - Comprehensive guides and examples
- ✅ **Beginner Friendly** - Clear instructions, helpful error messages
- ✅ **Professional Code** - Clean, commented, maintainable

---

## ✨ Features

### 🎯 Core Capabilities

| Feature              | Description                                   |
| -------------------- | --------------------------------------------- |
| **Text-to-Image**    | Generate images from text descriptions        |
| **Image-to-Image**   | Modify existing images with AI                |
| **Multiple Models**  | Stable Diffusion v1.5, SDXL, Kandinsky 2.2    |
| **GPU Optimized**    | FP16 precision, attention slicing, VAE tiling |
| **Auto-caching**     | Models download once, cached forever          |
| **Interactive CLI**  | User-friendly command-line interface          |
| **Batch Processing** | Generate multiple variations easily           |

### 🔧 Technical Features

- **Memory Optimization**: FP16 precision, attention slicing, gradient checkpointing
- **CUDA Acceleration**: Full GPU support with fallback to CPU
- **Smart Caching**: Automatic model caching via Hugging Face Hub
- **Error Handling**: Comprehensive error messages and recovery
- **Prompt Engineering**: Built-in negative prompts and quality enhancement
- **Output Management**: Organized file structure with metadata

---

## 🤖 Models Included

### 1. **Stable Diffusion v1.5**

- **Size**: ~4GB
- **Speed**: ⚡⚡⚡ Fast (15-30s per image)
- **Quality**: ⭐⭐⭐⭐ Excellent
- **Best For**: General purpose, photorealistic images
- **Use Case**: Quick iterations, realistic photos, portraits

### 2. **Stable Diffusion XL (SDXL)**

- **Size**: ~7GB
- **Speed**: ⚡⚡ Moderate (30-60s per image)
- **Quality**: ⭐⭐⭐⭐⭐ Outstanding
- **Best For**: High-quality, detailed images
- **Use Case**: Professional artwork, detailed scenes, high resolution

### 3. **Kandinsky 2.2**

- **Size**: ~10GB
- **Speed**: ⚡⚡ Moderate (30-60s per image)
- **Quality**: ⭐⭐⭐⭐⭐ Outstanding
- **Best For**: Artistic images, image-to-image transformations
- **Use Case**: Art styles, photo modifications, creative edits

### Model Comparison

| Model             | Text-to-Image | Image-to-Image | Photorealism | Artistic   | Speed  |
| ----------------- | ------------- | -------------- | ------------ | ---------- | ------ |
| **SD v1.5**       | ✅            | ❌             | ⭐⭐⭐⭐     | ⭐⭐⭐     | ⚡⚡⚡ |
| **SDXL**          | ✅            | ❌             | ⭐⭐⭐⭐⭐   | ⭐⭐⭐⭐   | ⚡⚡   |
| **Kandinsky 2.2** | ✅            | ✅             | ⭐⭐⭐⭐     | ⭐⭐⭐⭐⭐ | ⚡⚡   |

---

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.10 or higher
- **GPU**: NVIDIA GPU with 6GB+ VRAM (RTX 3060, 4050, or better)
- **CUDA**: 11.8 or higher
- **Disk Space**: 15GB free (for models)
- **OS**: Windows, Linux, or macOS

### Installation (5 minutes)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/ai-image-generation-suite.git
cd ai-image-generation-suite

# 2. Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Install dependencies
pip install -r requirements.txt

# 4. Login to Hugging Face (get token from https://huggingface.co/settings/tokens)
huggingface-cli login
```

### First Run (30 seconds)

```bash
# For Stable Diffusion
python generate_image.py

# For Kandinsky (with image-to-image)
python kandinsky_img2img.py
```

That's it! The script will guide you through the rest. 🎉

---

## 📦 Installation

### Step 1: System Requirements

Verify your system meets the requirements:

```bash
# Check Python version
python --version  # Should be 3.10+

# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Step 2: Install PyTorch

**Windows/Linux (CUDA 11.8)**:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**macOS (CPU only)**:

```bash
pip install torch torchvision
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies include**:

- `diffusers` - Hugging Face diffusion models
- `transformers` - Model architectures
- `accelerate` - GPU optimization
- `safetensors` - Safe model loading
- `Pillow` - Image processing
- `invisible-watermark` - Image watermarking

### Step 4: Authenticate

```bash
# Login to Hugging Face
huggingface-cli login

# Verify login
huggingface-cli whoami
```

Get your token from: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

---

## 💻 Usage

### Stable Diffusion - Text-to-Image

```bash
python generate_image.py
```

**Interactive prompts**:

1. Choose model (SD v1.5 or SDXL)
2. Enter your text prompt
3. Enter negative prompt (optional)
4. Wait for generation
5. Image auto-opens and saves to `outputs/`

**Example session**:

```
Which model? 1 (SD v1.5)
Prompt: a beautiful sunset over mountains, highly detailed, 8k
Negative: blurry, low quality
✅ Image generated in 20 seconds!
```

### Kandinsky - Image-to-Image

```bash
python kandinsky_img2img.py
```

**Interactive prompts**:

1. Choose mode (Text-to-Image or Image-to-Image)
2. Enter your prompt
3. For img2img: Select source image and strength
4. Wait for generation
5. Image auto-opens and saves

**Example session**:

```
Mode: 2 (Image-to-Image)
Prompt: transform into oil painting style, vibrant colors
Image: C:\Users\YourName\Pictures\photo.jpg
Strength: 0.7
✅ Image modified in 45 seconds!
```

### Advanced Usage

**Edit generation parameters** in the scripts:

```python
# In generate_image.py or kandinsky_img2img.py

NUM_INFERENCE_STEPS = 50    # Higher = better quality (20-100)
GUIDANCE_SCALE = 7.5        # How closely to follow prompt (4-12)
IMAGE_HEIGHT = 512          # Output height (384, 512, 768)
IMAGE_WIDTH = 512           # Output width (384, 512, 768)
```

---

## 📁 Project Structure

```
ai-image-generation-suite/
├── 📄 README.md                    # This file
├── 📄 requirements.txt             # Python dependencies
├── 📄 .gitignore                   # Git ignore rules
│
├── 🐍 generate_image.py            # Stable Diffusion script
├── 🐍 kandinsky_img2img.py         # Kandinsky script
│
├── 📁 docs/                        # Documentation
│   ├── SETUP_GUIDE.md             # Detailed setup instructions
│   ├── QUICKSTART.md              # Quick reference guide
│   └── KANDINSKY_GUIDE.md         # Kandinsky usage guide
│
├── 📁 outputs/                     # Generated images (auto-created)
│   ├── generated_*.png            # Output images
│   └── generated_*_prompt.txt     # Prompt metadata
│
└── 📁 assets/                      # Project assets (optional)
    └── examples/                   # Example images
```

---

## ⚡ Performance

### RTX 4050 (8GB VRAM) - Tested Configuration

| Model     | Resolution | Steps | Time   | VRAM Usage |
| --------- | ---------- | ----- | ------ | ---------- |
| SD v1.5   | 512×512    | 30    | 15-20s | ~4GB       |
| SD v1.5   | 768×768    | 30    | 30-40s | ~6GB       |
| SDXL      | 512×512    | 30    | 30-45s | ~6GB       |
| SDXL      | 768×768    | 30    | 60-90s | ~7.5GB     |
| Kandinsky | 512×512    | 50    | 30-60s | ~5GB       |

### Optimization Features

- ✅ **FP16 Precision** - Halves memory usage
- ✅ **Attention Slicing** - Reduces VRAM peaks
- ✅ **VAE Tiling** - Prevents OOM on high-res
- ✅ **Model Caching** - Fast subsequent runs
- ✅ **Gradient Checkpointing** - Memory-efficient training

### Performance Tips

**For Best Quality**:

```python
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 8.5
```

**For Fastest Speed**:

```python
NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 7.5
IMAGE_HEIGHT = 384
IMAGE_WIDTH = 384
```

---

## 🎨 Examples

### Text-to-Image Examples

**Prompt**: `"a cozy cabin in snowy mountains at night, warm lights, starry sky, oil painting style"`

- **Model**: Kandinsky 2.2
- **Steps**: 50
- **Result**: Artistic, painterly image with rich colors

**Prompt**: `"professional photo of a modern office space, minimalist design, natural lighting, 8k"`

- **Model**: SDXL
- **Steps**: 40
- **Result**: Photorealistic architectural image

**Prompt**: `"cute cat wearing sunglasses, professional photography, studio lighting"`

- **Model**: SD v1.5
- **Steps**: 30
- **Result**: Realistic pet portrait

### Image-to-Image Examples

**Original**: Daytime landscape photo
**Prompt**: `"same scene but at sunset, golden hour lighting, warm tones"`
**Strength**: 0.4
**Result**: Photo with sunset lighting applied

**Original**: Regular portrait
**Prompt**: `"oil painting style, impressionist, vibrant brush strokes"`
**Strength**: 0.7
**Result**: Portrait transformed into oil painting

---

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Error**: `torch.cuda.OutOfMemoryError`

**Solutions**:

```python
# Option 1: Reduce resolution
IMAGE_HEIGHT = 384
IMAGE_WIDTH = 384

# Option 2: Reduce steps
NUM_INFERENCE_STEPS = 20

# Option 3: Use SD v1.5 instead of SDXL
```

#### 2. Model Download Fails

**Error**: `401 Unauthorized` or connection timeout

**Solutions**:

```bash
# Re-login to Hugging Face
huggingface-cli login

# Check internet connection
ping huggingface.co

# Clear cache and retry
rm -rf ~/.cache/huggingface/
```

#### 3. Slow Generation

**Causes**:

- First run (downloading model)
- CPU mode (no CUDA)
- High inference steps

**Solutions**:

```bash
# Verify CUDA is working
python -c "import torch; print(torch.cuda.is_available())"

# Update GPU drivers
# Reduce NUM_INFERENCE_STEPS
```

#### 4. Poor Image Quality

**Solutions**:

- Increase `NUM_INFERENCE_STEPS` to 50-100
- Add quality keywords: "highly detailed, 8k, professional"
- Use negative prompts: "blurry, low quality, distorted"
- Try different models (SDXL for photorealism, Kandinsky for art)

### Getting Help

1. Check the [docs/](docs/) folder for detailed guides
2. Review error messages carefully
3. Verify CUDA installation: `nvidia-smi`
4. Check Hugging Face status: [status.huggingface.co](https://status.huggingface.co)

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Ways to Contribute

- 🐛 **Report bugs** - Open an issue with details
- 💡 **Suggest features** - Share your ideas
- 📝 **Improve docs** - Fix typos, add examples
- 🔧 **Submit PRs** - Add new models, optimizations
- ⭐ **Star the repo** - Show your support!

### Development Setup

```bash
# Fork and clone
git clone https://github.com/yourusername/ai-image-generation-suite.git
cd ai-image-generation-suite

# Create branch
git checkout -b feature/your-feature

# Make changes and test
python generate_image.py

# Commit and push
git add .
git commit -m "Add: your feature description"
git push origin feature/your-feature
```

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Model Licenses

- **Stable Diffusion v1.5**: [CreativeML Open RAIL-M](https://huggingface.co/spaces/CompVis/stable-diffusion-license)
- **SDXL**: [CreativeML Open RAIL++-M](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)
- **Kandinsky 2.2**: [Apache 2.0](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder/blob/main/LICENSE)

---

## � Acknowledgments

- [Hugging Face](https://huggingface.co/) - For the amazing Diffusers library
- [Stability AI](https://stability.ai/) - For Stable Diffusion models
- [Kandinsky](https://github.com/ai-forever/Kandinsky-2) - For Kandinsky models
- [PyTorch](https://pytorch.org/) - For the deep learning framework

---

## 📊 Project Stats

- **Models**: 3 (SD v1.5, SDXL, Kandinsky 2.2)
- **Features**: Text-to-Image, Image-to-Image
- **Lines of Code**: ~800
- **Documentation**: 4 comprehensive guides
- **Supported GPUs**: NVIDIA RTX series (6GB+ VRAM)

---

## 🔗 Links

- **Documentation**: [docs/](docs/)
- **Hugging Face**: [https://huggingface.co/](https://huggingface.co/)
- **PyTorch**: [https://pytorch.org/](https://pytorch.org/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/ai-image-generation-suite/issues)

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

<div align="center">

**Made with ❤️ by [Your Name]**

_Powered by Stable Diffusion, Kandinsky, and Hugging Face Diffusers_

[⬆ Back to Top](#-ai-image-generation-suite)

</div>
