# 🎨 Kandinsky 2.2 - Complete Guide

## What is Kandinsky 2.2?

Kandinsky 2.2 is a powerful AI image generation model that excels at:

- **Text-to-Image**: Create images from text descriptions
- **Image-to-Image**: Modify existing images with text prompts
- **High Quality**: Better understanding of complex prompts
- **Artistic Style**: Produces beautiful, artistic results

## 🚀 Quick Start

### Run the Script

```bash
python kandinsky_img2img.py
```

### What You'll Be Asked

1. **Mode Selection**:

   - Option 1: Text-to-Image (create new image)
   - Option 2: Image-to-Image (modify existing image)

2. **Text Prompt**: Describe what you want

3. **For Image-to-Image**:
   - Source image path
   - Modification strength (0.0-1.0)

## 📖 Usage Examples

### Example 1: Text-to-Image

```
Mode: 1
Prompt: "a magical forest with glowing mushrooms, fantasy art, detailed"
Negative: "low quality, blurry"
```

**Result**: Brand new image created from your description

### Example 2: Image-to-Image (Subtle Changes)

```
Mode: 2
Prompt: "make it sunset, golden hour lighting"
Source: C:\Users\YourName\Pictures\photo.jpg
Strength: 0.3
```

**Result**: Your photo with sunset lighting added

### Example 3: Image-to-Image (Strong Changes)

```
Mode: 2
Prompt: "transform into oil painting style, vibrant colors"
Source: C:\Users\YourName\Pictures\photo.jpg
Strength: 0.8
```

**Result**: Photo transformed into oil painting style

## 🎯 Modification Strength Guide

| Strength | Effect   | Use Case                                   |
| -------- | -------- | ------------------------------------------ |
| 0.1-0.3  | Subtle   | Color correction, lighting adjustments     |
| 0.4-0.6  | Moderate | Style changes, add/remove small elements   |
| 0.7-0.8  | Strong   | Major transformations, different art style |
| 0.9-1.0  | Extreme  | Almost completely new image                |

## 💡 Prompt Tips

### Good Prompts

✅ **Specific and descriptive**:

```
"a cozy cabin in snowy mountains at night, warm lights in windows,
starry sky, oil painting style, highly detailed"
```

✅ **Include style**:

```
"portrait of a cat, digital art, vibrant colors, detailed fur"
```

✅ **Mention quality**:

```
"futuristic city, cyberpunk, neon lights, highly detailed, 8k"
```

### Avoid

❌ Too vague: "a picture"
❌ Too short: "cat"
❌ Contradictory: "realistic anime" (unless that's what you want)

## 🎨 Creative Use Cases

### 1. Photo Enhancement

- **Original**: Regular photo
- **Prompt**: "professional photography, golden hour lighting, enhanced colors"
- **Strength**: 0.3-0.4

### 2. Style Transfer

- **Original**: Any photo
- **Prompt**: "oil painting style, impressionist, vibrant brush strokes"
- **Strength**: 0.7-0.8

### 3. Scene Transformation

- **Original**: Daytime photo
- **Prompt**: "same scene but at night, moonlight, stars"
- **Strength**: 0.5-0.6

### 4. Artistic Interpretation

- **Original**: Sketch or drawing
- **Prompt**: "detailed digital art, vibrant colors, fantasy style"
- **Strength**: 0.6-0.7

### 5. Object Addition/Removal

- **Original**: Landscape
- **Prompt**: "add a rainbow and birds in the sky"
- **Strength**: 0.4-0.5

## ⚙️ Advanced Settings

You can edit these in the script (`kandinsky_img2img.py`):

```python
# Line 23-27
NUM_INFERENCE_STEPS = 50    # Higher = better quality (30-100)
GUIDANCE_SCALE = 4.0        # How closely to follow prompt (3-7)
IMAGE_HEIGHT = 512          # Output height
IMAGE_WIDTH = 512           # Output width
STRENGTH = 0.75             # Default modification strength
```

### Quality vs Speed

**Best Quality**:

```python
NUM_INFERENCE_STEPS = 100
GUIDANCE_SCALE = 6.0
```

**Fastest**:

```python
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 4.0
```

**Balanced (Recommended)**:

```python
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 4.0
```

## 🔧 Troubleshooting

### Out of Memory Error

**Solution 1**: Reduce resolution

```python
IMAGE_HEIGHT = 384
IMAGE_WIDTH = 384
```

**Solution 2**: Reduce steps

```python
NUM_INFERENCE_STEPS = 30
```

**Solution 3**: Close other applications

### Slow Generation

- **First run**: Downloads ~10GB model (one-time)
- **Expected time**: 30-60 seconds per image
- **If slower**: Check if GPU is being used

### Image Quality Issues

**If image is blurry**:

- Increase `NUM_INFERENCE_STEPS` to 75-100
- Add "highly detailed, sharp, 8k" to prompt
- Use negative prompt: "blurry, low quality, pixelated"

**If image doesn't match prompt**:

- Increase `GUIDANCE_SCALE` to 5-7
- Make prompt more specific
- Try different strength values

**If colors are off**:

- Add color descriptions to prompt
- Adjust strength (lower for subtle changes)

## 📊 Performance Expectations

### RTX 4050 (8GB VRAM)

| Task           | Time      | Notes                |
| -------------- | --------- | -------------------- |
| First run      | 10-15 min | Downloading models   |
| Text-to-Image  | 30-60 sec | Per image            |
| Image-to-Image | 30-60 sec | Per image            |
| Model loading  | 30-60 sec | After first download |

## 🎓 Workflow Examples

### Workflow 1: Photo to Artwork

1. Run script: `python kandinsky_img2img.py`
2. Choose mode: `2` (Image-to-Image)
3. Prompt: `"oil painting style, impressionist, vibrant colors"`
4. Select your photo
5. Strength: `0.7`
6. Wait 30-60 seconds
7. Image opens automatically!

### Workflow 2: Create Variations

1. Generate base image (Text-to-Image)
2. Use that image as source (Image-to-Image)
3. Modify with different prompts
4. Create multiple variations

### Workflow 3: Iterative Refinement

1. Start with low strength (0.3)
2. Make small changes
3. Use output as new input
4. Repeat until perfect

## 🌟 Best Practices

1. **Start with Text-to-Image** to get base image
2. **Use Image-to-Image** for refinements
3. **Save your prompts** (automatically saved)
4. **Experiment with strength** values
5. **Be specific** in prompts
6. **Use negative prompts** to avoid unwanted elements
7. **Iterate**: Use outputs as new inputs

## 📁 Output Files

All generated images are saved in `outputs/` folder:

```
outputs/
├── kandinsky_txt2img_20231214_120000.png
├── kandinsky_txt2img_20231214_120000_prompt.txt
├── kandinsky_img2img_20231214_120500.png
└── kandinsky_img2img_20231214_120500_prompt.txt
```

## 🆚 Kandinsky vs Stable Diffusion

| Feature              | Kandinsky 2.2      | Stable Diffusion |
| -------------------- | ------------------ | ---------------- |
| Quality              | Excellent          | Very Good        |
| Prompt Understanding | Better             | Good             |
| Speed                | Slower             | Faster           |
| Model Size           | ~10GB              | ~4GB             |
| Artistic Style       | More artistic      | More realistic   |
| Best For             | Art, modifications | Photos, speed    |

## 🎯 When to Use What

**Use Kandinsky when**:

- You want artistic, stylized images
- You need better prompt understanding
- You're doing image-to-image modifications
- Quality > Speed

**Use Stable Diffusion when**:

- You want photorealistic images
- You need faster generation
- You have limited disk space
- Speed > Artistic quality

## 🚀 Next Steps

1. ✅ Generate your first image
2. 🎨 Try image-to-image modifications
3. 🔧 Experiment with different strengths
4. 📚 Try the example prompts above
5. 🌟 Create your own workflows

---

**Happy Creating!** 🎨
