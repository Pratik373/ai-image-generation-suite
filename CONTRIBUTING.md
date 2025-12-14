# Contributing to AI Image Generation Suite

Thank you for your interest in contributing! 🎉

## How to Contribute

### Reporting Bugs 🐛

If you find a bug, please open an issue with:

- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (GPU, OS, Python version)
- Error messages (if any)

### Suggesting Features 💡

We welcome feature suggestions! Please:

- Check if the feature already exists
- Describe the use case
- Explain why it would be useful
- Provide examples if possible

### Submitting Pull Requests 🔧

1. **Fork the repository**
2. **Create a branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes**
4. **Test thoroughly**
5. **Commit**: `git commit -m "Add: description of changes"`
6. **Push**: `git push origin feature/your-feature-name`
7. **Open a Pull Request**

### Code Style

- Follow PEP 8 guidelines
- Add comments for complex logic
- Update documentation if needed
- Keep functions focused and modular

### Testing

Before submitting:

- Test on your local machine
- Verify CUDA and CPU modes work
- Check for memory leaks
- Ensure error handling works

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/ai-image-generation-suite.git
cd ai-image-generation-suite

# Install in development mode
pip install -e .

# Run tests
python generate_image.py
python kandinsky_img2img.py
```

## Areas for Contribution

- 🆕 **New Models**: Add support for more diffusion models
- ⚡ **Optimizations**: Improve speed and memory usage
- 📚 **Documentation**: Improve guides and examples
- 🎨 **UI**: Create a web interface or GUI
- 🔧 **Features**: Batch processing, video generation, etc.

## Questions?

Feel free to open an issue for any questions!

Thank you for contributing! 🙏
