# AI Image Generator — powered by Stable Diffusion

A web application for generating high-quality images using Stable Diffusion with an intuitive Gradio interface. Generate stunning images from text prompts in seconds!

## 🚀 Features

- **Text-to-Image Generation**: Create images from natural language descriptions
- **Configurable Parameters**: Fine-tune generation with inference steps, guidance scale, and more
- **Multiple Images**: Generate 1-4 images per prompt for variety
- **Custom Dimensions**: Adjust image size up to 512x512 pixels
- **Reproducible Results**: Use seeds for consistent generation
- **Local Saving**: Automatically save generated images with metadata
- **GPU/CPU Support**: Automatic device detection with CPU fallback
- **Modern UI**: Clean, responsive full-width interface with real-time feedback
- **Example Integration**: One-click examples that auto-fill all parameters

## 🛠️ Technologies

- **Gradio**: Web interface framework
- **Stable Diffusion**: AI image generation model
- **PyTorch**: Deep learning backend
- **Diffusers**: Hugging Face diffusion models library
- **PIL**: Image processing and saving

## 📦 Installation & Local Setup

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended) or CPU
- 8GB+ RAM (16GB+ recommended for GPU)

### Step-by-Step Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd ai-image-generator
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:7860`

## 🎨 User Interface

The application features a modern, full-width layout with two main sections:

### Left Column - Input & Examples
- **Example Prompts**: Click any example to auto-fill all parameters
- **Text Prompt**: Enter your custom description
- **Generation Settings**: Configure inference steps, guidance scale, seed
- **Image Settings**: Set dimensions and number of images
- **Generate Button**: Start the image generation process

### Right Column - Output
- **Status Display**: Real-time feedback on generation progress
- **Image Gallery**: View generated images with download options
- **Auto-Save**: Images automatically saved to `outputs/` folder with metadata

## 🚀 Deploy on Hugging Face Spaces

### Method 1: Using Hugging Face Hub

1. **Create a new Space**
   - Go to [Hugging Face Spaces](https://huggingface.co/new-space)
   - Choose "Gradio" as the SDK
   - Set visibility (Public/Private)

2. **Upload your files**
   ```bash
   # Install huggingface_hub
   pip install huggingface_hub
   
   # Login to Hugging Face
   huggingface-cli login
   
   # Upload files to your space
   git clone https://huggingface.co/spaces/your-username/your-space-name
   cp -r ai-image-generator/* your-space-name/
   cd your-space-name
   git add .
   git commit -m "Add AI Image Generator"
   git push
   ```

### Method 2: Direct Upload

1. **Create Space on Hugging Face**
   - Visit [huggingface.co/new-space](https://huggingface.co/new-space)
   - Name: `ai-image-generator`
   - SDK: `Gradio`
   - Hardware: `CPU` (free) or `GPU` (paid)

2. **Upload files via web interface**
   - Upload all files from this repository
   - Ensure `app.py` is the main file

3. **Configure Space settings**
   - In Space settings, set:
     - **SDK**: Gradio
     - **Hardware**: CPU (free tier) or GPU (for faster generation)
     - **Visibility**: Public

## 💡 Example Prompts

The application includes comprehensive example prompts organized by categories:

### Built-in Examples
Try these prompts to get started:

- **Fantasy**: "A majestic dragon flying over a medieval castle, fantasy art, detailed, epic"
- **Cyberpunk**: "A cyberpunk cityscape at night, neon lights, futuristic, high-tech"
- **Nature**: "A serene lake with mountains in the background, peaceful landscape, photorealistic"
- **Animals**: "A cute robot playing with a cat, cartoon style, colorful, kawaii"
- **Portrait**: "A professional headshot of a businesswoman, corporate style, clean background"
- **Abstract**: "Colorful abstract art, geometric shapes, modern, vibrant"

### Example Files
- **`examples.json`**: Comprehensive collection with 8 categories (Nature & Landscapes, Fantasy & Magic, Cyberpunk & Sci-Fi, Animals & Pets, Portraits & People, Abstract & Artistic, Food & Still Life, Architecture & Urban)

### Enhanced Examples with Parameters
Each example now includes optimized parameters:
- **Inference Steps**: Tailored for each prompt type (45-80 steps)
- **Guidance Scale**: Optimized for prompt adherence (6.5-9.5)
- **Dimensions**: Standard 512x512 for consistency
- **Seed**: Null for random generation (can be customized)

### Using Examples in the Interface
1. **One-Click Examples**: Click on any example to automatically fill all form parameters
2. **Optimized Parameters**: Each example includes pre-configured settings for best results
3. **Custom Examples**: Modify the `examples.json` file to add your own prompt collections
4. **Parameter Learning**: See how different prompt types benefit from different parameter combinations

## ⚙️ Parameters Guide

- **Inference Steps** (20-100): More steps = better quality, slower generation
- **Guidance Scale** (1-20): Higher values = closer to prompt, may be more constrained
- **Seed**: Use same seed + prompt for identical results
- **Dimensions**: Up to 512x512 pixels (higher = more memory required)
- **Number of Images**: Generate 1-4 variations per prompt

## 🔧 Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce image dimensions (try 256x256)
   - Decrease number of inference steps
   - Use CPU instead of GPU

2. **Slow Generation**
   - Ensure you're using GPU if available
   - Reduce inference steps
   - Use smaller image dimensions

3. **Model Loading Issues**
   - Check internet connection
   - Verify Hugging Face access
   - Try restarting the application

### Performance Tips

- **GPU**: 10-30 seconds per image
- **CPU**: 2-5 minutes per image
- **Memory**: 4-8GB RAM recommended

## 📁 Project Structure

```
ai-image-generator/
├── app.py                    # Main Gradio application
├── utils/
│   ├── __init__.py          # Package marker
│   └── generation.py        # Image generation logic
├── outputs/                 # Generated images (auto-created)
├── examples.json            # Comprehensive example prompts by category
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Credits

- **Stable Diffusion Model**: [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- **Original Stable Diffusion**: [stabilityai/stable-diffusion](https://github.com/Stability-AI/stablediffusion)
- **Gradio Framework**: [gradio.app](https://gradio.app)
- **Hugging Face**: [huggingface.co](https://huggingface.co)

## 📞 Support

If you encounter any issues or have questions:

1. Check the [troubleshooting section](#-troubleshooting)
2. Open an issue on GitHub
3. Contact the maintainers

---

**Happy Image Generating! 🎨✨**