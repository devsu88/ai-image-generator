"""
AI Image Generator - Gradio Web App

A web application for generating images using Stable Diffusion
with a user-friendly Gradio interface.
"""

import json
import logging
import os
from typing import List, Optional, Tuple

import gradio as gr
import torch
from PIL import Image

from utils.generation import ImageGenerator, save_image_with_metadata


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global image generator instance
image_generator: Optional[ImageGenerator] = None


def load_examples() -> dict:
    """Load example prompts from JSON file."""
    try:
        with open('examples.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("examples.json not found, using default examples")
        return {
            "examples": []
        }
    except Exception as e:
        logger.error(f"Failed to load examples: {str(e)}")
        return {
            "examples": [],
        }


def initialize_generator() -> ImageGenerator:
    """
    Initialize the image generator with error handling.
    
    Returns:
        ImageGenerator: The initialized generator instance.
    
    Raises:
        RuntimeError: If initialization fails.
    """
    try:
        logger.info("Initializing Stable Diffusion model...")
        generator = ImageGenerator()
        logger.info("Model initialized successfully")
        return generator
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        raise RuntimeError(f"Model initialization failed: {str(e)}")


def generate_images_interface(
    prompt: str,
    num_inference_steps: int,
    # guidance_scale: float,  # Commented out - using default value 0.0
    seed: Optional[int],
    width: int,
    height: int,
    num_images: int
) -> Tuple[List[str], str]:
    """
    Generate images based on user input parameters.
    
    Args:
        prompt: Text prompt for image generation.
        num_inference_steps: Number of denoising steps.
        # guidance_scale: How closely to follow the prompt.  # Commented out
        seed: Random seed for reproducibility.
        width: Image width.
        height: Image height.
        num_images: Number of images to generate.
    
    Returns:
        Tuple[List[str], str]: (list of image paths, status message)
    """
    global image_generator
    
    try:
        # Initialize generator if not already done
        if image_generator is None:
            image_generator = initialize_generator()
        
        # Validate inputs
        if not prompt or not prompt.strip():
            return [], "❌ Please enter a prompt"
        
        logger.info(f"Generating {num_images} image(s) with prompt: '{prompt[:50]}...'")
        
        # Generate images
        images = image_generator.generate_images(
            prompt=prompt,
            num_images=num_images,
            num_inference_steps=num_inference_steps,
            guidance_scale=0.0,  # Using default value since guidance_scale is removed from UI
            width=width,
            height=height,
            seed=seed
        )
        
        # Save images and collect paths
        saved_paths = []
        for i, image in enumerate(images):
            metadata = {
                "num_inference_steps": num_inference_steps,
                "guidance_scale": 0.0,  # Using default value
                "seed": seed,
                "width": width,
                "height": height,
                "image_number": i + 1
            }
            
            filepath = save_image_with_metadata(
                image=image,
                prompt=prompt,
                metadata=metadata
            )
            saved_paths.append(filepath)
        
        status_msg = f"✅ Successfully generated {len(images)} image(s)!"
        logger.info(f"Generation completed: {status_msg}")
        
        return saved_paths, status_msg
        
    except Exception as e:
        error_msg = f"❌ Generation failed: {str(e)}"
        logger.error(f"Generation error: {str(e)}")
        return [], error_msg


def create_interface() -> gr.Blocks:
    """
    Create the Gradio interface for the AI Image Generator.
    
    Returns:
        gr.Blocks: The configured Gradio interface.
    """
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        max-width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .param-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .examples-section {
        background: #f0f8ff;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 2px solid #e1f5fe;
    }
    .gradio-row {
        gap: 2rem !important;
        max-width: 100% !important;
    }
    .gradio-column {
        min-width: 0 !important;
        flex: 1 !important;
    }
    .gradio-blocks {
        max-width: 100% !important;
    }
    """
    
    with gr.Blocks(css=css, title="AI Image Generator") as interface:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🎨 AI Image Generator — powered by Stable Diffusion</h1>
            <p style="font-size: 1.1em; color: #666;">Enter a prompt to generate an image in seconds</p>
        </div>
        """)
        
        # Load examples data first
        examples_data = load_examples()
        
        # Create examples for different categories
        all_examples = []

        # Add examples from categories
        if "examples" in examples_data:
            for category in examples_data["examples"]:
                if "prompts" in category:
                    for prompt_data in category["prompts"][:2]:  # Take first 2 from each category
                        if isinstance(prompt_data, dict):
                            all_examples.append([
                                prompt_data["text"],
                                prompt_data["num_inference_steps"],
                                # prompt_data["guidance_scale"],  # Commented out - using default value
                                prompt_data["seed"],
                                prompt_data["width"],
                                prompt_data["height"],
                                1  # num_images default
                            ])
                        else:
                            all_examples.append([prompt_data, 2, None, 1024, 1024, 1])  # Updated to match new parameters
        
        # Limit to 8 examples for better UI
        all_examples = all_examples[:8]
        
        with gr.Row():
            # Left column - Input parameters
            with gr.Column(scale=2):
                gr.Markdown("### 📝 Input Parameters")
                
                # Main prompt input
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="A beautiful sunset over mountains, digital art",
                    lines=3,
                    max_lines=5
                )
                
                # Generation parameters
                with gr.Group():
                    gr.Markdown("#### ⚙️ Generation Settings")
                    
                    num_inference_steps = gr.Slider(
                        minimum=1,
                        maximum=4,
                        value=2,
                        step=1,
                        label="Inference Steps",
                        info="More steps = better quality, slower generation"
                    )
                    
                    # guidance_scale = gr.Slider(
                    #     minimum=1.0,
                    #     maximum=20.0,
                    #     value=7.5,
                    #     step=0.1,
                    #     label="Guidance Scale",
                    #     info="How closely to follow the prompt"
                    # )
                    
                    seed = gr.Number(
                        label="Seed (optional)",
                        value=None,
                        precision=0,
                        info="Leave empty for random generation"
                    )
                
                # Image parameters
                with gr.Group():
                    gr.Markdown("#### 🖼️ Image Settings")
                    
                    with gr.Row():
                        width = gr.Number(
                            label="Width",
                            value=1024,
                            minimum=512,
                            maximum=1024,
                            step=64,
                            precision=0
                        )
                        
                        height = gr.Number(
                            label="Height", 
                            value=1024,
                            minimum=512,
                            maximum=1024,
                            step=64,
                            precision=0
                        )
                    
                    num_images = gr.Slider(
                        minimum=1,
                        maximum=4,
                        value=1,
                        step=1,
                        label="Number of Images",
                        info="Generate multiple variations"
                    )
                
                # Generate button
                generate_btn = gr.Button(
                    "🎨 Generate Images",
                    variant="primary",
                    size="lg"
                )
            
            # Right column - Output
            with gr.Column(scale=2):
                gr.Markdown("### 🖼️ Generated Images")
                
                # Status message
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                    value="Ready to generate images!"
                )
                
                # Image gallery
                gallery = gr.Gallery(
                    label="Generated Images",
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                    rows=2,
                    height="auto",
                    object_fit="contain"
                )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 2rem; padding: 1rem; border-top: 1px solid #eee;">
            <p>
                Powered by <a href="https://huggingface.co/docs/diffusers/using-diffusers/sdxl_turbo" target="_blank">Stable Diffusion XL Turbo</a> | 
                Built with <a href="https://gradio.app" target="_blank">Gradio</a>
            </p>
        </div>
        """)
        
        # Examples section - now that all components are defined
        with gr.Group():
            gr.Markdown("### 💡 Example Prompts - Click to Auto-Fill Form")
            gr.Examples(
                examples=all_examples,
                inputs=[
                    prompt_input,
                    num_inference_steps,
                    # guidance_scale,  # Commented out - removed from UI
                    seed,
                    width,
                    height,
                    num_images
                ],
                label="",
                elem_id="examples-section"
            )
        
        # Event handlers
        generate_btn.click(
            fn=generate_images_interface,
            inputs=[
                prompt_input,
                num_inference_steps,
                # guidance_scale,  # Commented out - removed from UI
                seed,
                width,
                height,
                num_images
            ],
            outputs=[gallery, status_output]
        )
        
    
    return interface


def main():
    """Main function to launch the Gradio interface."""
    try:
        logger.info("Starting AI Image Generator...")
        
        # Create and launch interface
        interface = create_interface()
        
        # Launch with appropriate settings
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            quiet=False
        )
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise


if __name__ == "__main__":
    main()
