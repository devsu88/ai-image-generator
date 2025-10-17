"""
Image generation utilities using Stable Diffusion.

This module provides the ImageGenerator class for generating images
using the Stable Diffusion model with configurable parameters.
"""

import logging
import os
from datetime import datetime
from typing import List, Optional, Tuple, Union

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageGenerator:
    """
    A class for generating images using Stable Diffusion.
    
    This class handles the initialization of the Stable Diffusion pipeline
    and provides methods for generating images with configurable parameters.
    """
    
    def __init__(self, model_id: str = "sd-legacy/stable-diffusion-v1-5"):
        """
        Initialize the ImageGenerator with a Stable Diffusion model.
        
        Args:
            model_id: The Hugging Face model ID for Stable Diffusion.
                     Defaults to "sd-legacy/stable-diffusion-v1-5".
        
        Raises:
            RuntimeError: If the model fails to load or initialize.
        """
        self.model_id = model_id
        self.pipeline = None
        self.device = self._get_device()
        
        try:
            logger.info(f"Loading Stable Diffusion model: {model_id}")
            logger.info(f"Using device: {self.device}")
            
            # Load the pipeline with appropriate device settings
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,  # Disable safety checker for better performance
                requires_safety_checker=False
            )
            
            # Move to device
            self.pipeline = self.pipeline.to(self.device)
            
            # Enable memory efficient attention if available
            if hasattr(self.pipeline, 'enable_attention_slicing'):
                self.pipeline.enable_attention_slicing()
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Failed to initialize Stable Diffusion model: {str(e)}")
    
    def _get_device(self) -> str:
        """
        Determine the best available device for inference.
        
        Returns:
            str: The device to use ('cuda' or 'cpu').
        """
        if torch.cuda.is_available():
            return "cuda"
        else:
            logger.warning("CUDA not available, using CPU. Generation will be slower.")
            return "cpu"
    
    def generate_images(
        self,
        prompt: str,
        num_images: int = 1,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None
    ) -> List[Image.Image]:
        """
        Generate images based on a text prompt.
        
        Args:
            prompt: The text prompt describing the desired image.
            num_images: Number of images to generate (1-4).
            num_inference_steps: Number of denoising steps (20-100).
            guidance_scale: How closely to follow the prompt (1-20).
            width: Width of the generated image (max 512).
            height: Height of the generated image (max 512).
            seed: Random seed for reproducible generation.
        
        Returns:
            List[Image.Image]: List of generated PIL Images.
        
        Raises:
            ValueError: If parameters are out of valid ranges.
            RuntimeError: If image generation fails.
        """
        # Validate parameters
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        if num_images < 1 or num_images > 4:
            raise ValueError("Number of images must be between 1 and 4")
        
        if num_inference_steps < 20 or num_inference_steps > 100:
            raise ValueError("Inference steps must be between 20 and 100")
        
        if guidance_scale < 1.0 or guidance_scale > 20.0:
            raise ValueError("Guidance scale must be between 1.0 and 20.0")
        
        if width > 512 or height > 512:
            raise ValueError("Image dimensions cannot exceed 512x512")
        
        if width < 64 or height < 64:
            raise ValueError("Image dimensions must be at least 64x64")
        
        try:
            logger.info(f"Generating {num_images} image(s) with prompt: '{prompt[:50]}...'")
            
            # Set seed for reproducibility
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
            
            # Generate images
            with torch.no_grad():
                result = self.pipeline(
                    prompt=prompt,
                    num_images_per_prompt=num_images,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height
                )
            
            images = result.images
            logger.info(f"Successfully generated {len(images)} image(s)")
            
            return images
            
        except Exception as e:
            logger.error(f"Image generation failed: {str(e)}")
            raise RuntimeError(f"Failed to generate images: {str(e)}")


def save_image_with_metadata(
    image: Image.Image,
    prompt: str,
    output_dir: str = "outputs",
    metadata: Optional[dict] = None
) -> str:
    """
    Save an image with metadata to the output directory.
    
    Args:
        image: The PIL Image to save.
        prompt: The prompt used to generate the image.
        output_dir: Directory to save the image in.
        metadata: Additional metadata to include in filename.
    
    Returns:
        str: The path to the saved image file.
    
    Raises:
        OSError: If the image cannot be saved.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_prompt = safe_prompt.replace(' ', '_')
    
    filename = f"generated_{timestamp}_{safe_prompt}.png"
    filepath = os.path.join(output_dir, filename)
    
    try:
        # Save the image
        image.save(filepath, "PNG")
        logger.info(f"Image saved to: {filepath}")
        
        # Save metadata if provided
        if metadata:
            metadata_file = filepath.replace('.png', '_metadata.txt')
            with open(metadata_file, 'w') as f:
                f.write(f"Prompt: {prompt}\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
        
        return filepath
        
    except Exception as e:
        logger.error(f"Failed to save image: {str(e)}")
        raise OSError(f"Failed to save image: {str(e)}")
