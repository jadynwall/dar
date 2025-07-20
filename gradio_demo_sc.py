import os
import sys
import torch
import numpy as np
import gradio as gr
from PIL import Image
import cv2
import tempfile
from omegaconf import OmegaConf
from diffusers.image_processor import VaeImageProcessor

# Add current directory to path
sys.path.append('.')

import torch
torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=False)  

# Import required modules
from src.featglac.feat_guidance import FeatureGuidance
from src.featglac.get_depth import get_depth_map

# Set up OpenCV
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# Initialize model and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config_path = "./src/featglac/config/default_sc.yaml"

# Load configuration and model
diff_handles_config = OmegaConf.load(config_path) if config_path is not None else None
diff_handles = FeatureGuidance(conf=diff_handles_config)
diff_handles.to(device)

def colorise_depth(depth_map):
    """Convert depth map to colored visualization"""
    max_depth = np.max(depth_map)
    min_depth = np.min(depth_map)
    depth_map = ((depth_map - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
    return depth_map

def process_scene_composition(background_image, foreground_image, mask_image, 
                            background_prompt, foreground_prompt, composition_prompt,
                            num_samples=1, seed=-1):
    """
    Process scene composition with the given inputs
    
    Args:
        background_image: PIL Image of background
        foreground_image: PIL Image of foreground object
        mask_image: PIL Image of mask (white for foreground, black for background)
        background_prompt: Text prompt describing background
        foreground_prompt: Text prompt describing foreground object
        composition_prompt: Text prompt for final composition
        num_samples: Number of samples to generate
    
    Returns:
        List of generated images
    """
    
    # Input validation
    if background_image is None or foreground_image is None or mask_image is None:
        raise gr.Error("Please upload all required images: background, foreground, and mask.")
    
    if not background_prompt or not foreground_prompt or not composition_prompt:
        raise gr.Error("Please provide all text prompts.")
    
    # Set random seed if provided
    if seed != -1:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Resize all images to 512x512
    background_image = background_image.convert("RGB").resize((512, 512))
    foreground_image = foreground_image.convert("RGB").resize((512, 512))
    mask_image = mask_image.convert("RGB").resize((512, 512))
    
    # Convert mask to binary (0 or 1)
    mask = np.array(mask_image) // 255
    
    # Generate depth maps
    depth_foreground = Image.fromarray(get_depth_map(foreground_image))
    depth_background = Image.fromarray(get_depth_map(background_image))
    
    # Convert to numpy arrays
    depth_foreground = np.array(depth_foreground)
    depth_background = np.array(depth_background)
    
    # Create combined depth map
    depth_control = depth_foreground * mask[:,:,0] + depth_background * (1 - mask[:,:,0])
    depth_control = torch.from_numpy(depth_control).unsqueeze(0).unsqueeze(0).to(torch.float32).to(device)
    
    # Prepare tensors for processing
    depth_background_tensor = torch.tensor(depth_background).unsqueeze(0).unsqueeze(0).to(device)
    depth_foreground_tensor = torch.tensor(depth_foreground).unsqueeze(0).unsqueeze(0).to(device)
    
    background_tensor = torch.tensor(np.array(background_image)).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    foreground_tensor = torch.tensor(np.array(foreground_image)).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    
    # Create temporary directory for null embeddings
    # null_text_emb_path = "./temp_null_embed"
    null_text_emb_path = "./examples/Gradio/null_embed"
    os.makedirs(null_text_emb_path, exist_ok=True)
    
    try:
    
        # Process background
        bg_name = "background"
        if os.path.exists(f"{null_text_emb_path}/{bg_name}_{background_prompt}_null_text.pt"):
            null_text_emb = torch.load(f"{null_text_emb_path}/{bg_name}_{background_prompt}_null_text.pt").to(device)
            init_noise = torch.load(f"{null_text_emb_path}/{bg_name}_{background_prompt}_init_noise.pt").to(device)
        else:
            null_text_emb, init_noise = diff_handles.invert_input_image(background_tensor, depth_background_tensor, background_prompt)
            init_noise = init_noise[-1]
            torch.save(null_text_emb.detach().cpu(), f"{null_text_emb_path}/{bg_name}_{background_prompt}_null_text.pt")
            torch.save(init_noise.detach().cpu(), f"{null_text_emb_path}/{bg_name}_{background_prompt}_init_noise.pt")
        
        null_text_emb_bg, init_noise_bg, activations_back, _ = diff_handles.generate_input_image(
            depth=depth_background_tensor, prompt=background_prompt, null_text_emb=null_text_emb, init_noise=init_noise)
        
        # Process foreground
        fg_name = "foreground"
        if os.path.exists(f"{null_text_emb_path}/{fg_name}_{foreground_prompt}_null_text.pt"):
            null_text_emb = torch.load(f"{null_text_emb_path}/{fg_name}_{foreground_prompt}_null_text.pt").to(device)
            init_noise = torch.load(f"{null_text_emb_path}/{fg_name}_{foreground_prompt}_init_noise.pt").to(device)
        else:
            null_text_emb, init_noise = diff_handles.invert_input_image(foreground_tensor, depth_foreground_tensor, foreground_prompt)
            init_noise = init_noise[-1]
            torch.save(null_text_emb.detach().cpu(), f"{null_text_emb_path}/{fg_name}_{foreground_prompt}_null_text.pt")
            torch.save(init_noise.detach().cpu(), f"{null_text_emb_path}/{fg_name}_{foreground_prompt}_init_noise.pt")
        
        null_text_emb_fg, init_noise_fg, activations_fore, _ = diff_handles.generate_input_image(
            depth=depth_foreground_tensor, prompt=foreground_prompt, null_text_emb=null_text_emb, init_noise=init_noise)
        
        # Prepare masks and activations for scene composition
        mpi_masks = [1 - mask, mask]
        activations = [activations_back, activations_fore]
        
        generated_images = []
        
        # Generate samples
        for i in range(num_samples):
            results = diff_handles.mpi_scene_comp(
                depth=depth_control, 
                prompt=composition_prompt,
                mpi_masks=mpi_masks,
                null_text_emb=None, 
                init_noise=init_noise_bg,
                activations=activations,
                use_input_depth_normalization=False
            )
            
            if diff_handles.conf.guided_diffuser.save_denoising_steps:
                edited_img, edited_disparity, denoising_steps = results
            else:
                edited_img, edited_disparity = results
            
            # Convert to PIL Image
            edited_img = edited_img.permute(0, 2, 3, 1).squeeze().cpu().numpy()
            edited_img = (edited_img * 255).astype(np.uint8)
            edited_img_pil = Image.fromarray(edited_img)
            
            generated_images.append(edited_img_pil)
        
        return generated_images
    
    except Exception as e:
        raise gr.Error(f"Error during scene composition: {str(e)}")

# Create Gradio interface
def create_demo():
    with gr.Blocks(title="Scene Composition Demo", css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .output-gallery {
            max-height: 600px;
        }
    """) as demo:
        gr.Markdown("# Scene Composition")
        gr.Markdown("Upload a background image, foreground image, and mask, then provide text prompts to generate composed scenes.")
        gr.Markdown("**Instructions:** Upload three images - background, foreground object, and a mask (white for foreground, black for background). Then provide text prompts describing each component and the desired final composition.")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Input Images")
                background_image = gr.Image(
                    label="Background Image", 
                    type="pil",
                    height=256
                )
                foreground_image = gr.Image(
                    label="Foreground Image", 
                    type="pil",
                    height=256
                )
                mask_image = gr.Image(
                    label="Mask Image (White = Foreground, Black = Background)", 
                    type="pil",
                    height=256
                )
            
            with gr.Column():
                gr.Markdown("### Text Prompts")
                background_prompt = gr.Textbox(
                    label="Background Prompt",
                    placeholder="e.g., a photo of a bright room with a table",
                    value="a photo of a bright room with a table"
                )
                foreground_prompt = gr.Textbox(
                    label="Foreground Prompt", 
                    placeholder="e.g., a photo of a bean bag",
                    value="a photo of a bean bag"
                )
                composition_prompt = gr.Textbox(
                    label="Composition Prompt",
                    placeholder="e.g., a photo of a bean bag behind a table in a bright room",
                    value="a photo of a bean bag behind a table in a bright room"
                )
                
                with gr.Accordion("Advanced Options", open=False):
                    num_samples = gr.Slider(
                        label="Number of Samples",
                        minimum=1,
                        maximum=4,
                        value=1,
                        step=1
                    )
                    seed = gr.Slider(
                        label="Random Seed",
                        minimum=-1,
                        maximum=999999999,
                        value=-1,
                        step=1,
                        info="Set to -1 for random seed"
                    )
                
                generate_button = gr.Button("Generate Composition", variant="primary")
                clear_button = gr.Button("Clear All", variant="secondary")
        
        with gr.Row():
            gr.Markdown("### Generated Results")
            with gr.Column():
                output_gallery = gr.Gallery(
                    label="Generated Images",
                    show_label=True,
                    elem_id="gallery",
                    columns=1,
                    height=512,
                    width=512,
                )
                status_text = gr.Textbox(
                    label="Status",
                    value="Ready to generate",
                    interactive=False
                )
        
        # Example inputs
        gr.Markdown("### Example Prompts")
        gr.Examples(
            examples=[
                [
                    "a photo of a bright room with a table",
                    "a photo of a bean bag", 
                    "a photo of a bean bag behind a table in a bright room"
                ],
                [
                    "a photo of a beach view",
                    "a photo of a table with cup and croissant",
                    "a photo of a table with cup and croissant with a beach view in the background"
                ],
                [
                    "a photo of a kitchen",
                    "a photo of a cat",
                    "a photo of a cat sitting on a kitchen counter"
                ]
            ],
            inputs=[background_prompt, foreground_prompt, composition_prompt]
        )
        
        # Connect the buttons
        def process_with_status(*args):
            try:
                status_text.update("Processing... Please wait.")
                result = process_scene_composition(*args)
                status_text.update("Generation completed successfully!")
                return result
            except Exception as e:
                status_text.update(f"Error: {str(e)}")
                raise e
        
        generate_button.click(
            fn=process_with_status,
            inputs=[
                background_image,
                foreground_image, 
                mask_image,
                background_prompt,
                foreground_prompt,
                composition_prompt,
                num_samples,
                seed
            ],
            outputs=output_gallery
        )
        
        # Clear function
        def clear_all():
            return None, None, None, "", "", "", 1, -1
        
        clear_button.click(
            fn=clear_all,
            outputs=[
                background_image,
                foreground_image,
                mask_image,
                background_prompt,
                foreground_prompt,
                composition_prompt,
                num_samples,
                seed
            ]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",  # Change from 0.0.0.0 to your server IP
        server_port=7860,
        share=False,
        debug=True,
        show_error=True,
        inbrowser=True,
        prevent_thread_lock=True
    ) 