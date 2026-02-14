#!/usr/bin/env python3
"""Gradio UI for Qwen-Image-Edit-2511 image editing."""

import torch
import gradio as gr
from PIL import Image

MODEL_ID = "Qwen/Qwen-Image-Edit-2511"
pipeline = None


def load_pipeline():
    global pipeline
    if pipeline is not None:
        return
    from diffusers import QwenImageEditPlusPipeline

    print(f"Loading {MODEL_ID} ...")
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    pipeline.enable_model_cpu_offload()
    pipeline.set_progress_bar_config(disable=None)
    print("Model loaded and ready.")


def edit_image(
    image1,
    image2,
    image3,
    prompt,
    negative_prompt,
    num_images,
    steps,
    cfg_scale,
    seed,
):
    if not prompt:
        raise gr.Error("Please enter a prompt.")

    images = [img for img in (image1, image2, image3) if img is not None]
    if not images:
        raise gr.Error("Upload at least one image.")

    load_pipeline()

    input_imgs = images if len(images) > 1 else images[0]

    generator = torch.manual_seed(int(seed)) if int(seed) >= 0 else None

    with torch.inference_mode():
        result = pipeline(
            image=input_imgs,
            prompt=prompt,
            negative_prompt=negative_prompt or " ",
            num_inference_steps=int(steps),
            true_cfg_scale=float(cfg_scale),
            guidance_scale=1.0,
            generator=generator,
            num_images_per_prompt=int(num_images),
        )
    return result.images


# ── Gradio UI ────────────────────────────────────────────────

with gr.Blocks(title="Qwen Image Edit 2511", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "## Qwen Image Edit 2511\n"
        "Upload 1-3 images and describe the edit you want. "
        "The model will generate edited images based on your prompt."
    )

    with gr.Row():
        # ── Left column: inputs ──
        with gr.Column(scale=1):
            img1 = gr.Image(type="pil", label="Image 1 (required)")
            with gr.Row():
                img2 = gr.Image(type="pil", label="Image 2 (optional)")
                img3 = gr.Image(type="pil", label="Image 3 (optional)")

            prompt = gr.Textbox(
                label="Prompt",
                placeholder="e.g. Change the background to a beach sunset",
                lines=3,
            )
            neg_prompt = gr.Textbox(
                label="Negative prompt (optional)",
                placeholder="e.g. distortion, ugly, blurry",
                lines=1,
            )

            with gr.Row():
                num_images = gr.Slider(
                    minimum=1, maximum=4, value=1, step=1, label="Images to generate"
                )
                steps = gr.Slider(
                    minimum=10, maximum=80, value=40, step=1, label="Inference steps"
                )
            with gr.Row():
                cfg_scale = gr.Slider(
                    minimum=1.0, maximum=10.0, value=4.0, step=0.5, label="CFG scale"
                )
                seed = gr.Number(value=-1, label="Seed (-1 = random)")

            generate_btn = gr.Button("Generate", variant="primary", size="lg")

        # ── Right column: output ──
        with gr.Column(scale=1):
            gallery = gr.Gallery(
                label="Results",
                columns=1,
                rows=1,
                object_fit="contain",
                show_fullscreen_button=True,
            )

    generate_btn.click(
        fn=edit_image,
        inputs=[img1, img2, img3, prompt, neg_prompt, num_images, steps, cfg_scale, seed],
        outputs=gallery,
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
