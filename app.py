import os
import random

import gradio as gr
import numpy as np
import SimpleITK as sitk
import torch
from monai.transforms import Resize
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

model_path = "./models"
max_length = 1024
image_size = (128, 256, 256)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16

loaded_models = {}


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def get_available_models():
    model_dirs = [
        d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))
    ]
    if not model_dirs:
        model_dirs = ["default"]
    return model_dirs


def load_model(model_name):
    if model_name in loaded_models:
        return loaded_models[model_name]

    current_model_path = (
        os.path.join(model_path, model_name) if model_name != "default" else model_path
    )

    tokenizer = AutoProcessor.from_pretrained(
        current_model_path,
        max_length=max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        current_model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    ).to(device=device)

    proj_out_num = (
        model.get_model().config.proj_out_num
        if hasattr(model.get_model().config, "proj_out_num")
        else 256
    )

    loaded_models[model_name] = {
        "model": model,
        "tokenizer": tokenizer,
        "proj_out_num": proj_out_num,
    }
    return loaded_models[model_name]


def load_gt(path):
    with open(path, "r") as f:
        return f.read()


def normalize_and_convert(image_slice):
    norm_slice = (
        (image_slice - np.min(image_slice))
        / (np.max(image_slice) - np.min(image_slice))
        * 255
    )
    norm_slice = norm_slice.astype(np.uint8)
    return Image.fromarray(norm_slice).convert("L")


def process_upload(file, text_input, model_name, gt_text):
    if not file.name.endswith(".nii.gz"):
        return ("Invalid file format. Please upload a .nii.gz file",) + (None,) * 8

    img = sitk.ReadImage(file.name)
    img_array = sitk.GetArrayFromImage(img)

    axial_slices = [img_array[i, :, :] for i in range(img_array.shape[0])]
    coronal_slices = [img_array[:, i, :] for i in range(img_array.shape[1])]
    sagittal_slices = [img_array[:, :, i] for i in range(img_array.shape[2])]

    axial_images = [normalize_and_convert(s) for s in axial_slices]
    coronal_images = [normalize_and_convert(s) for s in coronal_slices]
    sagittal_images = [normalize_and_convert(s) for s in sagittal_slices]

    model_data = load_model(model_name)
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]
    proj_out_num = model_data["proj_out_num"]

    resize_transform = Resize(spatial_size=image_size, mode="bilinear")
    image_input = np.expand_dims(img_array.copy(), axis=0)
    image_input = resize_transform(image_input)
    image_input = image_input.data.unsqueeze(0).to(device=device, dtype=dtype)

    image_tokens = "<im_patch>" * proj_out_num
    input_txt = image_tokens + text_input
    input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"].to(device=device)

    with torch.no_grad():
        generation = model.generate(
            images=image_input,
            inputs=input_ids,
            max_new_tokens=max_length,
            do_sample=True,
            top_p=0.9,
            temperature=1.0,
        )
    result = tokenizer.decode(generation[0], skip_special_tokens=True)

    gt_visib = gr.update(visible=bool(gt_text), value=load_gt(gt_text))

    axial_idx = len(axial_images) // 2
    coronal_idx = len(coronal_images) // 2
    sagittal_idx = len(sagittal_images) // 2

    return (
        result,
        gt_visib,
        axial_images,
        coronal_images,
        sagittal_images,
        gr.Slider(
            minimum=0,
            maximum=len(axial_images) - 1,
            value=axial_idx,
            step=1,
            label="Axial Slice",
        ),
        gr.Slider(
            minimum=0,
            maximum=len(coronal_images) - 1,
            value=coronal_idx,
            step=1,
            label="Coronal Slice",
        ),
        gr.Slider(
            minimum=0,
            maximum=len(sagittal_images) - 1,
            value=sagittal_idx,
            step=1,
            label="Sagittal Slice",
        ),
        axial_images[axial_idx],
        coronal_images[coronal_idx],
        sagittal_images[sagittal_idx],
    )


def update_slice(slice_idx, slice_list):
    return slice_list[slice_idx]


available_models = get_available_models()

examples = [
    [
        "./data/demo/001480/Axial_bone_window.nii.gz",
        "What is the CT phase of the image?",
        available_models[-1],
        "./data/demo/001480/text.txt",
    ],
    [
        "./data/demo/007771/Coronal_bone_window.nii.gz",
        "What plane is the image in?",
        available_models[-1],
        "./data/demo/007771/text.txt",
    ],
    [
        "./data/demo/015533/Sagittal_non_contrast.nii.gz",
        "Where is the lesion located?",
        available_models[-1],
        "./data/demo/015533/text.txt",
    ],
    [
        "./data/demo/009422/Axial_C__delayed.nii.gz",
        "Which organ has a mass lesion?",
        available_models[-1],
        "./data/demo/009422/text.txt",
    ],
    [
        "./data/demo/024421/Coronal_C__portal_venous_phase.nii.gz",
        "Where is the pleural effusion located?",
        available_models[-1],
        "./data/demo/024421/text.txt",
    ],
    [
        "./data/demo/004794/Sagittal_C__portal_venous_phase.nii.gz",
        "Is pelvic ascites present?",
        available_models[-1],
        "./data/demo/004794/text.txt",
    ],
    [
        "./data/demo/015260/Axial_non_contrast.nii.gz",
        "Generate a medical report based on this image.",
        available_models[-1],
        "./data/demo/015260/text.txt",
    ],
]

with gr.Blocks() as demo:
    gr.Markdown("# Vision Language Model for 3D Medical Image Analysis")

    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload .nii.gz File")
            text_input = gr.Textbox(
                label="Text Input", placeholder="Enter your text here..."
            )
            model_dropdown = gr.Dropdown(
                choices=available_models,
                value=available_models[-1],
                label="Select Model",
            )
            gt_text = gr.Textbox(label="Ground Truth", visible=False)
            submit_btn = gr.Button("Process")

            gr.Examples(
                examples=[[ex[0], ex[1], ex[2], ex[3]] for ex in examples],
                inputs=[file_input, text_input, model_dropdown, gt_text],
                outputs=[],
                fn=lambda: None,
                cache_examples=False,
                label="Examples",
            )

        with gr.Column():
            model_output = gr.Textbox(label="Model Output")
            gt_output = gr.Textbox(label="Ground Truth", visible=False)

            with gr.Tab("Axial View"):
                axial_slider = gr.Slider(label="Axial Slice")
                axial_image = gr.Image(label="Axial Slice Viewer")

            with gr.Tab("Coronal View"):
                coronal_slider = gr.Slider(label="Coronal Slice")
                coronal_image = gr.Image(label="Coronal Slice Viewer")

            with gr.Tab("Sagittal View"):
                sagittal_slider = gr.Slider(label="Sagittal Slice")
                sagittal_image = gr.Image(label="Sagittal Slice Viewer")

    axial_state = gr.State()
    coronal_state = gr.State()
    sagittal_state = gr.State()

    submit_btn.click(
        fn=process_upload,
        inputs=[file_input, text_input, model_dropdown, gt_text],
        outputs=[
            model_output,
            gt_output,
            axial_state,
            coronal_state,
            sagittal_state,
            axial_slider,
            coronal_slider,
            sagittal_slider,
            axial_image,
            coronal_image,
            sagittal_image,
        ],
    )

    axial_slider.change(
        fn=update_slice, inputs=[axial_slider, axial_state], outputs=axial_image
    )

    coronal_slider.change(
        fn=update_slice, inputs=[coronal_slider, coronal_state], outputs=coronal_image
    )

    sagittal_slider.change(
        fn=update_slice,
        inputs=[sagittal_slider, sagittal_state],
        outputs=sagittal_image,
    )

if __name__ == "__main__":
    seed_everything(42)
    load_model(available_models[-1])

    ngrok_token = os.environ.get("NGROK_TOKEN")
    if ngrok_token:
        import pyngrok.ngrok as ngrok

        ngrok.set_auth_token(ngrok_token)
        public_url = ngrok.connect(7860).public_url
        print(f"Public URL: {public_url}")
        demo.queue(max_size=16).launch(
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True,
        )
    else:
        demo.queue().launch(
            share=True,
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True,
        )
