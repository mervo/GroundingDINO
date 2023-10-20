import argparse
from functools import partial
import cv2
import requests
import os
from io import BytesIO
from PIL import Image
import numpy as np
from pathlib import Path


import warnings

import torch

# prepare the environment
os.system("python setup.py build develop --user")
os.system("pip install packaging==21.3")
os.system("pip install gradio")


warnings.filterwarnings("ignore")

import gradio as gr

from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import annotate, load_image, predict
import groundingdino.datasets.transforms as T

from huggingface_hub import hf_hub_download

from sahi_general.script.sahi_general import SahiGeneral
from groundingdino.util.inference import Model



# Use this command for evaluate the Grounding DINO model
config_file = "groundingdino/config/GroundingDINO_SwinB_cfg.py" # For faster inference, can use SwinT backbone config : GroundingDINO_SwinT_OGC.py
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "weights/groundingdino_swinb_cogcoor.pth"  # For faster inference, can use SwinT backbone weights: groundingdino_swint_ogc.pth

def load_model_hf(model_config_path, repo_id, filename, device='cpu'):
    args = SLConfig.fromfile(model_config_path) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model    

def image_transform_grounding(init_image):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image, _ = transform(init_image, None) # 3, h, w
    return init_image, image

def image_transform_grounding_for_vis(init_image):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
    ])
    image, _ = transform(init_image, None) # 3, h, w
    return image

#model = load_model_hf(config_file, ckpt_repo_id, ckpt_filenmae)

def run_grounding(input_image, grounding_caption, box_threshold, text_threshold,
                sahi_image_threshold, sahi_slice_dim ):
    model = Model(model_config_path=config_file, model_checkpoint_path=ckpt_filenmae, box_threshold=box_threshold)
    sahi_general = SahiGeneral(model=model,
                           sahi_image_height_threshold=sahi_image_threshold,
                           sahi_image_width_threshold=sahi_image_threshold,
                           sahi_slice_height=sahi_slice_dim,
                           sahi_slice_width=sahi_slice_dim,
                           sahi_overlap_height_ratio=0.2,
                           sahi_overlap_width_ratio=0.2,
                           sahi_postprocess_type="GREEDYNMM",
                           sahi_postprocess_match_metric="IOS",
                           sahi_postprocess_match_threshold=0.5,
                           sahi_postprocess_class_agnostic=True,
                           full_frame_detection=False)

    init_image = input_image.convert("RGB")
    original_size = init_image.size

    imgArray = np.array(init_image)
    bs = 1
    imgs = [imgArray for _ in range(bs)]

    detections = sahi_general.detect(imgs, grounding_caption)


    #_, image_tensor = image_transform_grounding(init_image)
    #image_pil: Image = image_transform_grounding_for_vis(init_image)

    # run grounidng
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model = model.to(device)

    draw_frame = imgArray.copy()

    for det in detections[0]:
        l = det['l']
        t = det['t']
        r = det['r']
        b = det['b']
        classname = det['label']
        confidence = det['confidence']
        cv2.rectangle(draw_frame, (l, t), (r, b), (0, 0, 255), 2)
        cv2.putText(draw_frame, f'{grounding_caption} ({confidence:.2f})', (l, t - 8), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    thickness=3, color=(0, 0, 255))

    #boxes, logits, phrases = predict(model, image_tensor, grounding_caption, box_threshold, text_threshold, device=device)
    #annotated_frame = annotate(image_source=np.asarray(image_pil), boxes=boxes, logits=logits, phrases=phrases)
    image_with_box = Image.fromarray(draw_frame)


    return image_with_box

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounding DINO demo", add_help=True)
    parser.add_argument("--debug", action="store_true", help="using debug mode")
    parser.add_argument("--share", action="store_true", help="share the app")
    args = parser.parse_args()

    block = gr.Blocks().queue()
    with block:
        gr.Markdown("# Open-Set Zero Shot Detection")
        gr.Markdown("### Open-World Detection with Text Prompts")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(source='upload', type="pil")
                grounding_caption = gr.Textbox(label="Detection Prompt")
                run_button = gr.Button(label="Run")
                with gr.Accordion("Advanced options", open=False):
                    box_threshold = gr.Slider(
                        label="Box Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.001
                    )
                    text_threshold = gr.Slider(
                        label="Text Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.001
                    )
                    sahi_image_threshold = gr.Slider(
                        label="SAHI Image Threshold", minimum=0, maximum=400, value=200, step=10
                    )
                    sahi_slice_dim = gr.Slider(
                        label="SAHI Slice Dimensions", minimum=0, maximum=600, value=300, step=10
                    )

            with gr.Column():
                gallery = gr.outputs.Image(
                    type="pil",
                    # label="grounding results"
                ).style(full_width=True, full_height=True)
                # gallery = gr.Gallery(label="Generated images", show_label=False).style(
                #         grid=[1], height="auto", container=True, full_width=True, full_height=True)

        run_button.click(fn=run_grounding, inputs=[
                        input_image, grounding_caption, box_threshold, text_threshold,
                        sahi_image_threshold, sahi_slice_dim ], outputs=[gallery])


    block.launch(server_name='0.0.0.0', server_port=7579, debug=args.debug, share=args.share)

