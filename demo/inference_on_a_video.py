'''
CUDA_VISIBLE_DEVICES=0 python demo/inference_on_a_video.py \
-c groundingdino/config/GroundingDINO_SwinB_cfg.py \
-p weights/groundingdino_swinb_cogcoor.pth \
-i "/data/datasets/vision_60/20230523_DIC-20230717T023351Z-001/20230523_DIC/MVI_2682.MP4" \
-o logs/ \
-t "spot robot" \
--box_threshold 0.35 \
--inference_fps 10 \
--visualize True
'''
import argparse
import itertools
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span

import time
import cv2

get_your_config_from_env_var = os.environ.get('CONFIG_NAME', 'default_value_if_not_set')

# comma separated strings
video_feed_names = os.environ.get('VIDEO_FEED_NAMES',
                                  'VIDEO1')
streams = os.environ.get('STREAMS', 'rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4')
manual_video_fps = os.environ.get('MANUAL_VIDEO_FPS', '-1')  # -1 to try to read from video stream metadata
source_types = os.environ.get('SOURCE_TYPES', 'rtsp')

# TODO set queue size to 2 for live video streams to skip frames, None to process every frame in video
# queue_size = None
queue_size = int(os.environ.get('QUEUE_SIZE', 2))
recording_dir = os.environ.get('RECORDING_DIR', None)
reconnect_threshold_sec = int(os.environ.get('RECONNECT_THRESHOLD_SEC', 5))
max_height = int(os.environ.get('MAX_HEIGHT', 1080))
method = os.environ.get('METHOD', 'cv2')


def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        # color = tuple(np.random.randint(0, 255, size=3).tolist())
        color = (255, 0, 0)
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask


def load_image(image_pil):
    # load image
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False,
                         token_spans=None):
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append('target' + f"({str(logit.max().item())[:4]})")
                # TODO change to show actual phrase used in label display
                # pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
    else:
        # given-phrase mode
        positive_maps = create_positive_map_from_span(
            model.tokenizer(text_prompt),
            token_span=token_spans
        ).to(image.device)  # n_phrase, 256

        logits_for_phrases = positive_maps @ logits.T  # n_phrase, nq
        all_logits = []
        all_phrases = []
        all_boxes = []
        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            # get phrase
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
            # get mask
            filt_mask = logit_phr > box_threshold
            # filt box
            all_boxes.append(boxes[filt_mask])
            # filt logits
            all_logits.append(logit_phr[filt_mask])
            if with_logits:
                logit_phr_num = logit_phr[filt_mask]
                all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
            else:
                all_phrases.extend([phrase for _ in range(len(filt_mask))])
        boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        pred_phrases = all_phrases

    return boxes_filt, pred_phrases


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, default="../groundingdino/config/GroundingDINO_SwinT_OGC.py",
                        required=False, help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, default="../weights/groundingdino_swint_ogc.pth", required=False,
        help="path to checkpoint file"
    )
    parser.add_argument("--target_video", "-i", type=str, required=False, help="path to video file")
    parser.add_argument("--text_prompt", "-t", type=str, default="vision 60", required=False, help="text prompt")
    parser.add_argument('--inference_folder', "-o", type=Path, help='path to save inference results')
    parser.add_argument('--inference_fps', type=float, default=5.0, help='fps of inference')
    parser.add_argument('--visualize', type=bool, default=False, help='enable for visualization')

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--token_spans", type=str, default=None, help=
    "The positions of start and end positions of phrases of interest. \
    For example, a caption is 'a cat and a dog', \
    if you would like to detect 'cat', the token_spans should be '[[[2, 5]], ]', since 'a cat and a dog'[2:5] is 'cat'. \
    if you would like to detect 'a cat', the token_spans should be '[[[0, 1], [2, 5]], ]', since 'a cat and a dog'[0:1] is 'a', and 'a cat and a dog'[2:5] is 'cat'. \
    ")

    parser.add_argument("--cpu-only", action="store_true", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    video_path = args.target_video
    text_prompt = args.text_prompt
    output_dir = args.inference_folder
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    token_spans = args.token_spans
    inference_fps = args.inference_fps

    # load model
    model = load_model(config_file, checkpoint_path, cpu_only=args.cpu_only)

    # set the text_threshold to None if token_spans is set.
    if token_spans is not None:
        text_threshold = None
        print("Using token_spans. Set the text_threshold to None.")

    if args.visualize:
        cv2.namedWindow('OUTPUT', cv2.WINDOW_NORMAL)

    # process video frames
    from video_utils.video_manager import VideoManager

    if args.target_video is not None:
        streams = str(args.target_video)
    print(f'Current Video: {streams}')

    vidManager = VideoManager(video_feed_names=video_feed_names.split(','),
                              streams=streams.split(','), source_types=source_types.split(','),
                              manual_video_fps=manual_video_fps.split(','), queue_size=queue_size,
                              recording_dir=recording_dir,
                              reconnect_threshold_sec=reconnect_threshold_sec, max_height=max_height, method=method)
    vidManager.start()
    videos_information = vidManager.get_all_videos_information()
    print(f'{videos_information}')

    # make dir
    if args.inference_folder is not None:
        now = datetime.now()
        day = now.strftime("%Y_%m_%d_%H-%M-%S-%f")
        out_vid_fp = os.path.join(
            output_dir, 'output_{}.avi'.format(day))
        out = cv2.VideoWriter(out_vid_fp, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), inference_fps,
                              (int(videos_information[0]['width']), int(videos_information[0]['height'])))
        print((videos_information[0]['height']))

    start_time = time.time()
    total_frame_count = 0
    for frame_count in itertools.count():
        frame_of_each_video_feed = vidManager.read()  # frames is list of arrays from 0 - 255, dtype uint8
        total_frame_count += 1
        for i, video_stream_information in enumerate(vidManager.videos):
            if len(frame_of_each_video_feed[i]) != 0:
                pil_img = Image.fromarray(cv2.cvtColor(frame_of_each_video_feed[i], cv2.COLOR_BGR2RGB))
                # pil_img.show()
                image_pil, image = load_image(pil_img)
                # run model
                boxes_filt, pred_phrases = get_grounding_output(
                    model, image, text_prompt, box_threshold, text_threshold, cpu_only=args.cpu_only,
                    token_spans=eval(f"{token_spans}")
                )

                # visualize pred
                size = image_pil.size
                pred_dict = {
                    "boxes": boxes_filt,
                    "size": [size[1], size[0]],  # H,W
                    "labels": pred_phrases,
                }
                # import ipdb; ipdb.set_trace()
                current_img = plot_boxes_to_image(image_pil, pred_dict)[0]
                cv2_img = cv2.cvtColor(np.array(current_img), cv2.COLOR_BGR2RGB)

                if args.visualize:
                    cv2.imshow('OUTPUT', cv2_img)

                if args.inference_folder is not None:
                    out.write(cv2_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            if args.inference_folder is not None:
                out.release()
            break

    time_taken_for_single_frame = (time.time() - start_time) / total_frame_count
    print("End to End Average FPS: ", 1.0 / time_taken_for_single_frame)  # FPS = 1 / time to process 1 frame

    vidManager.stop()
    cv2.destroyAllWindows()
