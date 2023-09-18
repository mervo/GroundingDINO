from importlib_resources import files
from pathlib import Path
from time import perf_counter

import cv2
import torch

from sahi_general.script.sahi_general import SahiGeneral
from groundingdino.util.inference import Model

imgpath = Path('/home/user/Downloads/runway_dmg_2.jpg')
if not imgpath.is_file():
    raise AssertionError(f'{str(imgpath)} not found')

output_folder = Path('../logs/1111')
output_folder.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = '../groundingdino/config/GroundingDINO_SwinB_cfg.py'
WEIGHTS_PATH = '../weights/groundingdino_swinb_cogcoor.pth'
'''
    SAHI library needs to be installed
    Model needs to have classname_to_idx function and get_detections_dict function
    classname_to_idx : int
        class index of the classname given
    get_detections_dict : List[dict]
        list of detections for each frame with keys: label, confidence, t, l, b, r, w, h
'''

model = Model(model_config_path=CONFIG_PATH, model_checkpoint_path=WEIGHTS_PATH)
"""
Args:
    model :
        any model
    sahi_image_height_threshold: int
        If image exceed this height, sahi will be performed on it.
        Defaulted to 900
    sahi_image_width_threshold: int
        If image exceed this width, sahi will be performed on it.
        Defaulted to 900
    sahi_slice_height: int
        Sliced image height.
        Defaulted to 512
    sahi_slice_width: int
        Sliced image width.
        Defaulted to 512
    sahi_overlap_height_ratio: float
        Fractional overlap in height of each window (e.g. an overlap of 0.2 for a window of size 512 yields an overlap of 102 pixels).
        Default to '0.2'.
    sahi_overlap_width_ratio: float
        Fractional overlap in width of each window (e.g. an overlap of 0.2 for a window of size 512 yields an overlap of 102 pixels).
        Default to '0.2'.
    sahi_postprocess_type: str
        Type of the postprocess to be used after sliced inference while merging/eliminating predictions.
        Options are 'NMM', 'GRREDYNMM' or 'NMS'. Default is 'GRREDYNMM'.
        Defaulted to "GREEDYNMM"
    sahi_postprocess_match_metric: str
        Metric to be used during object prediction matching after sliced prediction.
        'IOU' for intersection over union, 'IOS' for intersection over smaller area.
        Defaulted to "IOS"
    sahi_postprocess_match_threshold: float
        Sliced predictions having higher iou/ios than postprocess_match_threshold will be postprocessed after sliced prediction.
        Defaulted to 0.5
    sahi_postprocess_class_agnostic: bool
        If True, postprocess will ignore category ids.
        Defaulted to True
    full_frame_detection: bool
        If True, additional detection will be done on the full frame.
        Defaulted to True
"""
sahi_general = SahiGeneral(model=model,
                           sahi_image_height_threshold=900,
                           sahi_image_width_threshold=900,
                           sahi_slice_height=512,
                           sahi_slice_width=512,
                           sahi_overlap_height_ratio=0.2,
                           sahi_overlap_width_ratio=0.2,
                           sahi_postprocess_type="GREEDYNMM",
                           sahi_postprocess_match_metric="IOS",
                           sahi_postprocess_match_threshold=0.5,
                           sahi_postprocess_class_agnostic=True,
                           full_frame_detection=True)

img = cv2.imread(str(imgpath))
bs = 1
imgs = [img for _ in range(bs)]

classes = ['hole']

torch.cuda.synchronize()
tic = perf_counter()

# detections, labels = model.predict_with_caption(
#     image=img,
#     caption='hole',
#     box_threshold=0.24,
#     text_threshold=0.25
# )

detections = sahi_general.detect(imgs, classes)

torch.cuda.synchronize()
dur = perf_counter() - tic

print(f'Time taken: {(dur * 1000):0.2f}ms')

draw_frame = img.copy()

for det in detections[0]:
    l = det['l']
    t = det['t']
    r = det['r']
    b = det['b']
    classname = det['label']
    cv2.rectangle(draw_frame, (l, t), (r, b), (255, 255, 0), 1)
    cv2.putText(draw_frame, classname, (l, t - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))

output_path = output_folder / 'test_out.jpg'
print(output_path)
cv2.imwrite(str(output_path), draw_frame)
