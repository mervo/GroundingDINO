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

output_folder = Path('logs/1111')
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
sahi_general = SahiGeneral(model=model)

img = cv2.imread(str(imgpath))
bs = 1
imgs = [img for _ in range(bs)]

classes = ['hole']

torch.cuda.synchronize()
tic = perf_counter()

# detections, labels = model.predict_with_caption(
#     image=img,
#     caption='hole',
#     box_threshold=0.35,
#     text_threshold=0.25
# )

detections = sahi_general.detect(imgs, classes)

torch.cuda.synchronize()
dur = perf_counter() - tic

print(f'Time taken: {(dur * 1000):0.2f}ms')

draw_frame = img.copy()

for det in detections[0]:
    print(det)
    l = det['l']
    t = det['t']
    r = det['r']
    b = det['b']
    classname = det['label']
    cv2.rectangle(draw_frame, (l, t), (r, b), (255, 255, 0), 1)
    cv2.putText(draw_frame, classname, (l, t - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))

output_path = output_folder / 'test_out.jpg'
cv2.imwrite(str(output_path), draw_frame)
