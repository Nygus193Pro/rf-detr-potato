import cv2
import numpy as np
import time
from openvino.runtime import Core

# ===== ŚCIEŻKI =====
model_path = r"D:\Pycharm\rf_detr\openvino_ir_fp16\rf_detr.xml"
video_path = r"D:\Pycharm\rf_detr\demo.mp4"
out_path   = r"D:\Pycharm\rf_detr\demo_results_simple.avi"

# ===== PARAMETRY =====
prob_threshold = 0.90   # próg pewności dla "potato"
iou_threshold  = 0.40   # NMS
class_id       = 1      # 0 = background, 1 = potato
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-9)

def prepare_input(frame_bgr, in_w, in_h):
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = cv2.resize(img, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    img = (img - MEAN) / STD
    return img.transpose(2, 0, 1)[None].astype(np.float32)

def to_xyxy(cxcywh):
    cx, cy, w, h = cxcywh[:,0], cxcywh[:,1], cxcywh[:,2], cxcywh[:,3]
    x1 = cx - w/2; y1 = cy - h/2
    x2 = cx + w/2; y2 = cy + h/2
    return np.stack([x1, y1, x2, y2], axis=1)

def iou_one_to_many(box, boxes):
    x1 = np.maximum(box[0], boxes[:,0]); y1 = np.maximum(box[1], boxes[:,1])
    x2 = np.minimum(box[2], boxes[:,2]); y2 = np.minimum(box[3], boxes[:,3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    a1 = (box[2]-box[0]) * (box[3]-box[1])
    a2 = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
    return inter / (a1 + a2 - inter + 1e-9)

def nms(boxes, scores, iou_thr):
    if boxes.size == 0:
        return np.array([], dtype=np.int32)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ovr = iou_one_to_many(boxes[i], boxes[order[1:]])
        inds = np.where(ovr <= iou_thr)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int32)

# ===== OpenVINO =====
core = Core()
compiled = core.compile_model(core.read_model(model_path), "CPU")

# wejście (rozmiar)
n, c, in_h, in_w = compiled.input(0).shape

# wyjścia: wybierz po kształcie (prosto)
boxes_name  = None  # [1, N, 4]
logits_name = None  # [1, N, C>=2]
for out in compiled.outputs:
    shp = tuple(out.shape)
    if len(shp) == 3 and shp[-1] == 4:
        boxes_name = out.get_any_name()
    elif len(shp) == 3 and shp[-1] >= 2:
        logits_name = out.get_any_name()
if boxes_name is None:  boxes_name  = compiled.outputs[0].get_any_name()
if logits_name is None: logits_name = compiled.outputs[1].get_any_name()

# ===== Wideo =====
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("Nie mogę otworzyć wideo: " + video_path)

fps = cap.get(cv2.CAP_PROP_FPS) or 30
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (W, H))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

start_time = time.time()
frame_number = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_number += 1

    inp = prepare_input(frame, in_w, in_h)
    res = compiled([inp])

    boxes  = res[boxes_name][0]        # (N,4)  w [0..1], [cx,cy,w,h]
    logits = res[logits_name][0]       # (N,C)  logity: [bg, potato]
    probs = softmax(logits)
    scores = probs[:, class_id]

    idx = np.where(scores >= prob_threshold)[0]
    if idx.size > 0:
        xyxy = to_xyxy(boxes[idx])
        xyxy[:, [0,2]] *= W
        xyxy[:, [1,3]] *= H
        picked = nms(xyxy, scores[idx], iou_threshold)

        for (x1, y1, x2, y2), sc in zip(xyxy[picked].astype(int), scores[idx][picked]):
            x1 = max(0, min(W-1, x1)); y1 = max(0, min(H-1, y1))
            x2 = max(0, min(W-1, x2)); y2 = max(0, min(H-1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,255), 5)
            cv2.putText(frame, f"potato {sc:.2f}", (x1, max(0, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 5)

    writer.write(frame)

    if total_frames:
        print(f"Processed: {frame_number}/{total_frames}")
    else:
        print(f"Processed: {frame_number}")

cap.release()
writer.release()

elapsed = time.time() - start_time + 1e-9
fps_out = frame_number / elapsed
print(f"Done: {out_path} | Frames: {frame_number} | FPS ~ {fps_out:.2f}")
