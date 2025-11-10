import openvino as ov
import os

onnx_path = r"D:\Pycharm\rf_detr\output\inference_model.onnx"
out_dir = r"D:\Pycharm\rf_detr\openvino_ir_fp16"

os.makedirs(out_dir, exist_ok = True)

ov_model = ov.convert_model(onnx_path)

ov.save_model(ov_model, os.path.join(out_dir, "rf_detr.xml"))

print("Zapisano IR do:", out_dir)