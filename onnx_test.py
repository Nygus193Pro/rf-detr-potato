import onnxruntime as ort, numpy as np, cv2
sess = ort.InferenceSession(r"D:\Pycharm\rf_detr\output\inference_model.onnx", providers=["CPUExecutionProvider"])
img  = cv2.cvtColor(cv2.resize(cv2.imread(r"D:\Pycharm\Projekt3KMK\klatki2_640x640\Klatka_2311.jpg"), (560,560)), cv2.COLOR_BGR2RGB)
X    = np.expand_dims(np.transpose(img.astype(np.float32)/255.0, (2,0,1)), 0)
outs = sess.run(None, {sess.get_inputs()[0].name: X})
print([o.shape for o in outs]) 


onnx_path = r"D:\Pycharm\rf_detr\output\inference_model.onnx"
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])


inp = sess.get_inputs()[0]
print("Input name:", inp.name)
print("Input shape:", inp.shape)

