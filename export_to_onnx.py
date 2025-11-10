from rfdetr import  RFDETRBase

model = RFDETRBase(pretrain_weights = r"D:\Pycharm\rf_detr\output_model\checkpoint_best_total.pth")

model.export()