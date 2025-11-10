from rfdetr import RFDETRBase

if __name__ == '__main__':

    model = RFDETRBase()

    model.train(
        dataset_dir=r"D:\Pycharm\rf_detr\dataset",
        epochs=12,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir=r"D:\Pycharm\rf_detr\output_model"
    )