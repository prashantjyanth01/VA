cd /ees_app_/pytorch-YOLOv4/
python3 demo_darknet2onnx.py /ees_app_/samples/weights/anpr_tiny_yolo_w2e/23042021/yolov4-custom-tiny-anpr_23042021.cfg /ees_app_/samples/weights/anpr_tiny_yolo_w2e/23042021/yolov4-custom-tiny-anpr_23042021.names /ees_app_/samples/weights/anpr_tiny_yolo_w2e/23042021/yolov4-custom-tiny-anpr_23042021_b1.weights
trtexec --onnx=/ees_app_/samples/weights/anpr_tiny_yolo_w2e/23042021/engine.onnx --explicitBatch --saveEngine=/ees_app_/samples/weights/anpr_tiny_yolo_w2e/23042021/model.engine --fp16
#print('MODEL EXPORTED DONE /ees_app_/samples/weights/anpr_tiny_yolo_w2e/23042021/model.engine')
