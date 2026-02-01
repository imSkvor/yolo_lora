# yolo_lora
Finetunning YOLOv9 with CoLoRA on a limited 12GB VRAM

The project heavily uses the code from official YOLOv9 github repository: https://github.com/WongKinYiu/yolov9
The CoLoRA method was proposed here: https://arxiv.org/abs/2505.18315

The work is currently in progress, I'm testing my code on a relatively small dataset of Europe traffic signes (https://universe.roboflow.com/radu-oprea-r4xnm/traffic-signs-detection-europe/dataset/14), as it already was in YOLO format. It's quite small and both `-s` and `-m` YOLOv9 models plateau with the same metrics here.
