# Empty out tf install
pip uninstall -y tensorflow

# Get current env info
pip install tensorflow==1.4.0
pip install keras==2.1.5
pip install opencv-python
pip install imageai

# Get More Models
# YOLO:
wget https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5
# RetinaNET
wget https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5
