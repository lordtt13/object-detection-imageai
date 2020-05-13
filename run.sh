# Empty out tf install
pip uninstall -y tensorflow

# Get current env info
pip install tensorflow==1.4.0
pip install keras==2.1.5
pip install opencv-python
pip install imageai

# Download Model
wget https://github.com/lordtt13/object-detection-imageai/blob/master/yolo-tiny.h5?raw=true
