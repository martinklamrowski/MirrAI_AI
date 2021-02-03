# MirrAI_AI
Sister repo: https://www.github.com/nikhil-kharbanda/MirrAI_UI

## Dependencies for running inference on RPi4 (regular old tensorflow):
### Regular old tensorflow 1.15:
```bash
$ sudo apt-get install -y libhdf5-dev libc-ares-dev libeigen3-dev gcc gfortran python-dev libgfortran5 \
                          libatlas3-base libatlas-base-dev libopenblas-dev libopenblas-base libblas-dev \
                          liblapack-dev cython openmpi-bin libopenmpi-dev libatlas-base-dev python3-dev
$ sudo pip3 install keras_applications==1.0.8 --no-deps
$ sudo pip3 install keras_preprocessing==1.1.0 --no-deps
$ sudo pip3 install h5py==2.9.0
$ sudo pip3 install pybind11
$ pip3 install -U --user six wheel mock
$ sudo pip3 uninstall tensorflow
$ wget "https://raw.githubusercontent.com/PINTO0309/Tensorflow-bin/master/tensorflow-1.15.0-cp37-cp37m-linux_armv7l_download.sh"
$ sudo sh ./tensorflow-1.15.0-cp37-cp37m-linux_armv7l_download.sh
$ sudo pip3 install tensorflow-1.15.0-cp37-cp37m-linux_armv7l.whl

【Required】 Restart the terminal.
```

### cv2:
```bash
$ sudo apt-get install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev libavcodec-dev libavformat-dev \
                       libswscale-dev libv4l-dev libxvidcore-dev libx264-dev qt4-dev-tools libatlas-base-dev
$ sudo pip3 install opencv-python==3.4.6.27
```
***
## Clone the repo:
```bash
$ git clone https://github.com/martinklamrowski/MirrAI_AI.git
```
### -(for RPi4 deployment)
If you're cloning to deploy on RPi4, you only need the following files/dirs:
 - camera_capture.py, pi_script.py, res/util, res/yolo, res/checkpoints (empty for now), res/modanet.names.

To download checkpoint (pi_script.py uses the checkpoint, not the weights):
 - Will host it somewhere eventually.

### -(for raising your blood pressure)
If you're cloning to train, tweak, etc. keep everything.

To download chictopia and modanet >>> eeeeeeeeeyyyyyyah.

To download model weights:
 - Will host them somewhere eventually.

To get .tflite or .pb files from weights, run the requisite script.

