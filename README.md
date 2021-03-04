# MirrAI_AI
Sister repo: https://www.github.com/nikhil-kharbanda/MirrAI_UI.

Files for RPi deployment. It's all here boys.

## Hardware:
Oh come on.

## Dependencies:
On your das RPi4, you must install the following...

### cv2:
```bash
$ sudo apt-get install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev libavcodec-dev libavformat-dev \
                       libswscale-dev libv4l-dev libxvidcore-dev libx264-dev qt4-dev-tools libatlas-base-dev
$ sudo pip3 install opencv-python==3.4.6.27
```

### EdgeTPU Runtime:
```bash
$ echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

$ curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

$ sudo apt-get update

$ sudo apt-get install libedgetpu1-std
```

### PyCoral:
```bash
$ sudo apt-get install python3-pycoral
```

### This Repository!:
The UI depends on a /home/pi/Desktop/stylesense folder that contains all the resources needed to populate the screen.
```bash
$ mkdir stylesense && cd stylesense
$ git clone https://github.com/martinklamrowski/MirrAI_AI.git .
```