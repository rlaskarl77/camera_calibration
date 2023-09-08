# Calibrate camera with arUco markers using OpenCV and python. 

tested on:
- python: 3.10
- opencv: 4.6.0.66
  
This repository is **not guaranteed** to work with other version of OpenCV.

The calibration code of this repository is written based on [abakisita@github](https://github.com/abakisita/camera_calibration).

# Data structure

The overall folder structure should be organized as below:

```
    .
    ├── data
    │   ├── siteA
    │   │   ├── board.yaml
    │   │   ├── cam1
    │   │   │   ├── pic1.jpg
    │   │   │   └── ...
    │   │   ├── cam2
    │   │   │   ├── pic1.jpg
    │   │   │   └── ...
    │   │   └── ...
    │   ├── siteB
    │   │   └── ...
    │   └── ...
    ├── *.py
    ├── requirements.txt
    └── ...
```

# How to use

## Setup

1. git clone __this repository__
2. `cd camera_calibration`
3. install conda
4. `conda create -n cameracalib python=3.10`
5. `conda activate cameracalib`
6. `pip install -r requirements.txt`

## Prepare data

1. make `site*` folder inside `data` folder.
2. add pictures taken from the location.
3. generate `board.yaml` for each location.
   - The printed ArUco markers should be located precisely.
   - You need to add (x, y, z) positions of the ArUco marker corners.
   - Ids of the ArUco Markers should be specified as in same order.
   - You can run `draw_board.py` to see if the .yaml file is generated corretly.

## Calibrate camera

- Run `python camera_calibration.py` to get calibration parameters.
- Run `python camera_calibration.py --draw-results` to see the calibration results.
- Run `python calibrate_webcam.py` to test calibration with your webcam.
