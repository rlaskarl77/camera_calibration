import yaml
import numpy as np
import cv2
from cv2 import aruco


points = []


size = 0.53
z = 0.

x_oris = [0., 2.7, 5.4, 8.1]
y_oris = [0., 2.25, 4.5, 6.75, 8.1]


ids = [61, 62, 64, 86,
       65, 71, 73, 75, 
       69, 63, 70, 74,
       None, 90, 68, 76,
       None, 91, 72, 67,]

for idx, id in enumerate(ids):
    point = []
    
    if id is None:
        continue
    
    point.append([
        x_oris[idx%4],
        y_oris[idx//4],
        z
    ])
    
    point.append([
        x_oris[idx%4] + size,
        y_oris[idx//4],
        z
    ])
    
    point.append([
        x_oris[idx%4] + size,
        y_oris[idx//4] + size,
        z
    ])
    
    point.append([
        x_oris[idx%4],
        y_oris[idx//4] + size,
        z
    ])
    
    points.append(point)
    
board = dict()
board["points"] = points
board["ids"] = [id for id in ids if id is not None]

with open('board.yaml', 'w') as f:
    yaml.dump(board, f)
