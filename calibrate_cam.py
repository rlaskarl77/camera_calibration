import os
import yaml
from pathlib import Path
from typing import List
import argparse
from tqdm import tqdm

import numpy as np
import cv2
from cv2 import aruco


ROOT = Path(__file__).parent.absolute()
DATA_PATH = ROOT.joinpath("data")
SAVE_PATH = ROOT.joinpath("result")

IMAGE_FORMATS = ('.jpg', '.jpeg', '.JPG', '.png', '.PNG')


def calibrate_webcam(board: dict, board_params: dict, image_path: str) -> None:
    
    points = board["points"]
    ids = board["ids"]
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    aruco_params = aruco.DetectorParameters_create()
    
    aruco_board = aruco.Board_create(points, aruco_dict, ids)

    
    mtx = board_params["mtx"]
    dist = board_params["dist"]
    mtx = np.asarray(mtx, dtype=np.float32)
    dist = np.asarray(dist, dtype=np.float32)

    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    h, w = img_gray.shape[:2]

    corners, ids, rejectedImgPoints = \
        aruco.detectMarkers(img, aruco_dict, parameters=aruco_params)
        
    if corners == None:
        print("no markers are detected")
    else:
        ret, rvec, tvec = \
            aruco.estimatePoseBoard(corners, ids, aruco_board, mtx, dist, None, None)
        print(f"rotation: {rvec}\ntraslation: {tvec}")
        if ret != 0:
            img_aruco = aruco.drawDetectedMarkers(img, corners, ids, (0,255,0))
            img_aruco = cv2.drawFrameAxes(img_aruco, mtx, dist, rvec, tvec, 8)

    cv2.imwrite(os.path.join('calib', f'calib_{Path(image_path).name}'), img_aruco)


def load_calibration_results(intr_path: str) -> dict:
    
    board_params = dict()

    intrinsic_file = cv2.FileStorage(intr_path, cv2.FILE_STORAGE_READ)
    mtx = intrinsic_file.getNode("camera_matrix").mat()
    dist = intrinsic_file.getNode("distortion_coefficients").mat()
    
    board_params["mtx"] = mtx
    board_params["dist"] = dist
    
    intrinsic_file.release()
    
    return board_params


def parse_board(board_file: str) -> dict:
    
    with open(board_file, 'r') as f:
        board = yaml.load(f, Loader=yaml.FullLoader)
    
    assert "points" in board.keys(), f'"points" should be specified in {board_file}'
    assert "ids" in board.keys(), f'"ids" should be specified in {board_file}'
    
    board["points"] = np.asarray(board["points"], dtype=np.float32)
    board["ids"] = np.asarray(board["ids"], dtype=np.int32)
    
    assert board["points"].shape[-2:]==(4, 3), \
        f'the shape of "points" is not valid: \n'\
        + f'it should be (N, 4, 3) however it is board["points"].shape'
    assert len(board["points"])==len(board["ids"]), \
        f'the number of given points and their specified ids do not match\n' \
        + f'the number of giben points is {len(board["points"])}\n' \
        + f'but the number of ids is {len(board["ids"])}'
    
    return board
    

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--board-file', type=str, help='board.yaml file path')
    parser.add_argument('--intr-path', type=str, help='intr_cam*.yaml file path')
    parser.add_argument('--image-path', type=str, help='image file to calibrate')
    args = parser.parse_args()
    
    return args


def main():
    args = parse_arguments()
    
    board_file = args.board_file
    intr_path = args.intr_path
    image_path = args.image_path
    
    board = parse_board(board_file)
    board_params = load_calibration_results(intr_path)
    
    calibrate_webcam(board, board_params, image_path)
        

if __name__=='__main__':
    main()