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


def calibrate_camera(site_dict: dict, draw_results: bool) -> None:
    
    board = site_dict["board"]
    points = board["points"]
    ids = board["ids"]
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
    aruco_params = aruco.DetectorParameters_create()
    
    aruco_board = aruco.Board_create(points, aruco_dict, ids)
    
    site_path = os.path.join(SAVE_PATH, site_dict["site"])
    os.makedirs(site_path, exist_ok=True)
    
    if draw_results:
        draw_board_image(site_path, aruco_board)
    
    cameras = site_dict["cameras"]
    
    for camera_dict in cameras:
        try:
            image_path = camera_dict["image_path"]
            img = cv2.imread(image_path)
            img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            corners, ids, rejectedImgPoints = \
                aruco.detectMarkers(img_gray, aruco_dict, parameters=aruco_params)
            counter = np.array([len(ids)])
            
            assert len(ids)>0, \
                f'no markers are detected in camera {camera_dict["camera"]}'
            ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraAruco(
                corners, ids, counter, aruco_board, img_gray.shape, None, None)
            
            camera_path = os.path.join(site_path, camera_dict["camera"])
            os.makedirs(camera_path, exist_ok=True)
            
            save_calibration_results(camera_path, mtx, dist, rvecs[0], tvecs[0])
            
            if draw_results:
                draw_calibration_results(camera_path, img, corners, ids, 
                                         mtx, dist, rvecs[0], tvecs[0], 8)
        
        except Exception as e:
            print("exception happend with calibrating camera", e)


def draw_board_image(site_path:str, aruco_board, 
                     board_size: tuple=(640, 640), margin: int=20) -> None:
    
    board_image = aruco.drawPlanarBoard(aruco_board, board_size, margin)
    board_image_path = os.path.join(site_path, "board.png")
    cv2.imwrite(board_image_path, board_image)
    

def draw_calibration_results(camera_path: str, img: np.ndarray, corners: tuple, 
                             ids: np.ndarray, mtx: np.ndarray, dist: np.ndarray, 
                             rvec: np.ndarray, tvec: np.ndarray, length: int=8) -> None:
    
    result_image = aruco.drawDetectedMarkers(img, corners, ids)
    result_image = cv2.drawFrameAxes(img, mtx, dist, rvec, tvec, length)
    result_image_path = os.path.join(camera_path, "result.png")
    cv2.imwrite(result_image_path, result_image)


def save_calibration_results(camera_path: str, mtx: np.ndarray,
                             dist: np.ndarray, rvec: np.ndarray,
                             tvec: np.ndarray) -> None:
    
    intr_path = os.path.join(camera_path, f'intr_{Path(camera_path).stem}.xml')
    extr_path = os.path.join(camera_path, f'extr_{Path(camera_path).stem}.xml')

    intrinsic_file = cv2.FileStorage(intr_path, cv2.FILE_STORAGE_WRITE)
    intrinsic_file.write("camera_matrix", mtx)
    intrinsic_file.write("distortion_coefficients", dist)
    intrinsic_file.release()
    
    
    extrinsic_file = cv2.FileStorage(extr_path, cv2.FILE_STORAGE_WRITE)
    extrinsic_file.write("rvec", rvec)
    extrinsic_file.write("tvec", tvec)
    extrinsic_file.release()


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
    

def load_datas() -> List[dict]:
    
    datas = list()
    
    site_list = [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH)]
    
    for site in site_list:
        board_file = os.path.join(site, "board.yaml")
        
        if not os.path.exists(board_file) or not os.path.isfile(board_file):
            print(board_file)
            continue
        
        image_list = [os.path.join(site, f)
                      for f in os.listdir(site)
                      if Path(f).suffix in IMAGE_FORMATS]
        if not len(image_list)>0:
            continue
        
        site_dict = dict()
        site_dict["site"] = Path(site).stem
        site_dict["board"] = parse_board(board_file)
        site_dict["cameras"] = list()
        
        for image in image_list:
            camera_name = Path(image).stem
            image_dict = {
                "camera": camera_name,
                "image_path": image
            }
            site_dict["cameras"].append(image_dict)
        
        datas.append(site_dict)
    
    return datas
    

def parse_arguments():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--draw-results', action='store_true', 
                        help='to save calibrated results')
    
    args = parser.parse_args()
    
    return args


def main():
    
    args = parse_arguments()
    draw_results = args.draw_results
    
    datas = load_datas()
    
    for site_dict in tqdm(datas):
        calibrate_camera(site_dict, draw_results)
        

if __name__=='__main__':
    
    main()