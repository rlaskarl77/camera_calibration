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
    board_ids = np.array(board["ids"])
    
    # print(board)
    
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    aruco_params = aruco.DetectorParameters_create()
    
    aruco_board = aruco.Board_create(points, aruco_dict, board_ids)
    
    site_path = os.path.join(SAVE_PATH, site_dict["site"])
    os.makedirs(site_path, exist_ok=True)
    
    # if draw_results:
    #     draw_board_image(site_path, aruco_board)
    
    cameras = site_dict["cameras"]
    
    for camera_dict in cameras:
        print(camera_dict["camera"])
        
        if camera_dict["camera"]!='ch12_m':
            continue
        
        try:
            if True:
                image_paths = camera_dict["image_paths"]
                counter, corner_list, id_list = [], None, None
                
                # for image_path in tqdm(image_paths[::3000]):
                for image_path in image_paths:
                    
                    img = cv2.imread(image_path)
                    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
                    # print(img_gray.shape)
                    
                    corners, ids, rejectedImgPoints = \
                        aruco.detectMarkers(img_gray, aruco_dict, parameters=aruco_params)
                    
                    if ids is None or corners is None:
                        continue
                    
                    if 17 in ids:
                        img_result = img.copy()
                        
                    if corner_list is None:
                        corner_list = corners
                    else:
                        corner_list = np.vstack((corner_list, corners))
                    
                    # ids = np.intersect1d(ids, board_ids)
                    # print(ids)
                    
                    if id_list is None:
                        id_list = ids 
                    else:
                        id_list = np.vstack((id_list, ids))
                    
                    counter.append(len(ids))
                
                assert id_list is not None and len(id_list)>0, \
                    'at least one markers need to be detected'
                counter = np.array(counter)
                    
                assert len(corner_list)>0, \
                    f'no markers are detected in camera {camera_dict["camera"]}'
                    
                ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraAruco(
                    corners=corner_list, 
                    ids=id_list, 
                    counter=counter, 
                    board=aruco_board, 
                    imageSize=img_gray.shape[::-1], 
                    cameraMatrix=None, 
                    distCoeffs=None)
                    
                    # objPoints = aruco_board.objPoints
                    # print(objPoints)
                    # objIds = aruco_board.ids
                    
                    # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                    #     cl, il, c, aruco_board, img_gray.shape, None, None)
                
                camera_path = os.path.join(site_path, camera_dict["camera"])
                os.makedirs(camera_path, exist_ok=True)
                
                save_calibration_results(camera_path, mtx, dist, rvecs[0], tvecs[0])
            
                if draw_results:
                    draw_calibration_results(camera_path, img_result, corner_list, id_list, 
                                            mtx, dist, rvecs[-1], tvecs[-1], 2)
        
        except Exception as e:
            print("exception happend with calibrating camera:", e)


def draw_board_image(site_path:str, aruco_board, 
                     board_size: tuple=(640, 640), margin: int=20) -> None:
    
    board_image = aruco.drawPlanarBoard(aruco_board, board_size, margin, margin)
    board_image_path = os.path.join(site_path, "board.png")
    cv2.imwrite(board_image_path, board_image)
    

def draw_calibration_results(camera_path: str, img: np.ndarray, corners: tuple, 
                             ids: np.ndarray, mtx: np.ndarray, dist: np.ndarray, 
                             rvec: np.ndarray, tvec: np.ndarray, length: int=8) -> None:
    
    print("printing")
    
    img = img.copy()
    img2 = img.copy()
    result_image = aruco.drawDetectedMarkers(img, corners, ids)
    result_image = cv2.drawFrameAxes(img, mtx, dist, rvec, tvec, length)
    result_image_path = os.path.join(camera_path, f"result.png")
    cv2.imwrite(result_image_path, result_image)
    
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)
    
    # crop the image
    # x, y, w, h = roi
    # dst = dst[y:y+h, x:x+w]
    
    undist_image_path = os.path.join(camera_path, f"undistort.png")
    cv2.imwrite(undist_image_path, dst)
    
    # ori = cv2.drawFrameAxes(dst, newcameramtx, dist, rvec, tvec, 3)
    
    print("printed")


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
        
        camera_list = [os.path.join(site, f)
                      for f in os.listdir(site)
                      if os.path.isdir(os.path.join(site, f))]
        if not len(camera_list)>0:
            continue
        
        site_dict = dict()
        site_dict["site"] = Path(site).stem
        site_dict["board"] = parse_board(board_file)
        site_dict["cameras"] = list()
        
        for camera in camera_list:
            camera_name = Path(camera).stem
            image_list = [os.path.join(camera, f)
                          for f in os.listdir(camera)
                          if Path(f).suffix in IMAGE_FORMATS]
            camera_dict = {
                "camera": camera_name,
                "image_paths": image_list
            }
            site_dict["cameras"].append(camera_dict)
        
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