import os
import yaml
from pathlib import Path
from typing import List
import argparse
from tqdm import tqdm

import numpy as np
import cv2
from cv2 import aruco

import torch
import torch.nn.functional as F
import torch.optim as optim


ROOT = Path(__file__).parent.absolute()
DATA_PATH = ROOT.joinpath("data_multiple")
RESULT_PATH = ROOT.joinpath("result_single")
SAVE_PATH = ROOT.joinpath("result_multiple")

IMAGE_FORMATS = ('.jpg', '.jpeg', '.JPG', '.png', '.PNG')

MAX_EPOCHS = 1000
LR = 1e-1


def calibrate_camera_all(site_dict: dict, res: dict, draw_results: bool) -> None:
    
    undistort_images = False
    
    board = site_dict["board"]
    points = board["points"]
    board_ids = np.array(board["ids"])
    
    print(board)
    
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    aruco_params = aruco.DetectorParameters_create()
    
    aruco_board = aruco.Board_create(points, aruco_dict, board_ids)
    
    site_path = os.path.join(SAVE_PATH, site_dict["site"])
    os.makedirs(site_path, exist_ok=True)
    
    if draw_results:
        draw_board_image(site_path, aruco_board)
    
    cameras = site_dict["cameras"]
    camera_results = res["cameras"]
    
    
    all_cameras = [camera_dict["camera"] for camera_dict in cameras]
    camera_intrinsics = []
    camera_extrinsics = []
    distortions = []
    id_indicators = []
    all_corners = []
    all_points = points.copy() # N x 4 x 3
    
    for camera_dict, camera_res in zip(cameras, camera_results):
        
        print(camera_dict["camera"], camera_res["camera"])
        
        image_paths = camera_dict["image_paths"]
        
        
        mtx = camera_res["camera_matrix"]
        dist = camera_res["distortion_coefficients"]
        
        mtx = np.array(mtx, dtype=np.float32)
        dist = np.array(dist, dtype=np.float32)
        
        # undistort image
        if undistort_images:
            img0 = cv2.imread(image_paths[0])
            h, w = img0.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w, h))
            mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
        
        counter, corner_list, id_list = [], None, None
        
        image_path = image_paths[0]
        
        if undistort_images:
            img = cv2.imread(image_path)
            img_undist = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
            img_gray = cv2.cvtColor(img_undist,cv2.COLOR_RGB2GRAY)
        else:
            img = cv2.imread(image_path)
            img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        
        corners_, ids_, rejectedImgPoints = \
            aruco.detectMarkers(img_gray, aruco_dict, parameters=aruco_params)
        
        if ids_ is None or corners_ is None:
            continue
        
        ids, id_idxes, board_id_idxes = \
            np.intersect1d(ids_, board_ids, return_indices=True)
            
        if len(ids)==0:
            continue
        
        corners = np.array(corners_)[id_idxes]
        
        if id_list is None:
            id_list = ids
        else:
            id_list = np.hstack((id_list, ids))
            
        if corner_list is None:
            corner_list = corners
        else:
            corner_list = np.vstack((corner_list, corners))
        
        counter.append(len(ids))
        
        assert id_list is not None and len(id_list)>0, \
            'at least one markers need to be detected'
        counter = np.array(counter)
                
        assert len(corner_list)>0, \
            f'no markers are detected in camera {camera_dict["camera"]}'
        
        newdist = None
        if not undistort_images:
            newcameramtx = mtx
            newdist = dist
        
        ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraAruco(
            corners=corner_list, 
            ids=id_list, 
            counter=counter, 
            board=aruco_board, 
            imageSize=img_gray.shape[::-1], 
            cameraMatrix=newcameramtx, 
            distCoeffs=newdist)
    
        camera_path = os.path.join(site_path, camera_dict["camera"])
        os.makedirs(camera_path, exist_ok=True)
        
        save_calibration_results(camera_path, mtx, dist, rvecs[0], tvecs[0], '_0')
    
        if draw_results:
            draw_calibration_results(camera_path, img, corner_list, id_list, 
                                    mtx, dist, rvecs[0], tvecs[0], 2)
            
        
        # save parameters
        if undistort_images:
        
            newcameramtx = np.array(newcameramtx, dtype=np.float32)
            camera_intrinsics.append(newcameramtx)
        else:
            camera_intrinsics.append(mtx)
        
        rot = np.array(cv2.Rodrigues(rvecs[0])[0], dtype=np.float32)
        trs = np.array(tvecs[0], dtype=np.float32).reshape(3, 1)
        
        extrinsics = np.hstack((rot, trs))
        camera_extrinsics.append(extrinsics)
        
        id_indicator = np.zeros((len(board_ids), 4), dtype=np.int32) # (N, 4)
        
        for board_id_idx in board_id_idxes:
            
            id_indicator[board_id_idx] = np.ones(4, dtype=np.int32)
            
        id_indicators.append(id_indicator)
        
        corners_all = np.zeros((len(board_ids), 4, 2), dtype=np.float32) 
        
        for board_id_idx, corner in zip(board_id_idxes, corners):
            corners_all[board_id_idx] = corner[0]
        
        corners_all = corners_all.transpose() # (N, 4, 2)
        all_corners.append(corners_all)
        
        distortions.append(dist)
        
    camera_intrinsics = np.array(camera_intrinsics, dtype=np.float32)
    camera_extrinsics = np.array(camera_extrinsics, dtype=np.float32)
    id_indicators = np.array(id_indicators, dtype=np.int32) # C x N x 4
    all_corners = np.array(all_corners, dtype=np.float32) # C x N x 4 x 2
    
    camera_intrinsics = torch.from_numpy(camera_intrinsics).to(torch.float32) # C x 3 x 3
    camera_extrinsics = torch.from_numpy(camera_extrinsics).to(torch.float32) # C x 3 x 4
    id_indicators = torch.from_numpy(id_indicators).to(torch.float32) # C x N x 4
    all_corners = torch.from_numpy(all_corners).to(torch.float32) # C x N x 4 x 2
    all_points = torch.from_numpy(all_points).repeat(camera_intrinsics.shape[0], 1, 1, 1).to(torch.float32) # C x N x 4 x 3
    all_points = torch.cat((all_points, torch.ones_like(all_points[:, :, :, :1])), dim=-1) # C x N x 4 x 4
    
    all_points = all_points.transpose(3, 2) # C x 4 x N x 4
    all_points = all_points.reshape(camera_intrinsics.shape[0], 4, -1) # C x 4 x (4 x N)
    
    all_corners = all_corners.transpose(3, 2) # C x 2 x N x 4
    all_corners = all_corners.reshape(camera_intrinsics.shape[0], 2, -1) # C x 2 x (4 x N)
    
    id_indicators = id_indicators.reshape(camera_intrinsics.shape[0], -1) # C x (4 x N)
    id_indicators = id_indicators.unsqueeze(1) # C x 1 x (4 x N)
    
    all_corners = all_corners * id_indicators # C x 2 x (4 x N)
    
    camera_extrinsics.requires_grad = True
    
    
    optimizer = optim.SGD([camera_extrinsics], lr=LR, momentum=0.9)
    for epoch in tqdm(range(MAX_EPOCHS)):
        optimizer.zero_grad()
        
        threed2twod = torch.matmul(camera_intrinsics, camera_extrinsics)
        projected2d = torch.matmul(threed2twod, all_points) # C x 3 x (4 x N)
        projected2d = projected2d[:, :2, :] / projected2d[:, 2:, :] 
        
        projected2d = projected2d * id_indicators # C x 2 x (4 x N)
        
        error = F.mse_loss(projected2d, all_corners)
        print("epoch", epoch, error.detach().numpy())
        error.backward()
        optimizer.step()
        # torch.save(camera_extrinsics, os.path.join(site_path, f"extrinsics_e{epoch}.pt"))
        
    print("final_loss", error)
    torch.save(camera_extrinsics, os.path.join(site_path, "extrinsics.pt"))
    
    if error>1e-2:
        print("calibration failed")
        return
    
    for c_idx, cam in enumerate(cameras):
        camera_path = os.path.join(site_path, cam["camera"])
        os.makedirs(camera_path, exist_ok=True)
        mtx = camera_intrinsics[c_idx].numpy()
        camera_extrinsics = camera_extrinsics.detach()
        rvecs = camera_extrinsics[c_idx][:, :3].numpy()
        rvecs = cv2.Rodrigues(rvecs)[0]
        tvecs = camera_extrinsics[c_idx][:, 3:].numpy()
        
        dist = distortions[c_idx]
        save_calibration_results(camera_path, mtx, dist, 
                                 rvecs, tvecs, 
                                 '_final')
        
        
        image_paths = cam["image_paths"]
        
        print(f'saving {cam["camera"]} from image {image_paths[0]}')
        
        # undistort image
        if undistort_images:
            img0 = cv2.imread(image_paths[0])
            h, w = img0.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w, h))
            mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
        
        counter, corner_list, id_list = [], None, None
        
        image_path = image_paths[0]
        
        if undistort_images:
            
            img = cv2.imread(image_path)
            img_undist = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
            img_gray = cv2.cvtColor(img_undist,cv2.COLOR_RGB2GRAY)
        else:
            img = cv2.imread(image_path)
            img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        
    
        camera_path = os.path.join(site_path, camera_dict["camera"])
        os.makedirs(camera_path, exist_ok=True)
        
        if draw_results:
            draw_calibration_results(camera_path, img, corner_list, id_list, 
                                    mtx, dist, rvecs, tvecs, 2, '_final')


def draw_board_image(site_path:str, aruco_board, 
                     board_size: tuple=(640, 640), margin: int=20) -> None:
    
    board_image = aruco.drawPlanarBoard(aruco_board, board_size, margin, margin)
    board_image_path = os.path.join(site_path, "board.png")
    cv2.imwrite(board_image_path, board_image)
    

def draw_calibration_results(camera_path: str, img: np.ndarray, corners: tuple, 
                             ids: np.ndarray, mtx: np.ndarray, dist: np.ndarray, 
                             rvec: np.ndarray, tvec: np.ndarray, length: float=0.5,
                             tag: str='') -> None:
    
    img = img.copy()
    result_image = aruco.drawDetectedMarkers(img, corners, ids)
    result_image = cv2.drawFrameAxes(img, mtx, dist, rvec, tvec, length)
    result_image_path = os.path.join(camera_path, f"result{tag}.png")
    cv2.imwrite(result_image_path, result_image)


def save_calibration_results(camera_path: str, mtx: np.ndarray,
                             dist: np.ndarray, rvec: np.ndarray,
                             tvec: np.ndarray, tag: str='') -> None:
    
    intr_path = os.path.join(camera_path, f'intr_{Path(camera_path).stem}{tag}.xml')
    extr_path = os.path.join(camera_path, f'extr_{Path(camera_path).stem}{tag}.xml')

    intrinsic_file = cv2.FileStorage(intr_path, cv2.FILE_STORAGE_WRITE)
    intrinsic_file.write("camera_matrix", mtx)
    intrinsic_file.write("distortion_coefficients", dist)
    intrinsic_file.release()
    
    
    extrinsic_file = cv2.FileStorage(extr_path, cv2.FILE_STORAGE_WRITE)
    extrinsic_file.write("rvec", rvec)
    extrinsic_file.write("tvec", tvec)
    extrinsic_file.release()
    

def load_calibration_results(camera_path: str) -> tuple:
        
    intr_path = os.path.join(camera_path, f'intr_{Path(camera_path).stem}.xml')
    extr_path = os.path.join(camera_path, f'extr_{Path(camera_path).stem}.xml')
    
    intrinsic_file = cv2.FileStorage(intr_path, cv2.FILE_STORAGE_READ)
    camera_matrix = intrinsic_file.getNode("camera_matrix").mat()
    distortion_coefficients = intrinsic_file.getNode("distortion_coefficients").mat()
    intrinsic_file.release()
    
    extrinsic_file = cv2.FileStorage(extr_path, cv2.FILE_STORAGE_READ)
    rvec = extrinsic_file.getNode("rvec").mat()
    tvec = extrinsic_file.getNode("tvec").mat()
    extrinsic_file.release()
    
    return camera_matrix, distortion_coefficients, rvec, tvec


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


def load_results() -> List[dict]:
    
    results = list()
    
    site_list = [os.path.join(RESULT_PATH, f) for f in os.listdir(RESULT_PATH)]
    
    for site in site_list:
        camera_list = [os.path.join(site, f)
                      for f in os.listdir(site)
                      if os.path.isdir(os.path.join(site, f))]
        if not len(camera_list)>0:
            continue
        
        site_dict = dict()
        site_dict["site"] = Path(site).stem
        site_dict["cameras"] = list()
        
        for camera in camera_list:
            camera_name = Path(camera).stem
            camera_matrix, distortion_coefficients, rvec, tvec = \
                load_calibration_results(camera)
            camera_dict = {
                "camera": camera_name,
                "camera_path": camera,
                "camera_matrix": camera_matrix,
                "distortion_coefficients": distortion_coefficients,
                "rvec": rvec,
                "tvec": tvec,
            }
            site_dict["cameras"].append(camera_dict)
        
        results.append(site_dict)
        
    return results
    
    

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
    results = load_results()
    
    for site_dict, res in tqdm(zip(datas, results)):
        calibrate_camera_all(site_dict, res, draw_results)
        

if __name__=='__main__':
    
    main()