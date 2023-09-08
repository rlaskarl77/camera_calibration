import os
import yaml
from pathlib import Path
from typing import List
import argparse
from tqdm import tqdm

import numpy as np
import cv2
from cv2 import aruco
from cv2 import fisheye


ROOT = Path(__file__).parent.absolute()
DATA_PATH = ROOT.joinpath("data")
SAVE_PATH = ROOT.joinpath("test")

IMAGE_FORMATS = ('.jpg', '.jpeg', '.JPG', '.png', '.PNG')


def calibrate_camera(site_dict: dict, draw_results: bool) -> None:
    
    board = site_dict["board"]
    board_points = board["points"]
    board_ids = np.array(board["ids"])
    
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    aruco_params = aruco.DetectorParameters_create()
    
    aruco_board = aruco.Board_create(board_points, aruco_dict, board_ids)
    
    site_path = os.path.join(SAVE_PATH, site_dict["site"])
    os.makedirs(site_path, exist_ok=True)
    
    # draw ArUco board
    # draw_board_image(site_path, aruco_board)
    
    cameras = site_dict["cameras"]
    camera_dict = cameras[0]
    image_paths = camera_dict["image_paths"]
    
    counter, corner_list, id_list = [], None, None
    
    for image_path in image_paths:
        
        img = cv2.imread(image_path)
        img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        
        h, w = img_gray.shape[:2]
        if site_dict["site"]=="ch14":
            h, w = w, h
        
        corners_, ids_, rejectedImgPoints = \
            aruco.detectMarkers(img_gray, aruco_dict, parameters=aruco_params)
        
        if ids_ is None or corners_ is None:
            continue
        
        corners, ids = [], []
        
        for c, i in zip(corners_, ids_):
            if i in board_ids:
                corners.append(c)
                ids.append(i)
        
        ids = np.array(ids, dtype=np.int32)
            
        if corner_list is None:
            corner_list = corners
        else:
            corner_list = np.vstack((corner_list, corners))
        
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
    
        
    objpoints = []
    objp = np.zeros((np.sum(counter)*4, 3), np.float32)
    objp_dict = { k: v for (k, v) in zip(board_ids, board_points) }
    
    print(site_dict["site"], id_list.flatten().shape, objp.shape)
    for i, objid in enumerate(id_list.flatten()):
        objp[i*4:i*4+4] = objp_dict[objid]
    
    objpoints.append(objp)
    
    imgpoints = []
    imgcorners = np.zeros((np.sum(counter)*4, 2), np.float32)
    
    for i, corner in enumerate(corners):
        imgcorners[i*4:i*4+4] = corner
    
    imgpoints.append(imgcorners)
    
    
    result_file_path = os.path.join(site_path, "test_result.txt")
    result_file = open(result_file_path, "w")
        
    for method in ["aruco", "opencv", "fisheye", "reprojection", "ransac", "pnp"]:
        
        camera_path = os.path.join(site_path, method)
        os.makedirs(camera_path, exist_ok=True)
        
        if method=="aruco":
            
            ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraAruco(
                corners=corner_list, 
                ids=id_list, 
                counter=counter, 
                board=aruco_board, 
                imageSize=(w, h), 
                cameraMatrix=None, 
                distCoeffs=None,
                flags=cv2.CALIB_FIX_K3
            )
                    
        elif method=="ransac":
            
            best_ret = 1e9
            best_idx = 0
            best_bound = 0
        
            for i in range(len(corner_list)):
                
                cl, il, c = corner_list[i:i+1], id_list[i:i+1], np.array([1])
                
                ret_, mtx_, dist_, rvecs_, tvecs_ = aruco.calibrateCameraAruco(
                    corners=cl,
                    ids=il,
                    counter=c,
                    board=aruco_board,
                    imageSize=(w, h),
                    cameraMatrix=None,
                    distCoeffs=None
                )
                
                errors = calculate_projection_error(
                    objp=objpoints[0],
                    corners=imgpoints[0],
                    mtx=mtx_,
                    dist=dist_,
                    rvec=rvecs_[0],
                    tvec=tvecs_[0]
                )
                mask = errors < 1000
                in_bound = np.sum(mask)

                cl = [corner_list[i] for i, v in enumerate(mask) if v]
                il = id_list[mask]
                c = np.array([in_bound])
                
                # print(cl, il, c)
                
                if in_bound >= best_bound:
                    best_bound = in_bound
                    ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraAruco(
                        corners=cl,
                        ids=il,
                        counter=c,
                        board=aruco_board,
                        imageSize=(w, h),
                        cameraMatrix=None,
                        distCoeffs=None
                    )
                    
        elif method=="opencv":
            
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objectPoints=objpoints, 
                imagePoints=imgpoints, 
                imageSize=(w, h), 
                cameraMatrix=None, 
                distCoeffs=None
            )
        
        elif method=="fisheye":
            
            ret, mtx, dist, rvecs, tvecs = fisheye.calibrate(
                objectPoints=np.array(objpoints).reshape(1, 1, -1, 3), 
                imagePoints=np.array(imgpoints).reshape(1, 1, -1, 2), 
                image_size=(w, h), 
                K=np.zeros((3, 3)), 
                D=np.zeros((4, 1)),
                rvecs=[np.zeros((1, 1, 3), dtype=np.float64)],
                tvecs=[np.zeros((1, 1, 3), dtype=np.float64)],
                flags=fisheye.CALIB_FIX_SKEW+fisheye.CALIB_FIX_K3+fisheye.CALIB_FIX_K4
            )
        
        elif method=="reprojection":
            
            ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraAruco(
                corners=corner_list, 
                ids=id_list, 
                counter=counter, 
                board=aruco_board, 
                imageSize=(w, h), 
                cameraMatrix=None, 
                distCoeffs=None)
            
            detectedCorners, detectedIds, rejectedCorners, recoveredIdxs = \
                aruco.refineDetectedMarkers(image=img_gray,
                                            board=aruco_board, 
                                            detectedCorners=corner_list, 
                                            detectedIds=id_list, 
                                            rejectedCorners=rejectedImgPoints, 
                                            cameraMatrix=mtx, 
                                            distCoeffs=dist)
            
            ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraAruco(
                corners=detectedCorners, 
                ids=detectedIds, 
                counter=counter, 
                board=aruco_board, 
                imageSize=(w, h), 
                cameraMatrix=mtx, 
                distCoeffs=dist)
        
        elif method=="pnp":
            if site_dict["site"]!="ch14":
                continue
            
            intr_path = "result/ch14_m/ch14_m/intr_ch14_m.xml"
            intrinsic_file = cv2.FileStorage(intr_path, cv2.FILE_STORAGE_READ)
            
            mtx = intrinsic_file.getNode("camera_matrix").mat()
            dist = intrinsic_file.getNode("distortion_coefficients").mat()
            
            mtx = np.asarray(mtx, dtype=np.float32)
            dist = np.asarray(dist, dtype=np.float32)
            
            ret, rvec, tvec, _ = cv2.solvePnPRansac(
                objectPoints=np.array(objpoints), 
                imagePoints=np.array(imgpoints), 
                cameraMatrix=mtx, 
                distCoeffs=dist,
                )
            
            rvecs, tvecs = [rvec], [tvec]
            
            
        save_calibration_results(camera_path, mtx, dist, rvecs[0], tvecs[0])
    
        draw_calibration_results(camera_path, img, corner_list, id_list, 
                                mtx, dist, rvecs[0], tvecs[0], 2)
    
        draw_grid_on_image(camera_path, img, corner_list, id_list, 
                                mtx, dist, rvecs[0], tvecs[0], 2)
        
        draw_undistorted_image(camera_path, img, corner_list, id_list, 
                                mtx, dist, rvecs[0], tvecs[0], 2)
        
        result_file.write(f"method: {method} | rms: {ret:.5f}\n")
    
    result_file.close()


def draw_board_image(site_path:str, aruco_board, 
                     board_size: tuple=(640, 640), margin: int=20) -> None:
    
    board_image = aruco.drawPlanarBoard(aruco_board, board_size, margin, margin)
    board_image_path = os.path.join(site_path, "board.png")
    cv2.imwrite(board_image_path, board_image)
    

def draw_calibration_results(camera_path: str, img: np.ndarray, corners: tuple, 
                             ids: np.ndarray, mtx: np.ndarray, dist: np.ndarray, 
                             rvec: np.ndarray, tvec: np.ndarray, length: int=2) -> None:
    
    img = img.copy()
    result_image = aruco.drawDetectedMarkers(img, corners, ids)
    result_image = cv2.drawFrameAxes(img, mtx, dist, rvec, tvec, length)
    result_image_path = os.path.join(camera_path, f"result.png")
    cv2.imwrite(result_image_path, result_image)
    

def calculate_projection_error(objp: str, corners: tuple, mtx: np.ndarray, 
                               dist: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    
    linepts, jac = cv2.projectPoints(objp, rvec, tvec, mtx, dist)
    errors = np.zeros(len(corners)//4, dtype=np.float32)
    err = np.sum((linepts - corners)**2, axis=1)
    
    for i in range(len(errors)//4):
        errors[i] = np.sqrt(np.mean(err[i*4:i*4+4]))
    
    errors = np.nan_to_num(errors, nan=1e8)
    # print(errors)
    
    return errors
    

def draw_grid_on_image(camera_path: str, img: np.ndarray, corners: tuple, 
                        ids: np.ndarray, mtx: np.ndarray, dist: np.ndarray, 
                        rvec: np.ndarray, tvec: np.ndarray, length: int=2) -> None:
    
    img = img.copy()
    (h, w, _) = img.shape
    
    linep = np.zeros((2001*6, 3), np.float32)
    
    linep[:2001*3, 0] = np.tile(np.arange(-10, 10.01, 0.01, dtype=float), 3)
    linep[:2001, 1:2] = -10.
    linep[2001*2:2001*3, 1:2] = 10.
    
    linep[2001*3:, 1] = np.tile(np.arange(-10, 10.01, 0.01, dtype=float), 3)
    linep[2001*3:2001*4, 0:1] = -10.
    linep[2001*5:2001*6, 0:1] = 10.
    
    linepts, jac = cv2.projectPoints(linep, rvec, tvec, mtx, dist)
    
    for linep in linepts:
        # print(imgpt)
        (x, y) = map(lambda x: int(min(1e8, max(-1e8, x))), linep.flatten())
        if x<0 or x>=w or y<0 or y>=h:
            continue
        cv2.line(img, (x, y), (x, y), (25, 25, 25), 8)
    
    objp = np.zeros((41*41, 3), np.float32)
    objp[:, :2] = np.mgrid[-10:10.5:0.5, -10:10.5:0.5].T.reshape(-1, 2)
    imgpts, jac = cv2.projectPoints(objp, rvec, tvec, mtx, dist)
    
    for imgpt in imgpts:
        # print(imgpt)
        (x, y) = map(int, imgpt.flatten())
        if x<0 or x>=w or y<0 or y>=h:
            continue
        cv2.line(img, (x, y), (x, y), (205, 205, 205), 10)
    
    origin = np.zeros((1, 3), np.float32)
    oript, _ = cv2.projectPoints(origin, rvec, tvec, mtx, dist)
    
    
    (x, y) = map(int, oript.flatten())
    if not (x<0 or x>=w or y<0 or y>=h):
        cv2.line(img, (x, y), (x, y), (255, 255, 255), 15)
        cv2.putText(img, 'origin', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)
    
        
    result_image_path = os.path.join(camera_path, f"grid.png")
    cv2.imwrite(result_image_path, img)
    

def draw_undistorted_image(camera_path: str, img: np.ndarray, corners: tuple, 
                           ids: np.ndarray, mtx: np.ndarray, dist: np.ndarray, 
                           rvec: np.ndarray, tvec: np.ndarray, length: int=2) -> None:
    
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    # x, y, w, h = roi
    # cropped_image = dst[y:y+h, x:x+w]
    # cropped_image_path = os.path.join(camera_path, f"cropped.png")
    # cv2.imwrite(cropped_image_path, cropped_image)
    
    undistorted_image = cv2.drawFrameAxes(dst, newcameramtx, dist, rvec, tvec, length)
    undistorted_image_path = os.path.join(camera_path, f"undistorted.png")
    cv2.imwrite(undistorted_image_path, undistorted_image)


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
    
    site_list = [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) \
        if f in ["ch12", "ch14", "ch16"]]
    
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