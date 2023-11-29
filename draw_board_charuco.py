import yaml
import argparse
from typing import Tuple

import numpy as np
import cv2
from cv2 import aruco


def draw_board(board: dict, board_size: Tuple[int, int], 
               margin: int, out_file: str) -> None:

    points = board["points"]
    ids = board["ids"]
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
    aruco_board = aruco.Board_create(points, aruco_dict, ids)
    
    draw_board_image(out_file, aruco_board, board_size, margin)


def draw_board_image(board_image_path:str, aruco_board, 
                     board_size: tuple=(640, 640), margin: int=20) -> None:
    
    board_image = aruco.drawPlanarBoard(aruco_board, board_size, margin)
    cv2.imwrite(board_image_path, board_image)


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


def generate_charuco() -> dict:
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    
    num_row = 8
    num_columns = 8
    board_width = .2 # in meter
    board_height = .2 # in meter
    square_length = board_width / num_row
    marker_length = 15 * 0.001 # in meter
    charuco_board = aruco.CharucoBoard.create(num_columns, num_row, square_length, marker_length, aruco_dict)
    
    board = dict()
    board["ids"] = np.array(charuco_board.ids).astype(np.int32).tolist()
    board["points"] = np.stack([points for points in charuco_board.objPoints]).astype(np.float32).tolist()
    
    with open('charuco.yaml', 'w') as f:
        yaml.dump(board, f)
    
    board_size = (640, 640)
    board_image = charuco_board.draw(board_size)
    cv2.imwrite('charuco_board.png', board_image)
    

def parse_arguments():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--board-file', type=str, help='board.yaml file path')
    parser.add_argument('--board-size', type=int, nargs=2, default=[640, 640],
                        metavar=('width', 'height'), help='board size in width, height')
    parser.add_argument('--margin', type=int, default=10, help='margin of board image')
    parser.add_argument('--out-file', type=str, default='board.png', 
                        help='board image file name')
    
    args = parser.parse_args()
    
    return args


def main():
    
    args = parse_arguments()
    
    board_file = args.board_file
    board_size = tuple(args.board_size)
    margin = args.margin
    out_file = args.out_file
    
    board = parse_board(board_file)
    draw_board(board, board_size, margin, out_file)


if __name__=='__main__':
    
    generate_charuco()