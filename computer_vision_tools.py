import os
from dotenv import load_dotenv

from ultralytics import YOLO
from matplotlib.pyplot import figure
from matplotlib import pyplot as plt

import numpy as np
from numpy import asarray
from PIL import Image

import cv2

from shapely.geometry import Polygon
from inference_sdk import InferenceHTTPClient



def order_points(pts):
    # order a list of 4 coordinates:
    # 0: top-left,
    # 1: top-right
    # 2: bottom-right,
    # 3: bottom-left

    if len(pts) < 4:
        raise ValueError("At least 4 points are required.")

    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


# calculates iou between two polygons

def calculate_iou(box_1, box_2):
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou


def detect_corners_API(image_Path, confidence=50):
    load_dotenv()
    api_key = os.getenv('ROBOFLOW_API_KEY')

    image = Image.open(image_Path)
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=api_key
    )

    result = CLIENT.infer(image, model_id=f"chessboard-detection-yqcnu/3?confidence={confidence}")

    # Extracting the x, y coordinates of each corner
    corners = [[prediction['x'], prediction['y']] for prediction in result['predictions']]
    return np.array(corners, dtype="float32")


def detect_corners_local(image_Path, confidence=50):
    model = YOLO("best_corner_detection_model.pt")

    # Run detection on the image
    results = model.predict(image_Path, conf=confidence / 100)
    #print(results)
    # Access the first element of the results list
    first_result = results[0]

    for detection in results[0].boxes:  # Access the first result and iterate over each detected box
        # Extract corner coordinates
        x1, y1, x2, y2 = detection.xyxy[0]  # Bounding box coordinates
        # Extract confidence score
        conf_score = detection.conf[0]
        # Extract class label
        class_label = detection.cls[0]

        # print(f"Corner detected at ({x1:.2f}, {y1:.2f}), ({x2:.2f}, {y2:.2f}) with confidence {conf_score:.2f} and label {class_label}")

    if first_result.boxes is not None:
        corners = [[(boxes[0] + boxes[2]) / 2, (boxes[1] + boxes[3]) / 2] for boxes in
                   first_result.boxes.xyxy.cpu().numpy()]
        # print(corners)
        corners = corners[:4]

    else:
        corners = np.array([])  # If no corners are detected, return an empty array

    return np.array(corners, dtype="float32")


# perspective transforms an image with four given corners

def four_point_transform(image, pts, additional_height=0):
    img = Image.open(image)
    # img.show()
    image = asarray(img)
    rect = order_points(pts)

    rect[0][1] = rect[0][1] - additional_height
    rect[1][1] = rect[1][1] - additional_height

    (tl, tr, br, bl) = rect

    # compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # construct set of destination points to obtain a "birds eye view"
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    img = Image.fromarray(warped, "RGB")
    # img.show()
    # return the warped image
    img.save("transformer_image.jpg")
    return img


# calculates chessboard grid

def plot_grid_on_transformed_image(image, additional_height=0):
    corners = np.array([[0, 0],
                        [image.size[0], 0],
                        [0, image.size[1]],
                        [image.size[0], image.size[1]]])

    corners = order_points(corners)

    figure(figsize=(10, 10), dpi=80)

    # im = plt.imread(image)
    implot = plt.imshow(image)

    TL = corners[0]
    BL = corners[3]
    TR = corners[1]
    BR = corners[2]

    def interpolate(xy0, xy1):
        x0, y0 = xy0
        x1, y1 = xy1
        dx = (x1 - x0) / 8
        dy = (y1 - y0) / 8
        pts = [(x0 + i * dx, y0 + i * dy) for i in range(9)]
        return pts

    def interpolate_with_extra_height(xy0, xy1, height):
        x0, y0 = xy0
        x1, y1 = xy1
        dx = (x1 - x0) / 8
        dy = (y1 - height - y0) / 8
        pts = [(x0 + i * dx, y0 + i * dy + (i != 0) * height) for i in range(9)]
        return pts

    ptsT = interpolate(TL, TR)
    ptsL = interpolate_with_extra_height(TL, BL, additional_height)
    ptsR = interpolate_with_extra_height(TR, BR, additional_height)
    ptsB = interpolate(BL, BR)

    for a, b in zip(ptsL, ptsR):
        plt.plot([a[0], b[0]], [a[1], b[1]], 'ro', linestyle="--")
    for a, b in zip(ptsT, ptsB):
        plt.plot([a[0], b[0]], [a[1], b[1]], 'ro', linestyle="--")

    plt.axis('off')

    plt.savefig('chessboard_transformed_with_grid.jpg')
    return ptsT, ptsL


# detects chess pieces

# detects chess pieces

def chess_pieces_detector_API(image):
    # model_trained = YOLO("best_transformed_detection.pt")
    # results = model_trained.predict(source=image, line_thickness=1, conf=0.5, augment=False, save_txt=True, save=True)
    #
    # print("results\n",results)
    #
    # boxes = results[0].boxes
    # detections = boxes.xyxy.numpy()

    load_dotenv()
    api_key = os.getenv('ROBOFLOW_API_KEY')

    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=api_key
    )

    result = CLIENT.infer(image, model_id="sp2-ym1iq-9on5y/1")

    boxes = [i["class_id"] for i in result["predictions"]]

    detections = [[i["x"] - i["width"] / 2, i["y"] - i["height"] / 2, i["x"] + i["width"] / 2, i["y"] + i["height"] / 2]
                  for i in result["predictions"]]
    detections = np.array(detections, dtype="float32")

    return detections, boxes

def chess_pieces_detector_local(image, confidence=50):
    model = YOLO("best_chesspiece_model.pt")

    # Run detection on the image
    results = model.predict("transformer_image.jpg", conf=confidence / 100)

    # Access the first element of the results list
    first_result = results[0]

    boxes = [int(box.cls[0].item()) for box in first_result.boxes]
    detections = [box.xyxy[0].tolist() for box in first_result.boxes]

    return detections, boxes


def connect_detection_to_square(squares, detection_box):
    list_of_iou = []

    box_x1 = detection_box[0]
    box_y1 = detection_box[1]

    box_x2 = detection_box[2]
    box_y2 = detection_box[1]

    box_x3 = detection_box[2]
    box_y3 = detection_box[3]

    box_x4 = detection_box[0]
    box_y4 = detection_box[3]

    # cut high pieces
    if box_y4 - box_y1 > 60:
        box_complete = np.array([[box_x1, box_y1 + 40], [box_x2, box_y2 + 40], [box_x3, box_y3], [box_x4, box_y4]])
    else:
        box_complete = np.array([[box_x1, box_y1], [box_x2, box_y2], [box_x3, box_y3], [box_x4, box_y4]])

    for i in range(8):
        for j in range(8):
            list_of_iou.append(calculate_iou(box_complete, squares[i][j]))
    num = list_of_iou.index(max(list_of_iou))

    return num // 8, num % 8  # returns the position of the square that matches the piece detection the most


def FEN_transformation(ptsT, ptsL, detections, boxes):
    # Extract x-coordinates and y-coordinates from points
    x_coords = [pt[0] for pt in ptsT]
    y_coords = [pt[1] for pt in ptsL]

    # Initialize the FEN annotation grid
    FEN_annotation = []

    # Iterate to create grid squares
    for row in range(8):  # Rows 8 to 1
        current_row = []
        for col in range(8):  # Columns a to h
            square = np.array([
                [x_coords[col], y_coords[row + 1]],
                [x_coords[col + 1], y_coords[row + 1]],
                [x_coords[col + 1], y_coords[row]],
                [x_coords[col], y_coords[row]]
            ])
            current_row.append(square)
        FEN_annotation.append(current_row)

    board_FEN = [['1' for i in range(8)] for j in range(8)]

    di = {0: 'b', 1: 'k', 2: 'n',
          3: 'p', 4: 'q', 5: 'r',
          6: 'B', 7: 'K', 8: 'N',
          9: 'P', 10: 'Q', 11: 'R'}

    for piece_number, detection_box in zip(boxes, detections):
        i, j = connect_detection_to_square(FEN_annotation, detection_box)
        board_FEN[i][j] = di[piece_number]

    print(board_FEN)
    complete_board_FEN = [''.join(line) for line in board_FEN]
    return complete_board_FEN


def fix_queen_king_issue(past_fen, current_fen):
    past_fen_list = [list(row) for row in past_fen.split("/")]
    current_fen_list = [list(row) for row in current_fen.split("/")]
    updated_fen_list = [list(row) for row in current_fen.split("/")]

    # Here we assume that the past fen is correct and the current fen is always right for any detections other than queen and king
    for i in range(8):
        for j in range(8):
            # 1. checks if king was confused with a queen
            # a. static
            if (past_fen_list[i][j] == 'k') and (current_fen_list[i][j] in ['q', 'n', 'r', 'b', 'p']):
                updated_fen_list[i][j] = 'k'
            # b. king made a move
            if (current_fen_list[i][j] == '1') and (past_fen_list[i][j] == 'k'):  # the king made a move
                # get all the neighbouring positions
                positions = [[x, y] for x in range(max(i - 1, 0), min(i + 1, 7) + 1) for y in
                             range(max(j - 1, 0), min(j + 1, 7) + 1) if not (x == i and y == j)]
                for pos in positions:
                    if current_fen_list[pos[0]][pos[1]] == 'q' and past_fen_list[pos[0]][pos[1]] in ['Q', 'B', 'P', 'R',
                                                                                                     'N', '1']:
                        updated_fen_list[pos[0]][pos[1]] = 'k'
                        break
            # same logic for white king
            if (past_fen_list[i][j] == 'K') and (current_fen_list[i][j] in ['Q', 'N', 'R', 'B', 'P']):
                updated_fen_list[i][j] = 'K'
                # b. king made a move
            if (current_fen_list[i][j] == '1') and (past_fen_list[i][j] == 'K'):  # the king made a move
                # get all the neighbouring positions
                positions = [[x, y] for x in range(max(i - 1, 0), min(i + 1, 7) + 1) for y in
                             range(max(j - 1, 0), min(j + 1, 7) + 1) if not (x == i and y == j)]
                for pos in positions:
                    if current_fen_list[pos[0]][pos[1]] == 'Q' and past_fen_list[pos[0]][pos[1]] in ['q', 'b', 'p', 'r',
                                                                                                     'n', '1']:
                        updated_fen_list[pos[0]][pos[1]] = 'K'
                        break
            # 2. check if queen was confused with king
            # a. static
            if past_fen_list[i][j] in 'q' and current_fen_list[i][j] in ['k', 'n', 'r', 'b', 'p']:
                updated_fen_list[i][j] = 'q'

            # b. dynamic (queen made a move)
            if past_fen_list[i][j] == 'q' and current_fen_list[i][j] == '1':
                # we find all possible positions that he queen could take
                position = [[x, j] for x in range(8) if x != i] + [[i, y] for y in range(8) if y != j] + [[i + x, j + x]
                                                                                                          for x in
                                                                                                          range(-7, 8)
                                                                                                          if
                                                                                                          0 <= i + x <= 7 and 0 <= j + x <= 7 and x != 0] + [
                               [i - x, j + x] for x in range(-7, 8) if 0 <= i - x <= 7 and 0 <= j + x <= 7 and x != 0]

                for pos in position:
                    if current_fen_list[pos[0]][pos[1]] == 'k' and past_fen_list[pos[0]][pos[1]] not in ['k', 'b', 'p',
                                                                                                         'r', 'n']:
                        updated_fen_list[pos[0]][pos[1]] = 'q'
                        break
            # check for other color
            # a. static
            if past_fen_list[i][j] in 'Q' and current_fen_list[i][j] in ['K', 'N', 'R', 'B', 'P']:
                updated_fen_list[i][j] = 'Q'

            # b. dynamic (queen made a move)
            if past_fen_list[i][j] == 'Q' and current_fen_list[i][j] == '1':
                # we find all possible positions that he queen could take
                position = [[x, j] for x in range(8) if x != i] + [[i, y] for y in range(8) if y != j] + [[i + x, j + x]
                                                                                                          for x in
                                                                                                          range(-7, 8)
                                                                                                          if
                                                                                                          0 <= i + x <= 7 and 0 <= j + x <= 7 and x != 0] + [
                               [i - x, j + x] for x in range(-7, 8) if 0 <= i - x <= 7 and 0 <= j + x <= 7 and x != 0]

                for pos in position:
                    if current_fen_list[pos[0]][pos[1]] == 'K' and past_fen_list[pos[0]][pos[1]] not in ['K', 'B', 'P',
                                                                                                         'R', 'N']:
                        updated_fen_list[pos[0]][pos[1]] = 'Q'
                        break

    return '/'.join([''.join(row) for row in updated_fen_list])


def make_black_spots_blacker(image_path, output_path, threshold=50):
    """
    Enhances the black spots/pieces in the image by making them darker.

    Parameters:
    - image_path: Path to the input image.
    - output_path: Path to save the output image.
    - threshold: Grayscale intensity below which pixels are considered "dark" (0-255).
    """
    # Open the image
    img = Image.open(image_path).convert("RGB")

    # Convert to grayscale to identify dark areas
    grayscale = img.convert("L")

    # Create a mask for dark areas
    mask = grayscale.point(lambda p: 255 if p < threshold else 0, mode="1")

    # Darken the black spots
    darkened = img.point(lambda p: p * 0.5 if p < threshold else p)

    # Combine the darkened areas with the original image
    img = Image.composite(darkened, img, mask)

    # Save the modified image
    img.save(output_path)
    print(f"Image saved to {output_path}")

def detect_board(image_Path,corners, additional_height,chess_piece_detection_confidence, grayscale_intensity_threshold):

    transformed_image = four_point_transform(image_Path, corners, additional_height)

    make_black_spots_blacker("transformer_image.jpg", "transformer_image.jpg", grayscale_intensity_threshold)

    ptsT, ptsL = plot_grid_on_transformed_image(transformed_image, additional_height)

    detections, boxes = chess_pieces_detector_local(transformed_image, chess_piece_detection_confidence)

    complete_board_FEN = FEN_transformation(ptsT, ptsL, detections, boxes)

    current_fen = '/'.join(complete_board_FEN)

    return current_fen