import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.ChessTools import *
from tools.computer_vision_tools import *

#from picamzero import Camera
from time import *

# Capture a frame
cap = cv2.VideoCapture(0)
corners = []
# cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.9)  # Reduce brightness

additional_height = 20
chess_piece_detection_confidence = 40
grayscale_intensity_threshold = 70
past_fen = "rnbkqbnr/pppppppp/11111111/11111111/11111111/11111111/PPPPPPPP/RNBQKBNR"
image_Path = "../images/image.png"

print("waiting for cam to read 4 corners...")
while len(corners) != 4:
    ret, img = cap.read()  # Unpack the tuple
    if ret:  # Check if the frame was captured successfully
        cv2.imwrite(image_Path, img)
    else:
        print("Failed to capture image from webcam.")

    cap.release()  # Release the webcam resource

    corners = detect_corners_local(image_Path,confidence=9)
    sleep(0.1)
#api models

# corners = detect_corners_API(image_Path,9)
#
# transformed_image = four_point_transform(image_Path, corners,additional_height)
#
# ptsT, ptsL = plot_grid_on_transformed_image(transformed_image,additional_height)
#
# detections, boxes = chess_pieces_detector_API(transformed_image)
#
# complete_board_FEN = FEN_transformation(ptsT,ptsL,detections,boxes)
#
# to_FEN = '/'.join(complete_board_FEN)
#
# lichess_URL = "https://lichess.org/analysis/"+to_FEN
#
# print(lichess_URL)
#
# print(to_FEN)
# print_board_from_fen(compress_fen(to_FEN))


#local models
#while(1):

start_time = time()

current_fen = detect_board(image_Path, corners, additional_height, chess_piece_detection_confidence,
                           grayscale_intensity_threshold)

current_fen = rotate_fen_90_degrees(current_fen)

print("detected board" , current_fen)
print_board_from_fen(compress_fen(current_fen))


if not is_one_move(past_fen, current_fen):  # check if there is more than 1 or no moves made
    print("doodoo")
else:
    current_fen = fix_fen(past_fen, current_fen)

    print(current_fen)
    print_board_from_fen(compress_fen(current_fen))

    move = determine_chess_move(past_fen, current_fen)

    print(move)


elapsed_time = time() - start_time
print(f"Time taken for this iteration: {elapsed_time:.2f} seconds")

