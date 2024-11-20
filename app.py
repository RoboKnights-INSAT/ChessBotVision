from ChessTools import *
from computer_vision_tools import *

#from picamzero import Camera
from time import *

# Capture a frame
cap = cv2.VideoCapture(0)
corners = []
# cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.9)  # Reduce brightness 


print("waiting for cam to read 4 corners...")
while len(corners) != 4:
    ret, img = cap.read()  # Unpack the tuple
    if ret:  # Check if the frame was captured successfully
        image_Path = "images/image.png"
        cv2.imwrite(image_Path, img)
    else:
        print("Failed to capture image from webcam.")

    cap.release()  # Release the webcam resource
    additional_height = 20

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

transformed_image = four_point_transform(image_Path, corners,additional_height)

ptsT, ptsL = plot_grid_on_transformed_image(transformed_image,additional_height)

#make_black_spots_blacker("transformer_image.jpg","transformer_image.jpg",30)

detections, boxes =chess_pieces_detector_local(transformed_image,30)

complete_board_FEN = FEN_transformation(ptsT,ptsL,detections,boxes)

to_FEN = '/'.join(complete_board_FEN)

lichess_URL = "https://lichess.org/analysis/"+to_FEN

print(lichess_URL)

print(to_FEN)

print_board_from_fen(compress_fen(to_FEN))

elapsed_time = time() - start_time
print(f"Time taken for this iteration: {elapsed_time:.2f} seconds")

