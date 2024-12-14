import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary libraries
import time
from tools.ChessTools import *
from tools.computer_vision_tools import *
import serial
import RPi.GPIO as GPIO

from stockfish import Stockfish


turn = True
color = "white"
additional_height = 50
corners = []
chess_corner_detection_confidence = 9
chess_piece_detection_confidence = 40
grayscale_intensity_threshold = 70
image_Path = "../images/image.png"
fen0 = "rnbqkbnr/pppppppp/11111111/11111111/11111111/11111111/PPPPPPPP/RNBQKBNR"
past_fen = ""
stockfish_path = "/home/raspberrypi/Stockfish/src/stockfish"
stockfish = ""
counter_corner_detections = 0
matrice_xy_positions = [[],[],[],[],[],[],[],[]]
offset_x = 10  # Horizontal translation
offset_y = 10  # Vertical translation
gamemode = ""
elo = 0
#arduino = serial.Serial(port='/dev/ttyUSB0', baudrate=9600, timeout=1)
out_of_bound_x = -offset_x
out_of_bound_y = 4*offset_y
child_made_move = False
#GPIO.setmode(GPIO.BCM)  # Use BCM pin numbering
button_pin = 17 
#GPIO.setup(button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)



def capture_image(image_Path):
    # Capture a frame
    cap = cv2.VideoCapture(0)
    ret, img = cap.read()  # Unpack the tuple

    if ret:  # Check if the frame was captured successfully
        cv2.imwrite(image_Path, img)
    else:
        print("Failed to capture image from webcam.")

    cap.release()  # Release the webcam resource

    # future code for picamera
    # cam = Camera()
    # cam.start_preview()
    # cam.take_photo(image_Path)
    # cam.stop_preview()

def send(arduino,command):
    try:
        # Send the command to the Arduino
        arduino.write((command + '\n').encode('utf-8'))
        print(f"Sent to Arduino: {command}")
    except Exception as e:
        print(f"Error: {e}")
def receive(arduino):
    try:
        # Wait for and read the response
        response = arduino.readline().decode('utf-8').strip()
        print(f"Received from Arduino: {response}")
        return response

    except Exception as e:
        print(f"Error: {e}")
        return None



def setup():
    global turn, color, corners, chess_corner_detection_confidence, image_Path, past_fen, grayscale_intensity_threshold
    global chess_piece_detection_confidence, chess_corner_detection_confidence, fen0, stockfish_path, stockfish
    global matrice_xy_positions, offset_x, offset_y, gamemode, elo, arduino

    # Initialize the serial connection
    #arduino = serial.Serial(port='/dev/ttyUSB0', baudrate=9600, timeout=1)  # Adjust the port as needed
    
    # Generate positions for each square
    for row in range(8):
        row_positions = []
        for col in range(8):
            # Calculate x, y positions
            x = col * offset_x
            y = (7 - row) * offset_y  # Invert row numbering
            row_positions.append((x, y))
        matrice_xy_positions.append(row_positions)

    #player choses color

    # color = input("Do you want to play with white or black? (enter white/black) : ")
    # while color != "white" and color != "black":
    #     color = input("Do you want to play with white or black? (enter white/black) : ")
    turn = color == "white"

    # setup fen0 here (puzzle/normal game)

    #do lcd stuff and inputs of gamemode and elo here here

    # elo , gamemode = init_game()

    gamemode = 'classical' #default
    elo = 1200 #default

    # Initialize Stockfish engine
    stockfish = Stockfish(path=stockfish_path,parameters={"UCI_Elo": elo})  # Replace with your path
    stockfish.set_depth(15)

    while past_fen != fen0:

        # Capture a frame
        capture_image(image_Path)

        # detect board

        corners = detect_corners_local(image_Path, chess_corner_detection_confidence)

        past_fen = detect_board(image_Path,corners, additional_height,chess_piece_detection_confidence,grayscale_intensity_threshold)

        #print_board_from_fen(past_fen)

        past_fen = fix_queen_king_issue(fen0, past_fen)

        print_board_from_fen(past_fen)

        if past_fen != fen0:
            print("Please reorganize the pieces!!!")

    stockfish.set_fen_position(compress_fen(past_fen))

    #disables ne passant and king/queen side castling
    disable_special_moves(stockfish)


def main():
    global turn, color, corners, chess_corner_detection_confidence, image_Path, past_fen, grayscale_intensity_threshold
    global chess_piece_detection_confidence, chess_corner_detection_confidence, fen0, stockfish, counter_corner_detections
    global matrice_xy_positions, offset_x, offset_y, gamemode, elo, arduino, child_made_move

    try:
        while True:
            if turn:   #the chessbot arm's move
                try:
                    best_move = stockfish.get_best_move()
                    x1 = 8-int(best_move[1])
                    y1 = ord(best_move[0]) - ord('a')

                    x2 = 8 - int(best_move[3])
                    y2 = ord(best_move[2]) - ord('a')

                    if(len(best_move) == 5):
                        print("promotion here")

                    if past_fen.split()[x2][y2] != '1':
                        takes = f"{matrice_xy_positions[x2][y2][0]}:{matrice_xy_positions[x2][y2][1]}|{out_of_bound_x}:{out_of_bound_y}"
                        # give orders to arduino slave here
                        send(arduino, takes)
                        response = receive(arduino)
                        print(response)

                    orders = f"{matrice_xy_positions[x1][y1][0]}:{matrice_xy_positions[x1][y1][1]}|{matrice_xy_positions[x2][y2][0]}:{matrice_xy_positions[x2][y2][1]}"
                    # give orders to arduino slave here
                    send(arduino, orders)
                    response = receive(arduino)
                    print(response)

                    #treat response of arduino here

                    #end
                    if is_game_over(stockfish):
                        print("game is over chess robot won")
                        break

                    stockfish.make_moves_from_current_position([best_move])
                    past_fen = expand_fen(stockfish.get_fen_position().split(" ")[0])
                    turn = False

                except Exception as e:
                    print(e)

            else:
                #detect if button is pressed  and update the child_made_move (possibly using interupts to modify child_made_move
                if GPIO.input(button_pin)==GPIO.HIGH:
                    child_made_move = True

                if(child_made_move):

                    counter_detections = 0
                    move = ""
                    current_fen = ""

                    while move == "" and counter_detections<15:

                        capture_image(image_Path)

                        counter_detections += 1

                        counter_corner_detections += 1

                        #after certain number of chess piece detection redetect the corners of the chess board just in case

                        if(counter_corner_detections>50):
                            corners = detect_corners_local(image_Path, chess_corner_detection_confidence)
                            counter_corner_detections = 0

                        current_fen = detect_board(image_Path, corners, additional_height, chess_piece_detection_confidence,grayscale_intensity_threshold)

                        # print_board_from_fen(current_fen)

                        if not is_one_move(past_fen,current_fen):#check if there is more than 1 or no moves made
                            continue

                        current_fen = fix_fen(past_fen, current_fen)

                        print_board_from_fen(current_fen)

                        move = determine_chess_move(past_fen,current_fen)

                    if move == "":
                        # there was a problem in the board detection or the player made more than one move
                        print("there was a problem in the board detection or the player made more than one move")
                        child_made_move = False
                    else:
                        if is_move_valid(past_fen, move):

                            if gamemode == "educational":
                                if(move == stockfish.get_best_move()):
                                    #print in the lcd :)
                                    print("good job you got the best move")
                                else:
                                    #print in the lcd the best move
                                    print(f"dumb child the best move is {stockfish.get_best_move()} not {move}")

                            if is_game_over(stockfish) and stockfish.get_best_move() == move:
                                print("game is over chess child won")
                                break

                            stockfish.make_moves_from_current_position([move])
                            past_fen = current_fen
                            print_board_from_fen(past_fen)
                            child_made_move = False
                            turn = True

                        else:
                            print("the move is invalid bad child try again")
                            child_made_move = False
            
            GPIO.cleanup()  # Reset GPIO settings

            time.sleep(0.1)  # Debounce delay

    except KeyboardInterrupt:
        print("Exiting program.")


# Cleanup GPIO
def cleanup_gpio():
    GPIO.cleanup()
    print("GPIO cleaned up.")


# Entry point of the script
if __name__ == "__main__":
    setup()
    main()  # Run main program
    # cleanup_gpio()
