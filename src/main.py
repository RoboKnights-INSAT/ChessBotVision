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
import paho.mqtt.client as mqtt

import requests

# The base URL for your API
BASE_URL = 'http://192.168.31.125:5000/api/api'

def game_update(arm_id, fen, move_number, move):
    url = f'{BASE_URL}/game-update'
    data = {
        'arm_id': arm_id,
        'fen': fen + " w KQkq - 0 1",
        'move_number': move_number,
        'move': move,
    }
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        return response.json()  # Return the response JSON if successful
    except requests.exceptions.RequestException as e:
        return {'error': str(e)}

def initialize_game(arm_id, initial_fen):
    url = f'{BASE_URL}/initialize-game'
    data = {
        'arm_id': arm_id,
        'initial_fen': initial_fen + " w KQkq - 0 1",
    }
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        return response.json()  # Return the response JSON if successful
    except requests.exceptions.RequestException as e:
        return {'error': str(e)}

def end_game(arm_id, result, winner=None):
    url = f'{BASE_URL}/end-game'
    data = {
        'arm_id': arm_id,
        'result': result,
    }
    if winner:
        data['winner'] = winner
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        return response.json()  # Return the response JSON if successful
    except requests.exceptions.RequestException as e:
        return {'error': str(e)}

# MQTT Configuration
broker = "localhost"
port = 1883
topic_mode = "chess/mode"
topic_elo = "chess/elo"
topic_color = "chess/color"
topic_send_hint = "chess/SendHint"
topic_hint = "lcd/hint"
topic_restart = "chess/restart"
topic_done = "chess/done"

# Variables
arm_id = "beta-arm"
gamemode = ""
elo = 0
gameStarted = False
turn = True
color = ""
color_child = ""
additional_height = 50
corners = []
chess_corner_detection_confidence = 9
chess_piece_detection_confidence = 25
grayscale_intensity_threshold = 70
image_Path = "../images/image.png"
fen0 = "rnbqkbnr/pppppppp/11111111/11111111/11111111/11111111/PPPPPPPP/RNBQKBNR"
past_fen = ""
stockfish_path = "/home/raspberrypi/Stockfish/src/stockfish"
stockfish = ""
counter_corner_detections = 0
# matrice_xy_positions = [[],[],[],[],[],[],[],[]]
matrice_xy_positions = []
offset_x = 1  # Horizontal translation
offset_y = 1  # Vertical translation
# gamemode = ""
# elo = 0
arduino = serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout=1)
out_of_bound_x = 0
out_of_bound_y = 0
child_made_move = False
#GPIO.setmode(GPIO.BCM)  # Use BCM pin numbering
button_pin = 17 
#GPIO.setup(button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
moves_counter = 2



# Callback for when the client receives a message
def on_message(client, userdata, message):
    global gamemode, elo, color, gameStarted, child_made_move
    if message.topic == topic_mode:
        mode = int(message.payload.decode())
        gamemode = {1: "classical", 2: "puzzle", 3: "educational"}.get(mode, "unknown")
        print(f"Game mode set to: {gamemode}")

    elif message.topic == topic_elo:
        elo = int(message.payload.decode())
        print(f"ELO set to: {elo}")

    elif message.topic == topic_color:
        color = eval(message.payload.decode())
        color = "white" if color else "black"
        print(f"color set to: {color}")
        gameStarted = True
    
    elif message.topic == topic_send_hint:
        hint = stockfish.get_best_move()
        print("my hint:",hint)
        client.publish(topic_hint, hint)  # Publish selected mode
    elif message.topic == topic_restart:
        print(f"Restarting game...")
        gameStarted = False
        client.loop_stop()  # Stop the MQTT loop
        os.execv(sys.executable, ['python3'] + sys.argv)  # Restart the script


    elif message.topic == topic_done:
        print(f"move done...")
        # data = {
        #     'key1': 'value1',
        #     'key2': 'value2'
        # }
        # response = requests.post(url, json=data)
        child_made_move = True

        
    




    


# Main function to set up MQTT
def setup_mqtt():
    client = mqtt.Client()
    client.on_message = on_message

    client.connect(broker, port)
    client.subscribe(topic_mode)
    client.subscribe(topic_elo)
    client.subscribe(topic_color)
    client.subscribe(topic_send_hint)
    client.subscribe(topic_restart)
    client.subscribe(topic_done)

    client.loop_start()  # Non-blocking loop to keep receiving messages
    return client

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
        return response

    except Exception as e:
        print(f"Error: {e}")
        return None



def setup():
    global turn, color, corners, chess_corner_detection_confidence, image_Path, past_fen, grayscale_intensity_threshold
    global chess_piece_detection_confidence, chess_corner_detection_confidence, fen0, stockfish_path, stockfish
    global matrice_xy_positions, offset_x, offset_y, gamemode, elo, arduino, color_child

    # Initialize the serial connection
    arduino = serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout=1)  # Adjust the port as needed
    mqtt_client = setup_mqtt()
    print("waiting for user to choose settings...")
    while not gameStarted :
        pass
    print(f"Starting game in mode:{gamemode} with elo: {elo}...  ")
    print(f"The user is playing {color}...  ")
    # Generate positions for each square
    for row in range(8):
        row_positions = []
        for col in range(8):
            # Calculate x, y positions
            x = col * offset_x + 1
            y = (7 - row) * offset_y + 1 # Invert row numbering
            row_positions.append((x, y))
        matrice_xy_positions.append(row_positions)

    #player choses color
    if(color == "white"):
        color_child = "black"
    else:
        color_child = "white"
    # color = input("Do you want to play with white or black? (enter white/black) : ")
    # while color != "white" and color != "black":
    #     color = input("Do you want to play with white or black? (enter white/black) : ")
    turn = color == "white"

    # setup fen0 here (puzzle/normal game)

    #do lcd stuff and inputs of gamemode and elo here here

    # elo , gamemode = init_game()
    
    # Initialize Stockfish engine
    stockfish = Stockfish(path=stockfish_path,parameters={"UCI_Elo": elo})  # Replace with your path
    stockfish.set_depth(15)
    # past_fen = "rnbqkbnr/pppp1ppp/1111p111/1111111111/1111P111/11111111/PPPP1PPP/RNBQKBNR"     #........................................................................................................................................
    # past_fen = rotate_fen_90_degrees(past_fen) #........................................................................................................................................
    # past_fen = rotate_fen_90_degrees(past_fen) #........................................................................................................................................


    while past_fen != fen0:
        # break    #........................................................................................................................................
        corners = []
        # detect board
        while len(corners) != 4:
            print("detecting corners")
            # Capture a frame
            capture_image(image_Path)

            corners = detect_corners_local(image_Path, chess_corner_detection_confidence)
            if(len(corners)!=4):
                print("didn't detect 4 corners")
            time.sleep(0.1)
        print("detected 4 corners")

        past_fen = detect_board(image_Path,corners, additional_height,chess_piece_detection_confidence,grayscale_intensity_threshold)

        #print_board_from_fen(past_fen)
        past_fen = rotate_fen_90_degrees(past_fen)

        past_fen = fix_fen(fen0, past_fen)

        print("detected past fen",past_fen)

        print_board_from_fen(compress_fen(past_fen))

        if past_fen != fen0:
            print("Please reorganize the pieces!!!")

    
    #set initial position
    stockfish.set_position(compress_fen(past_fen))

    #disables ne passant and king/queen side castling
    disable_special_moves(stockfish)


def main():
    global turn, color, corners, chess_corner_detection_confidence, image_Path, past_fen, grayscale_intensity_threshold
    global chess_piece_detection_confidence, chess_corner_detection_confidence, fen0, stockfish, counter_corner_detections
    global matrice_xy_positions, offset_x, offset_y, gamemode, elo, arduino, child_made_move, color_child

    try:
        while True:
            if turn:   #the chessbot arm's move
                print("Robot turn",color)
                try:
                    best_move = stockfish.get_best_move()
                    print(best_move)
                    x1 = 8-int(best_move[1])
                    y1 = ord(best_move[0]) - ord('a')

                    x2 = 8 - int(best_move[3])
                    y2 = ord(best_move[2]) - ord('a')
                    if(len(best_move) == 5):
                        print("promotion here")
                    # past_fen = [                       #........................................................................................................................................
                    #       ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'] ,
                    #       ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'], 
                            
                    #       ['1', '1', '1', '1', '1', '1', '1', '1'],  
                    #       ['1', '1', '1', '1', '1', '1', '1', '1'],  
                    #       ['1', '1', '1', '1', '1', '1', '1', '1'],  
                    #       ['1', '1', '1', '1', '1', '1', '1', '1'] ,
                    #         ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'], 
                    #         ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
                            
                            
                    #     ]
                    
                    # if False:                  #........................................................................................................................................
                    if past_fen.split('/')[x2][y2] != '1':        #........................................................................................................................................
                        takes = f"{matrice_xy_positions[x2][y2][0]}:{matrice_xy_positions[x2][y2][1]}|{out_of_bound_x}:{out_of_bound_y}"
                        # give orders to arduino slave here
                        send(arduino, takes)

                        response = ""
                        print("waiting for arduino response")
                        while not response :
                            response = receive(arduino) 
                        print("received",response)
                        
                        response = ""
                    print(matrice_xy_positions)      #........................................................................................................................................
                    orders = f"{matrice_xy_positions[x1][y1][0]}:{matrice_xy_positions[x1][y1][1]}|{matrice_xy_positions[x2][y2][0]}:{matrice_xy_positions[x2][y2][1]}"     #........................................................................................................................................
                    # orders = f"{  -(matrice_xy_positions[x1][y1][0]+1 - 8)}:{-(matrice_xy_positions[x1][y1][1]+ 1  - 8)}|{-(matrice_xy_positions[x2][y2][0]+1 - 8)}:{- (matrice_xy_positions[x2][y2][1]+1 - 8)}"     #........................................................................................................................................

                    send(arduino, orders)
                    response = ""
                    while not response :
                        response = receive(arduino) 
                    print("received",response)
                    response = ""

                    #treat response of arduino here

                    #end
                    if is_game_over(stockfish):
                        print("game is over chess robot won")
                        break

                    stockfish.make_moves_from_current_position([best_move])
                    past_fen = expand_fen(stockfish.get_fen_position().split(" ")[0])
                    print(past_fen)
                    print_board_from_fen(compress_fen(past_fen))
                    turn = False

                except Exception as e:
                    print(e)

            else:
                
                if(child_made_move):

                    counter_detections = 0
                    move = ""
                    current_fen = ""

                    while move == "" and counter_detections<100:

                        capture_image(image_Path)

                        counter_detections += 1

                        counter_corner_detections += 1

                        #after certain number of chess piece detection redetect the corners of the chess board just in case

                        if(counter_corner_detections>50):
                            corners = []
                            while len(corners) != 4:
                                print("detecting corners")
                                # Capture a frame
                                capture_image(image_Path)

                                corners = detect_corners_local(image_Path, chess_corner_detection_confidence)
                                if(len(corners)!=4):
                                    print("didn't detect 4 corners")
                                time.sleep(0.1)
                            counter_corner_detections = 0

                        current_fen = detect_board(image_Path, corners, additional_height, chess_piece_detection_confidence,grayscale_intensity_threshold)
                        
                        current_fen = rotate_fen_90_degrees(current_fen)
                        print("detected current fen",current_fen)
                        print_board_from_fen(compress_fen(current_fen))

                        if not is_one_move(past_fen,current_fen):#check if there is more than 1 or no moves made
                            continue

                        current_fen = fix_fen(past_fen, current_fen)

                        print("fixed fen" ,current_fen)
                        print_board_from_fen(compress_fen(current_fen))

                        move = determine_chess_move(past_fen,current_fen)
                        print("the childs move:" ,move)

                    if move == "":
                        # there was a problem in the board detection or the player made more than one move
                        print("there was a problem in the board detection or the player made more than one move")
                        child_made_move = False
                    else:
                        if is_move_valid(past_fen, move, color_child):
                            if gamemode == "educational":
                                if(move == stockfish.get_best_move()):
                                    #print in the lcd :)
                                    print("good job you got the best move")
                                else:
                                    #print in the lcd the best move
                                    print(f"The best move is {stockfish.get_best_move()} not {move}")

                            if is_game_over(stockfish) and stockfish.get_best_move() == move:
                                print("game is over chess player won")
                                break

                            stockfish.make_moves_from_current_position([move])
                            past_fen = current_fen
                            print_board_from_fen(compress_fen(past_fen))
                            child_made_move = False
                            turn = True

                        else:
                            print("the move is invalid try again")
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
