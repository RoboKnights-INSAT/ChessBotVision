# from RPLCD.i2c import CharLCD
# import time
# import RPi.GPIO as GPIO  # Import Raspberry Pi GPIO library
# # sudo python main.py
# # I2C Address (typically 0x27 or 0x3F, depending on your module)
# i2c_address = 0x27
# # LCD Configuration
# lcd = CharLCD(i2c_expander='PCF8574', address=i2c_address, port=1, cols=16, rows=2, dotsize=8)

# # GPIO Button Setup
# GPIO.setwarnings(False)  # Ignore warning for now
# GPIO.setmode(GPIO.BCM)  # Use BCM GPIO numbering

# button1Pin = 17  # Button 1: Proceed
# button2Pin = 22   # Button 2: Change Input
# button3Pin = 27  # Button 3: give instruction
# GPIO.setup(button1Pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
# GPIO.setup(button2Pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
# GPIO.setup(button3Pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# # Variables
# inModeSelect = True
# currentMode = 1  # 1 = Normal, 2 = Puzzle, 3 = Educational
# elo = 400  # Starting ELO
# eloMax = 3200  # Max ELO

# # Helper function to check if a button is pressed
# def buttonPressed(pin):
#     if GPIO.input(pin) == GPIO.LOW:  # Active LOW
#         time.sleep(0.09)  # Debounce
#         if GPIO.input(pin) == GPIO.LOW:  # Still pressed
#             return True
#     return False

# # Initial greeting
# lcd.write_string('Hello, I am     Chess Coach')
# time.sleep(2)  # Pause for 2 seconds
# lcd.clear()

# def modeSelection():
#     global inModeSelect, currentMode
#     lcd.clear()
#     while inModeSelect:
#         lcd.cursor_pos = (0, 0)
#         if currentMode == 1:
#             lcd.write_string('Mode: Classic   ')
#         elif currentMode == 2:
#             lcd.write_string('Mode: Puzzle   ')
#         else:
#             lcd.write_string('Mode: Educational      ')

#         if buttonPressed(button1Pin):  # Proceed
#             inModeSelect = False  # Exit mode select
#             lcd.clear()
#             if currentMode == 1:
#                 eloSelection("normal")
#             elif currentMode == 2:
#                 startPuzzleMode()
#             else:
#                 eloSelection("educational")
#                 # startEducationalMode()

#         if buttonPressed(button2Pin):  # Change Mode
#             currentMode = (currentMode + 1)%3   # Cycle through modes
#             time.sleep(0.1)  # Simple debounce

# def eloSelection(mode):
#     global elo
#     lcd.clear()
#     while True:
#         lcd.cursor_pos = (0, 0)
#         lcd.write_string(f'ELO: {elo}   ')

#         if buttonPressed(button1Pin):  # Confirm ELO
#             if mode == "classic":
#                 startNormalMode()
#             else:
#                 startEducationalMode()
                
#             break

#         if buttonPressed(button2Pin):  # Increment ELO
#             elo += 50
#             if elo > eloMax:
#                 elo = 400  # Reset ELO if max exceeded
#             time.sleep(0.1)  # Simple debounce

# def startNormalMode():
#     lcd.clear()
#     lcd.cursor_pos = (0, 0)
#     lcd.write_string('Starting Game')
#     lcd.cursor_pos = (1, 0)
#     lcd.write_string('Mode: Normal')
#     time.sleep(2)  # Simulate starting
#     lcd.clear()

# def startPuzzleMode():
#     lcd.clear()
#     lcd.cursor_pos = (0, 0)
#     lcd.write_string('Starting Game')
#     lcd.cursor_pos = (1, 0)
#     lcd.write_string('Mode: Puzzle')
#     time.sleep(2)  # Simulate starting
#     lcd.clear()

# def startEducationalMode():
#     lcd.clear()
#     lcd.cursor_pos = (0, 0)
#     lcd.write_string('Starting Game')
#     lcd.cursor_pos = (1, 0)
#     lcd.write_string('Mode: Edu')
#     time.sleep(2)  # Simulate starting
#     lcd.clear()
#     displayEducationalHint()

# def displayEducationalHint():
#     lcd.clear()
#     lcd.cursor_pos = (0, 0)
#     lcd.write_string('Click yellow')
#     lcd.cursor_pos = (1, 0)
#     lcd.write_string('for hint!')
#     while True:   
#         if buttonPressed(button3Pin):  # If button 3 is pressed
#             lcd.clear()
#             lcd.cursor_pos = (0, 0)
#             lcd.write_string('Hahaha nope')
#             time.sleep(3)  # Display for 3 seconds
#             lcd.clear()
#             lcd.cursor_pos = (0, 0)
#             lcd.write_string('Click yellow')
#             lcd.cursor_pos = (1, 0)
#             lcd.write_string('for hint!')
        

# # Main Logic
# modeSelection()

# # Cleanup GPIO
# GPIO.cleanup()

















# from RPLCD.i2c import CharLCD
# from gpiozero import Button
# import time

# # LCD Configuration
# i2c_address = 0x27
# lcd = CharLCD(i2c_expander='PCF8574', address=i2c_address, port=1, cols=16, rows=2, dotsize=8)

# # GPIO Button Setup
# button_proceed = Button(17)  # Proceed
# button_change = Button(22)   # Change Input
# button_help = Button(27)     # Help Button (Educational mode)

# # Variables
# current_mode = 1  # 1 = Classic, 2 = Puzzle, 3 = Educational
# elo = 400  # Starting ELO
# elo_max = 3200  # Max ELO
# last_display = ["", ""]  # Cache to track last LCD content


# # Helper function to display messages
# def display_message(line1, line2="", duration=None):
#     """Displays a message on the LCD only if it has changed."""
#     global last_display
#     if line1 != last_display[0] or line2 != last_display[1]:
#         lcd.clear()
#         lcd.cursor_pos = (0, 0)
#         lcd.write_string(line1.ljust(16))
#         lcd.cursor_pos = (1, 0)
#         lcd.write_string(line2.ljust(16))
#         last_display = [line1, line2]

#     if duration:
#         time.sleep(duration)


# # Greeting
# def greet_player():
#     """Displays the greeting message."""
#     display_message('Hello, I am', 'Chess Coach', 3)
#     lcd.clear()


# # Mode Selection
# def mode_selection():
#     """Handles mode selection."""
#     global current_mode
#     while True:
#         # Display the current mode
#         if current_mode == 1:
#             display_message('Mode: Classic', 'Press Proceed')
#         elif current_mode == 2:
#             display_message('Mode: Puzzle', 'Press Proceed')
#         elif current_mode == 3:
#             display_message('Mode:Educational', 'Press Proceed')

#         # Cycle modes
#         if button_change.is_pressed:
#             current_mode = (current_mode % 3) + 1  # Cycle through modes
#             time.sleep(0.4)  # Simple debounce

#         # Confirm mode and go to ELO selection
#         if button_proceed.is_pressed:
#             elo_selection()
#             break


# # ELO Selection
# def elo_selection():
#     """Handles ELO selection for all game modes."""
#     global elo
#     time.sleep(0.2)
#     while True:
#         display_message('Choose level',f'ELO: {elo}' )
        
#         # Increase ELO
#         if button_change.is_pressed:
#             elo += 50
#             if elo > elo_max:
#                 elo = 400  # Reset to minimum ELO
#             time.sleep(0.15)  # Simple debounce

#         # Confirm ELO and start the game
#         if button_proceed.is_pressed:
#             start_game()
#             break


# # Start Game
# def start_game():
#     """Starts the game based on the selected mode."""
#     if current_mode == 1:  # Classic Mode
#         display_message('Starting Game', 'Mode: Classic', 3)
#     elif current_mode == 2:  # Puzzle Mode
#         display_message('Starting Game', 'Mode: Puzzle', 3)
#     elif current_mode == 3:  # Educational Mode
#         display_message('Starting Game', 'Mode:Educational', 3)
#         educational_mode()


# # Educational Mode
# def educational_mode():
#     """Educational mode with hints."""
#     display_message('Educational Mode', 'Press Help!')
#     # Check if Help button is pressed
#     while True:
#         start_time = None  # Track the press duration

#         # Check if Help button is pressed
#         while button_help.is_pressed:
#             if start_time is None:
#                 start_time = time.time()  # Record the initial press time
            
#             # Check how long the button is held
#             if time.time() - start_time >= 3:  # 3 seconds hold
#                 display_message('Restarting...', '', 2)
#                 mode_selection()  # Restart to mode selection
#                 return

#         # If Help button is released within 3 seconds, show a hint
#         if start_time is not None and (time.time() - start_time < 3):
#             display_message('Hint:', 'Control the center', 3)
#             display_message('Educational Mode', 'Press Help!')



# # Main Logic
# try:
#     greet_player()  # Greet the player
#     mode_selection()  # Start mode selection
# except KeyboardInterrupt:
#     pass
# finally:
#     lcd.clear()



























from RPLCD.i2c import CharLCD
from gpiozero import Button
import time
import paho.mqtt.client as mqtt

# MQTT Configuration
broker = "broker.hivemq.com"
port = 1883
topic_mode = "chess/mode"
topic_elo = "chess/elo"

# LCD Configuration
i2c_address = 0x27
lcd = CharLCD(i2c_expander='PCF8574', address=i2c_address, port=1, cols=16, rows=2, dotsize=8)

# GPIO Button Setup
button_proceed = Button(17)  # Proceed
button_change = Button(22)   # Change Input
button_help = Button(27)     # Help Button (Educational mode)

# Variables
current_mode = 1  # 1 = Classic, 2 = Puzzle, 3 = Educational
elo = 400  # Starting ELO
elo_max = 3200  # Max ELO
last_display = ["", ""]  # Cache to track last LCD content

# MQTT Client Setup
mqtt_client = mqtt.Client()
mqtt_client.connect(broker, port)


# Helper function to display messages
def display_message(line1, line2="", duration=None):
    """Displays a message on the LCD only if it has changed."""
    global last_display
    if line1 != last_display[0] or line2 != last_display[1]:
        lcd.clear()
        lcd.cursor_pos = (0, 0)
        lcd.write_string(line1.ljust(16))
        lcd.cursor_pos = (1, 0)
        lcd.write_string(line2.ljust(16))
        last_display = [line1, line2]

    if duration:
        time.sleep(duration)


# Mode Selection
def mode_selection():
    """Handles mode selection."""
    global current_mode
    while True:
        if current_mode == 1:
            display_message('Mode: Classic', 'Press Proceed')
        elif current_mode == 2:
            display_message('Mode: Puzzle', 'Press Proceed')
        elif current_mode == 3:
            display_message('Mode:Educational', 'Press Proceed')

        if button_change.is_pressed:
            current_mode = (current_mode % 3) + 1
            time.sleep(0.4)  # Debounce

        if button_proceed.is_pressed:
            mqtt_client.publish(topic_mode, current_mode)  # Publish selected mode
            elo_selection()
            break


# ELO Selection
def elo_selection():
    """Handles ELO selection."""
    global elo
    time.sleep(0.2)
    while True:
        display_message('Choose level', f'ELO: {elo}')
        if button_change.is_pressed:
            elo += 50
            if elo > elo_max:
                elo = 400  # Reset to minimum
            time.sleep(0.15)  # Debounce

        if button_proceed.is_pressed:
            mqtt_client.publish(topic_elo, elo)  # Publish selected ELO
            start_game()
            break


# Start Game
def start_game():
    """Starts the game."""
    display_message('Starting Game', 'Mode Set', 3)


# Main Logic
try:
    display_message('Hello, I am', 'Chess Coach', 3)
    mode_selection()
except KeyboardInterrupt:
    pass
finally:
    lcd.clear()
    mqtt_client.disconnect()

