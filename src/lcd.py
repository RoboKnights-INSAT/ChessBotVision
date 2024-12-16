
from RPLCD.i2c import CharLCD
from gpiozero import Button
import time
import paho.mqtt.client as mqtt


# MQTT Configuration
broker = "localhost"
port = 1883
topic_mode = "chess/mode"
topic_elo = "chess/elo"
topic_color = "chess/color"
topic_hint = "lcd/hint"
topic_send_hint = "chess/SendHint"
topic_restart = "chess/restart"
topic_done = "chess/done"


# LCD Configuration
i2c_address = 0x27
lcd = CharLCD(i2c_expander='PCF8574', address=i2c_address, port=1, cols=16, rows=2, dotsize=8)

# GPIO Button Setup
button_proceed = Button(17)  # Proceed
button_change = Button(22)   # Change Input
button_help = Button(27)     # Help Button (Educational mode)

# Variables
current_mode = 1  # 1 = Classic, 2 = Puzzle, 3 = Educational
color = True # True for white, False for black ??
human_turn = True  
elo = 400  # Starting ELO
elo_max = 3200  # Max ELO
last_display = ["", ""]  # Cache to track last LCD content
# MQTT Callback
def on_message(client, userdata, message):
    if message.topic == topic_hint:
        hint = message.payload.decode()
        display_message('Hint:', f"{hint[0]}{hint[1]}","to",f"{hint[2]}{hint[3]}", 4)
        display_message('Educational Mode', 'Press Help!')

# MQTT Client Setup
mqtt_client = mqtt.Client()
mqtt_client.on_message = on_message
mqtt_client.connect(broker, port)
mqtt_client.subscribe(topic_hint)

# Start the loop
mqtt_client.loop_start()

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


# Greeting
def greet_player():
    """Displays the greeting message."""
    display_message('Hello, I am', 'Chess Coach', 3)
    lcd.clear()


# Mode Selection
def mode_selection():
    """Handles mode selection."""
    global current_mode
    while True:
        # Display the current mode
        if current_mode == 1:
            display_message('Mode: Classic', 'Press Proceed')
        elif current_mode == 2:
            display_message('Mode: Puzzle', 'Press Proceed')
        elif current_mode == 3:
            display_message('Mode:Educational', 'Press Proceed')

        # Cycle modes
        if button_change.is_pressed:
            current_mode = (current_mode % 3) + 1  # Cycle through modes
            time.sleep(0.4)  # Simple debounce

        # Confirm mode and go to ELO selection
        if button_proceed.is_pressed:
            mqtt_client.publish(topic_mode, current_mode)  # Publish selected mode
            elo_selection()
            break


# color selection
def color_selection():
    """Handles color selection for all game modes."""
    global color
    time.sleep(0.2)
    while True:
        display_message('Choose your',f'color: {"White" if color else "Black"}' )
        
        # change color
        if button_change.is_pressed:
            color = not color
            time.sleep(0.15)  # Simple debounce

        # Confirm ELO and start the game
        if button_proceed.is_pressed:
            mqtt_client.publish(topic_color, color)  # Publish selected ELO
            start_game()
            break


# ELO Selection
def elo_selection():
    """Handles ELO selection for all game modes."""
    global elo
    time.sleep(0.2)
    while True:
        display_message('Choose level',f'ELO: {elo}' )
        
        # Increase ELO
        if button_change.is_pressed:
            elo += 50
            if elo > elo_max:
                elo = 400  # Reset to minimum ELO
            time.sleep(0.15)  # Simple debounce

        # Confirm ELO and start the game
        if button_proceed.is_pressed:
            mqtt_client.publish(topic_elo, elo)  # Publish selected ELO
            color_selection()
            break


# Start Game
def start_game():
    """Starts the game based on the selected mode."""
    if current_mode == 1:  # Classic Mode
        display_message('Starting Game', 'Mode: Classic', 3)
        Classic_mode()
    elif current_mode == 2:  # Puzzle Mode
        display_message('Starting Game', 'Mode: Puzzle', 3)
    elif current_mode == 3:  # Educational Mode
        display_message('Starting Game', 'Mode:Educational', 3)
        educational_mode()


def check_finish_turn():
    global human_turn
    if button_proceed.is_pressed  :
        human_turn = False
        mqtt_client.publish(topic_done, "done")
        time.sleep(1)


def restart():
    display_message('Restarting...', '', 2)
    mqtt_client.publish(topic_restart, "restartiii")  # Publish restart signal
    mode_selection()  # Restart to mode selection


# Educational Mode
def educational_mode():
    """Educational mode with hints."""
    display_message('Educational Mode', 'Press Help!')
    # Check if Help button is pressed
    while True:
        start_time = None  # Track the press duration

        # Check if Help button is pressed
        while button_help.is_pressed:
            if start_time is None:
                start_time = time.time()  # Record the initial press time
            
            # Check how long the button is held
            if time.time() - start_time >= 3:  # 3 seconds hold
                restart()
                return

        # If Help button is released within 3 seconds, show a hint
        if start_time is not None and (time.time() - start_time < 3):
            display_message('wait for ', 'Hint', 4)
            mqtt_client.publish(topic_send_hint, "hetli hint") 

        # if proceed cicked
        check_finish_turn()
        
          




# Classic Mode
def Classic_mode():
    """Classic mode with hints."""
    display_message('Classic Mode')
    # Check if Help button is pressed
    while True:
        start_time = None  # Track the press duration

        # Check if Help button is pressed
        while button_help.is_pressed:
            if start_time is None:
                start_time = time.time()  # Record the initial press time
            
            # Check how long the button is held
            if time.time() - start_time >= 3:  # 3 seconds hold
                restart()
                return

        # If Help button is released within 3 seconds, show a hint
        if start_time is not None and (time.time() - start_time < 3):
            display_message('3-second click', 'for restart', 3)
            display_message('Classic Mode')
        # if proceed cicked
        check_finish_turn() 

# puzzle Mode
def puzzle_mode():
    """puzzle mode with hints."""
    display_message('Puzzle Mode')
    # Check if Help button is pressed
    while True:
        start_time = None  # Track the press duration

        # Check if Help button is pressed
        while button_help.is_pressed:
            if start_time is None:
                start_time = time.time()  # Record the initial press time
            
            # Check how long the button is held
            if time.time() - start_time >= 3:  # 3 seconds hold
                restart()
                return

        # If Help button is released within 3 seconds, show a hint
        if start_time is not None and (time.time() - start_time < 3):
            display_message('3-second click', 'for restart', 3)
            display_message('Puzzle Mode')
        # if proceed cicked
        check_finish_turn()

# Main Logic
try:
    greet_player()  # Greet the player
    mode_selection()  # Start mode selection
except KeyboardInterrupt:
    pass
finally:
    lcd.clear()