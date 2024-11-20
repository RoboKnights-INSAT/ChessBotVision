
# Chess Coach Project

This repository contains the implementation of a **Chess Coach** system designed to enhance the chess-playing experience using AI and robotics. The system provides support for both **educational** and **normal** game modes and integrates computer vision, chess engines, and robotics.

## Features

### 1. Chessboard Detection
- Utilizes a **YOLOv8** model to detect the chessboard layout and its grid.

### 2. Chess Piece Recognition
- A **YOLOv11** model identifies the positions and types of chess pieces within each box on the board.

### 3. FEN Generation
- Converts the detected game state into **Forsyth–Edwards Notation (FEN)**, representing the current chessboard status.

### 4. Stockfish Integration
- Uses the **Stockfish chess engine** to suggest the best moves:
  - **ELO Rating**: Adjusts suggestions based on the configured skill level.
  - **Game Modes**:
    - **Normal**: The robotic arm plays the move suggested by Stockfish.
    - **Educational**: A button allows the child to see and optionally play the suggested best move.

### 5. Game Logging
- Sends the complete game details to a backend server after the game ends for analysis and storage.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/chess-coach.git
   cd chess-coach
   ```

2. **Install Dependencies**:
   - Ensure Python 3.8+ is installed.
   - Install the required Python libraries:
     ```bash
     pip install -r requirements.txt
     ```

3. **Download YOLO Models**:
   - The models are expoted in "best_chesspiece_model.pt" and "best_corner_detection_model.pt"

4. **Configure Stockfish**:
   - Download the Stockfish engine and place it in the `engines/` directory.
   - Update the configuration file to set the path to Stockfish.

5. **Connect Hardware**:
   - Ensure the robotic arm and any additional hardware are connected and configured.

## Usage

1. **Start the Chess Coach**:
   ```bash
   python main.py
   ```

2. **Select Game Mode**:
   - Choose between **Educational** or **Normal** mode in the LCD Interface.

3. **Play the Game**:
   - Follow the prompts for move suggestions, or let the robotic arm play in normal mode.

4. **End of Game**:
   - Game data will be automatically sent to the backend server.

## Project Structure

```
chess-bot-vision-TSYP/
├── images/
│   ├── image7.jpg
│   ├── image8.jpg
│   ├── image9.jpg
│   ├── image10.jpg
│   ├── image11.jpg
│   ├── image12.jpg
│   ├── image13.jpg
│   ├── image14.jpg
│   └── transformer_image.jpg
├── .gitignore                  # Git ignore file
├── app.py                      # Main application entry point
├── best_chesspiece_model.pt    # YOLO model for chess piece detection
├── best_corner_detection_model.pt # Model for chessboard corner detection
├── chessboard_transformed_with_grid.jpg # Processed chessboard visualization
├── ChessTools.py               # Utility functions for chess-specific operations
├── computer_vision_tools.py    # Computer vision tools for detection
├── main.py                     # Main logic for chess game interaction
├── notebooks/
│   ├── app.ipynb               # Jupyter notebook for app development
│   └── chess_move_detection.ipynb # Notebook for move detection analysis
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
```


## License

This project is licensed under the [MIT License](LICENSE).
