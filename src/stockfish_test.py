from stockfish import Stockfish

# Path to your Stockfish binary
stockfish_path = "/home/raspberrypi/Stockfish/src/stockfish"
elo = 1000  # Setting the Elo of Stockfish

# Initialize Stockfish engine
stockfish = Stockfish(path=stockfish_path, parameters={"UCI_Elo": elo})

# Set the depth of search (higher depth = stronger engine)
stockfish.set_depth(15)

# Function to print the current board in a readable format
def print_board(stockfish):
    board = stockfish.get_board_visual()
    print(board)

# Function to simulate a game
def simulate_game():
    stockfish.set_position("rnbqkbnr/ppppppp/11111111/11111111/11111111/11111111/PPPPPPPP/RNBQKBNR")  # Start from the initial position

    # Play until the game ends
    while True:
        # Get the best move for the current player (Stockfish plays for both sides)
        move = stockfish.get_best_move()
        print(move)
        
        if not move:
            print("Game over")
            break

        # Make the move on the board
        stockfish.make_moves_from_current_position([move])
        
        # Print the current board after the move
        print(f"Move: {move}")
        print_board(stockfish)
        break


if __name__ == "__main__":
    simulate_game()

