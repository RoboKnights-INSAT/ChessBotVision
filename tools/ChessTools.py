import chess

def compress_fen(fen):
    """
    Compresses a FEN representation by replacing consecutive '1's with their count.

    Args:
        fen (str): The input FEN string to be compressed.

    Returns:
        str: The compressed FEN string.
    """
    compressed_fen = []

    for row in fen.split('/'):
        count = 0
        new_row = ""

        for char in row:
            if char == '1':
                count += 1
            else:
                if count > 0:
                    new_row += str(count)
                    count = 0
                new_row += char

        if count > 0:
            new_row += str(count)

        compressed_fen.append(new_row)

    return '/'.join(compressed_fen)


def expand_fen(fen):
    """
    Expands a compressed FEN representation by replacing counts with '1's.

    Args:
        fen (str): The input compressed FEN string to be expanded.

    Returns:
        str: The expanded FEN string.
    """
    expanded_fen = []

    for row in fen.split('/'):
        new_row = ""

        for char in row:
            if char.isdigit():
                new_row += '1' * int(char)  # Repeat '1' based on the number
            else:
                new_row += char

        expanded_fen.append(new_row)

    return '/'.join(expanded_fen)

def rotate_fen_90_degrees(fen: str) -> list:
    matrix = []
    ranks = fen.split('/')
    rotated_fen = []
    #convert fen to matrix
    for rank in ranks:
        row = []
        for char in rank:
            row.append(char)
        matrix.append(row)
    #rotate fen by 90 degrees
    for j in range(8):
        rank = ''
        for i in range(7,-1,-1):
            rank += matrix[i][j]
        rotated_fen.append(rank)
    return '/'.join(rotated_fen)


def determine_chess_move(past_fen, current_fen):
    past_fen_list = past_fen.split("/")
    past_fen_list.reverse()
    current_fen_list = current_fen.split("/")
    current_fen_list.reverse()
    start = ""
    start_key = True
    start_pos = []
    finish = ""
    finish_key = True
    promote = ""
    for i in range(8):
        for j in range(8):
            if past_fen_list[i][j] != current_fen_list[i][j]:
                if (current_fen_list[i][j] == '1') and (start_key):
                    start = chr(97 + j) + str(i + 1)
                    start_pos = [i, j]
                    start_key = False
                elif finish_key:
                    finish = chr(97 + j) + str(i + 1)
                    # Check for promotion if a pawn has reached the last rank
                    if i == 7 and past_fen_list[start_pos[0]][start_pos[1]] == 'P':  # white pawn promotion
                        promote = current_fen_list[i][j].lower()
                    if i == 0 and past_fen_list[start_pos[0]][start_pos[1]] == 'p':  # black pawn promotion
                        promote = current_fen_list[i][j].lower()
                    finish_key = False
                else:
                    print("past and current fen are seperated by multiple moves")
                    return ''  # false value
    if start_key or finish_key:
        print("past and current fen are seperated by half a move or identical")
        return ''  # false value

    return start + finish + promote


def is_one_move(past_fen, current_fen):
    past_fen_list = past_fen.split("/")
    past_fen_list.reverse()
    current_fen_list = current_fen.split("/")
    current_fen_list.reverse()
    start_key = True
    finish_key = True
    
    white_pieces = ['P', 'N', 'R', 'B', 'Q', 'K']
    black_pieces = ['p', 'n', 'r', 'b', 'q', 'k']

    for i in range(8):
        for j in range(8):
            if (past_fen_list[i][j] in white_pieces + ['1'] and  current_fen_list[i][j] in black_pieces) or (past_fen_list[i][j] in black_pieces + ['1'] and  current_fen_list[i][j] in white_pieces ):
                if(finish_key):
                    finish_key = False
                else:
                    print("past and current fen are seperated by multiple moves")
                    return False  # false value
            if(past_fen_list[i][j] in white_pieces + black_pieces and current_fen_list[i][j] == '1'):
                if(start_key):
                    start_key = False
                else:
                    print("past and current fen are seperated by multiple moves")
                    return False  # false value
    if start_key or finish_key:
        print("past and current fen are seperated by half a move or identical")
        return False  # false value

    return True

def is_move_valid(past_fen, move, color):
    past_fen = compress_fen(past_fen)
    # Initialize the board with the previous FEN position
    if(color == "white"):
        past_fen += " w"
    else:
        past_fen += " b"
    past_fen += " KQkq - 0 1"
    
    board = chess.Board(past_fen)
    legal_moves_uci = [str(move) for move in board.legal_moves]

    try:
        chess_move = chess.Move.from_uci(move)
    except ValueError:
        print("Invalid move format.")
        return False
    # Check if the move is valid in the current position
    if chess_move in board.legal_moves:
        return True
    else:
        return False


def print_board_from_fen(fen):
    board = chess.Board(fen)
    print(board)


def disable_special_moves(stockfish):
    # Get the current FEN
    fen = stockfish.get_fen_position()

    # Split the FEN string into parts
    fen_parts = fen.split()

    # Remove castling rights and en passant
    fen_parts[2] = "-"  # Disable castling
    fen_parts[3] = "-"  # Disable en passant

    # Reconstruct and set the modified FEN
    modified_fen = " ".join(fen_parts)
    stockfish.set_fen_position(modified_fen)


def is_game_over(stockfish):
    evaluation = stockfish.get_evaluation()
    if evaluation["type"] == "mate":
        if evaluation["value"] == 1:
            print("Game Over: The next move will deliver checkmate.")
            return True
        else:
            print(f"Mate in {evaluation['value']} moves detected.")
    return False
