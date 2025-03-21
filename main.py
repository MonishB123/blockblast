import neat
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import time

class Spot:
    def __init__(self, color = "RED", occupied = False):
        self.color = color
        self.occupied = occupied
    def __str__(self):
        colors = {"RED" : "\033[31m", "GREEN" : "\033[32m", "YELLOW" : "\033[33m", "BLUE" : "\033[34m", "MAGENTA" : "\033[35m", "CYAN" : "\033[36m"}
        reset = "\033[0m"
        if self.occupied:
            return colors[self.color] + '\u2588' + reset
        else:
            return '\u2581'

class GameBoard:
    def __init__(self):
        self.board = [[Spot() for _ in range(8)] for _ in range(8)]
        # Dictionary of pieces with relative coordinate offsets
        self.pieces = {
            "single": {(0, 0)},  # Single block
            
            # # Line pieces
            # "2-horizontal": {(0, 0), (1, 0)},  
            # "2-vertical": {(0, 0), (0, 1)},  
            # "3-horizontal": {(0, 0), (1, 0), (2, 0)},  
            # "3-vertical": {(0, 0), (0, 1), (0, 2)},  
            # "4-horizontal": {(0, 0), (1, 0), (2, 0), (3, 0)},  
            # "4-vertical": {(0, 0), (0, 1), (0, 2), (0, 3)},  

            # # L-shaped blocks
            # "L-right": {(0, 0), (1, 0), (2, 0), (2, 1)},  
            # "L-left": {(0, 0), (1, 0), (2, 0), (0, 1)},  
            # "L-up": {(0, 0), (0, 1), (0, 2), (1, 2)},  
            # "L-down": {(0, 0), (0, 1), (0, 2), (-1, 2)},  

            # # Square (2x2)
            # "square": {(0, 0), (1, 0), (0, 1), (1, 1)},
            # "square-3x3": {(0, 0), (1, 0), (2, 0), 
            #     (0, 1), (1, 1), (2, 1), 
            #     (0, 2), (1, 2), (2, 2)},

            # # T-shaped blocks
            # "T-up": {(0, 0), (1, 0), (2, 0), (1, 1)},  
            # "T-down": {(0, 0), (1, 0), (2, 0), (1, -1)},  
            # "T-left": {(0, 0), (0, 1), (0, 2), (1, 1)},  
            # "T-right": {(0, 0), (0, 1), (0, 2), (-1, 1)},  

            # # Z-shaped blocks (S-shapes)
            # "Z-right": {(0, 0), (1, 0), (1, 1), (2, 1)},  
            # "Z-left": {(0, 0), (-1, 0), (-1, 1), (-2, 1)},  
            # "S-right": {(0, 0), (1, 0), (0, 1), (-1, 1)},  
            # "S-left": {(0, 0), (-1, 0), (0, 1), (1, 1)},  
        }

    def place_piece(self, piece_name, x, y, _color = "RED"):
        if piece_name not in self.pieces:
            return False
            

        for dx, dy in self.pieces[piece_name]:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < 8 and 0 <= new_y < 8 and not self.board[new_y][new_x].occupied:
                self.board[new_y][new_x].occupied = True
                self.board[new_y][new_x].color = _color
            else:
                return False
        
        return True
    
    def clear_lines(self):
        rows_to_clear = set()
        cols_to_clear = set()

        # Check for full rows
        for y in range(8):
            if all(self.board[y][x].occupied for x in range(8)):
                rows_to_clear.add(y)

        # Check for full columns
        for x in range(8):
            if all(self.board[y][x].occupied for y in range(8)):
                cols_to_clear.add(x)

        # Clear full rows
        for y in rows_to_clear:
            for x in range(8):
                self.board[y][x].occupied = False

        # Clear full columns
        for x in cols_to_clear:
            for y in range(8):
                self.board[y][x].occupied = False

        # Print message if anything was cleared
        if rows_to_clear or cols_to_clear:
            print(f"Cleared {len(rows_to_clear)} row(s) and {len(cols_to_clear)} column(s)!")
        else:
            #print("No lines cleared.")
            pass

        return len(rows_to_clear) + len(cols_to_clear)

    def isPossible(self, piece_names):
        """ Check if any of the given pieces can fit on the board. """
        for piece_name in piece_names:
            if piece_name not in self.pieces:
                continue

            for x in range(8):
                for y in range(8):
                    if all(
                        (0 <= x + dx < 8 and 0 <= y + dy < 8 and not self.board[y + dy][x + dx].occupied)
                        for dx, dy in self.pieces[piece_name]
                    ):
                        return True  # A valid placement exists

        return False  # No valid placement found

    def __str__(self):
        total = ""
        for row in self.board:
            for item in row:
                total += str(item)
            total += "\n"
        return total

def play_game(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    board = GameBoard()
    score = 0
    piece_names = list(board.pieces.keys())
    available_pieces = random.sample(piece_names, 1)  # Select initial 3 random pieces
    hard_limit_iters = 1000

    # Randomize board state (adjust probability as needed)
    # for i in range(8):
    #     for j in range(8):
    #         board.board[i][j].occupied = random.random() < 0.2
    # board.clear_lines()
    
    while board.isPossible(available_pieces) or hard_limit_iters < 0:  # Continue until no valid move is left
        hard_limit_iters -= 1
        # Flatten board state as input (8x8 grid -> 64 inputs)
        inputs = [int(board.board[i][j].occupied) for i in range(8) for j in range(8)]

        # One-hot encode the available pieces
        piece_encoding = [1 if piece in available_pieces else 0 for piece in piece_names]
        inputs.extend(piece_encoding)  # Append to input list

        outputs = net.activate(inputs)

        # Board placement (64 nodes)
        position_index = np.argmax(outputs[:64])  # Get highest activated position
        x, y = divmod(position_index, 8)  # Convert index to (x, y) coordinates
        # Piece selection (N nodes)
        piece_index = np.argmax(outputs[64:64+len(piece_names)])
        piece_name = piece_names[piece_index]

        # x_vals.append(x)
        # y_vals.append(y)
        # piecechosen.append(piece_name)
        if board.place_piece(piece_name, x, y, random.choice(["RED", "BLUE", "GREEN", "YELLOW", "CYAN"])):
            score += 10
            linescleared = board.clear_lines()
            score += linescleared * 80
            available_pieces.remove(piece_name)  # Remove used piece
            if not available_pieces:  # If all 3 are used, pick new ones
                available_pieces = random.sample(piece_names, 1)

        else:
            return score  # Ignore invalid moves
    
    return score


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        time.sleep(0.01)
        genome.fitness = play_game(genome, config)
        """Print the mean and median of genome connection weights."""
        # weights = [conn.weight for conn in genome.connections.values()]

        # if weights:  # Ensure there are weights before calculating statistics
        #     mean_weight = np.mean(weights)
        #     median_weight = np.median(weights)
        #     print(f"Mean weight: {mean_weight:.4f}, Median weight: {median_weight:.4f}")

def run_neat():
    config_path = "base_config.txt"  # Ensure this file is correctly formatted
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    print(config.genome_config.__dict__)
    
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    
    winner = pop.run(eval_genomes, 200)
    return winner, config

def print_game(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    board = GameBoard()
    score = 0
    piece_names = list(board.pieces.keys())
    available_pieces = random.sample(piece_names, 1)  # Select initial 3 random pieces
    
    while board.isPossible(available_pieces):  # Continue until no valid move is left
        # Flatten board state as input (8x8 grid -> 64 inputs)
        inputs = [int(board.board[i][j].occupied) for i in range(8) for j in range(8)]

        # One-hot encode the available pieces
        piece_encoding = [1 if piece in available_pieces else 0 for piece in piece_names]
        inputs.extend(piece_encoding)  # Append to input list

        # Neural network activation
        outputs = net.activate(inputs)

        # Extract move choices
        # Board placement (64 nodes)
        position_index = np.argmax(outputs[:64])  # Get highest activated position
        x, y = divmod(position_index, 8)  # Convert index to (x, y) coordinates

        # Piece selection (N nodes)
        piece_index = np.argmax(outputs[64:64+len(piece_names)])
        piece_name = piece_names[piece_index]
        print(board)
        print(available_pieces)
        print(x, y, piece_name, sep=" ")

        if board.place_piece(piece_name, x, y, random.choice(["RED", "BLUE", "GREEN", "YELLOW", "CYAN"])):
            print(board)
            score += 10
            score += board.clear_lines() * 10
            print("lines cleared: " + str((score - 10)/10))
            print(board)
            available_pieces.remove(piece_name)  # Remove used piece
            if not available_pieces:  # If all 3 are used, pick new ones
                available_pieces = random.sample(piece_names, 1)

        else:
            print("invalid move")
            return score  # Ignore invalid moves
    
    return score

if __name__ == "__main__":
    x_vals = []
    y_vals = []
    piecechosen = []
    winner, config = run_neat()
    plt.hist(x_vals)
    plt.show()
    plt.hist(y_vals)
    plt.show()
    plt.hist(piecechosen)
    plt.show()
    print_game(winner, config)
