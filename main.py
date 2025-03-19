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

    def __str__(self):
        total = ""
        for row in self.board:
            for item in row:
                total += str(item)
            total += "\n"
        return total

if __name__ == "__main__":
    myboard = GameBoard()
    print(myboard)