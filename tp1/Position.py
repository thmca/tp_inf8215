import math

class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, other):
        return math.fabs(other.x - self.x) + math.fabs(other.y - self.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __repr__(self):
        return "(%d,%d)" % (self.x, self.y)

    def __str__(self):
        return "(%d,%d)" % (self.x, self.y)

    def __hash__(self):
        return int(self.x * 1 + self.y * 2)