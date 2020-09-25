import numpy as np
import copy


def change_position(snail_position, d):
    if d == 1:
        snail_position.x -= 1
    elif d == 2:
        snail_position.y += 1
    elif d == 3:
        snail_position.x += 1
    elif d == 4:
        snail_position.y -= 1

    return snail_position


class State:
    """
    Contructeur d'un état initial
    """

    def __init__(self, pos):
        """
        pos donne la position du escargot i;
        """
        self.pos = np.array(copy.deepcopy(pos))
        """
        venoms garde les positions du poisons
        """
        self.venoms = set()

        """
        d et prev premettent de retracer l'état précédent et le dernier mouvement effectué
        """
        self.d = self.prev = None

        self.nb_moves = 0
        self.h = 0

    """
    Constructeur d'un état à partir mouvement (d)
    """

    def move(self, d):
        # TODO

        newState = copy.deepcopy(self)
        newState.prev = copy.deepcopy(self)
        newState.d = d

        for i in range(len(d)):

            pos_snail = self.pos[i]
            d_snail = d[i]

            if d_snail != 0:
                temp_pos = (pos_snail.x, pos_snail.y)
                newState.venoms.add(temp_pos)
                new_pos = change_position(pos_snail, d_snail)
                newState.pos[i] = new_pos

        return newState

    def __eq__(self, other):
        if not isinstance(other, State):
            return NotImplemented
        if len(self.pos) != len(other.pos):
            print("les états n'ont pas le même nombre de limaces")

        return (len(self.venoms) == len(other.venoms) and np.array_equal(self.pos,
                                                                         other.pos) and self.venoms == other.venoms)

    def __hash__(self):
        h = 0
        for snail_position in self.pos:
            h += snail_position.x * 25 + snail_position.y * 30
        for venom_position in self.venoms:
            h += snail_position.x * 25 + snail_position.y * 30
        return int(h)

    def __lt__(self, other):
        return (self.nb_moves + self.h) < (other.nb_moves + other.h)

    def __repr__(self):
        out = ""
        for index, limace in enumerate(self.pos):
            out += "Escargot %d (%d,%d) | " % (index + 1, limace.x, limace.y)
        return "State: " + out[:-3]

    def __str__(self):
        out = ""
        for index, limace in enumerate(self.pos):
            out += "Escargot %d (%d,%d) | " % (index + 1, limace.x, limace.y)
        return out[:-3]
