from collections import deque
import heapq
import time
from IPython.display import clear_output, display
import matplotlib.pyplot as plt
import numpy as np
import copy
import itertools


def format_states(final_moves, state):
    new_states = []

    for move in final_moves:
        d = []
        for i in range(len(move)):
            d.append(move[i][0])
        fake_state = copy.deepcopy(state)
        new_states.append(fake_state.move(d))
    return new_states


def prepare_string(index, count, state):

    alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                'U', 'V', 'W', 'X', 'Y', 'Z']
    if state.d[index] == 0:
        direction = "le centre"
    elif state.d[index] == 1:
        direction = "le haut"
    elif state.d[index] == 2:
        direction = "la droite"
    elif state.d[index] == 3:
        direction = "le bas"
    else:
        direction = "la gauche"

    if index == len(state.d) - 1:
        punctuation = '.'
    else:
        punctuation = ', '

    return "Escargot {0} vers {1}{2}".format(alphabet[count], direction, punctuation)


class Labyrinth:

    def __init__(self, lines, columns, exits, walls):
        self.nb_lines = lines
        self.nb_columns = columns
        self.exits = exits
        self.walls = walls

        self.free_pos = None

    """ est il final? """

    def success(self, state):
        # TODO
        if state.prev is not None:
            count = 0
            for i in range(len(state.d)):
                if state.pos[i] == self.exits[i]:
                    count += 1

            return count == len(state.d)

        return False

    def init_positions(self, state):
        self.free_pos = np.ones((self.nb_lines, self.nb_columns), dtype=bool)
        # TODO

        for wall in self.walls:
            self.free_pos[wall.x, wall.y] = False

        for snail in state.pos:
            self.free_pos[snail.x, snail.y] = False

        for venom in state.venoms:
            self.free_pos[venom[0], venom[1]] = False

    # THIS FUNCTION HAS BEEN CREATED BY US
    def check_if_move_in_grid(self, possible_move):

        # check for y
        left_min = possible_move.y < 0
        right_max = possible_move.y > (self.nb_columns - 1)

        # check for x
        down_max = possible_move.x > (self.nb_lines - 1)
        up_min = possible_move.x < 0

        return not (left_min or right_max or down_max or up_min)


    # THIS FUNCTION HAS BEEN CREATED BY US
    def calculate_moves(self, possible_move, d, other_exits, moves):
        if self.check_if_move_in_grid(possible_move):
            if self.free_pos[possible_move.x, possible_move.y] and possible_move not in other_exits:
                moves.append((d, possible_move))

    def possible_moves_snail(self, snail_position, exit_position, other_exits):
        moves = []
        # TODO

        if snail_position == exit_position:
            moves.append((0, snail_position))
        else:
            # moves.append((0, snail_position))

            movement1 = copy.deepcopy(snail_position)
            movement1.x -= 1
            self.calculate_moves(movement1, 1, other_exits, moves)

            movement2 = copy.deepcopy(snail_position)
            movement2.y += 1
            self.calculate_moves(movement2, 2, other_exits, moves)

            movement3 = copy.deepcopy(snail_position)
            movement3.x += 1
            self.calculate_moves(movement3, 3, other_exits, moves)

            movement4 = copy.deepcopy(snail_position)
            movement4.y -= 1
            self.calculate_moves(movement4, 4, other_exits, moves)

        return moves

    def possible_moves(self, state):
        self.init_positions(state)
        snail_moves = []
        # TODO
        for i in range(len(state.pos)):
            other_exits = copy.deepcopy(self.exits)
            other_exits.pop(i)
            snail_moves.append(self.possible_moves_snail(state.pos[i], self.exits[i], other_exits))

        possible_moves = list(itertools.product(*snail_moves))
        final_moves = copy.deepcopy(possible_moves)
        for move in possible_moves:
            count = 0
            positions = []
            for i in range(len(move)):
                if move[i][1] in positions:
                    final_moves.remove(move)
                    break
                else:
                    positions.append(move[i][1])
                if move[i][1] == state.pos[i]:
                    count += 1
            if count == len(state.pos):
                final_moves.remove(move)

        new_states = format_states(final_moves, state)

        return new_states

    def isFinalState(self, state):

        # return (state.pos[0] == self.exits[0] and state.pos[1] == self.exits[1] and state.pos[2] == self.exits[2])
        return (state.pos[0] == self.exits[0] and state.pos[1] == self.exits[1])

    def solve(self, state):
        to_visit = set()
        fifo = deque([state])
        to_visit.add(state)
        # TODO

        solution = state

        while fifo:
            s = copy.deepcopy(fifo.popleft())
            to_visit.add(s)

            if not self.success(s):
                next_states = self.possible_moves(s)
                for next in next_states:
                    if next not in to_visit:
                        fifo.append(next)
            else:
                solution = s
                break;

        return solution

    """
    Estimation du nombre de coup restants 
    """

    def estimee1(self, state):
        # TODO
        return 0

    def estimee2(self, state):
        # TODO
        return 0

    def solve_Astar(self, state):
        to_visit = set()
        to_visit.add(state)

        priority_queue = []
        state.h = self.estimee1(state)
        heapq.heappush(priority_queue, state)

        # TODO
        return None

    def print_solution(self, state):
        # TODO
        previous = state.prev
        path_states = deque([state])

        while previous.prev is not None:
            path_states.append(previous)
            previous = copy.deepcopy(previous.prev)
        index = 0
        for s in path_states:
            index += 1
            count = 0
            move_string = "{0}. ".format(index)
            for i in range(len(s.d)):
                move_string += prepare_string(i, count, s)
                count += 1
            print(move_string)

    def print_labyrinth(self, state, show_all=True):

        nb_rows = self.nb_lines
        nb_cols = self.nb_columns
        snails_str = ["L_a", "L_b", "L_c", "L_d", "L_e"]
        venom_str = "x"
        exits_str = ["[E_a]", "[E_b]", "[E_c]", "[E_d]", "[E_e]"]
        colors = ["red", "blue", "green", "pink", "yelllow"]

        if state.prev == None:
            # Prepare table
            cell_text = [["" for j in range(nb_cols + 1)] for i in range(nb_rows)]

            fig, ax = plt.subplots()
            fig.set_size_inches((nb_cols + 1) * .5, (nb_rows + 1) * .5)
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=cell_text, cellLoc='center', colColours=["gray"] * (nb_cols + 1),
                             colWidths=[0.2 for j in range(nb_cols + 1)],
                             colLabels=[""] + ['$\\bf{%d}$' % val for val in range(nb_cols)], loc='center')
            table.set_fontsize(13)
            table.auto_set_font_size(False)

            cellDict = table.get_celld()
            for i in range(nb_cols + 1):
                for j in range(nb_rows + 1):
                    cellDict[(j, i)].set_height(.2)

            for i in range(nb_rows + 1):
                table[(i, 0)].visible_edges = "open"
            for i in range(nb_cols + 1):
                table[(0, i)].visible_edges = "open"
            for i in range(nb_rows):
                table[(i + 1, 0)].get_text().set_text("$\\bf{%d}$" % i)

            # Walls
            for i in self.walls:
                table[(i.x + 1, i.y + 1)].set_facecolor("black")
            # Exits
            for index, i in enumerate(self.exits):
                table[(i.x + 1, i.y + 1)].get_text().set_text(exits_str[index])
                table[(i.x + 1, i.y + 1)].get_text().set_color(colors[index])
            # Snails
            for index, snail in enumerate(state.pos):
                table[(snail.x + 1, snail.y + 1)].get_text().set_text(snails_str[index])
                table[(snail.x + 1, snail.y + 1)].get_text().set_color(colors[index])

            if not show_all:
                clear_output(wait=True)
            fig.suptitle('Snail labyrinth (step = %d)' % 0, x=0.5, y=1.2, fontsize=16)
            display(fig)
            if not show_all:
                time.sleep(1)

            return fig, table, 1

        fig, table, n = self.print_labyrinth(state.prev, show_all)

        # Snails
        for index, snail in enumerate(state.pos):
            table[(state.prev.pos[index].x + 1, state.prev.pos[index].y + 1)].get_text().set_text(venom_str)  # Venom
            table[(snail.x + 1, snail.y + 1)].get_text().set_text(snails_str[index])
            table[(snail.x + 1, snail.y + 1)].get_text().set_color(colors[index])

        if not show_all:
            clear_output(wait=True)
        fig.suptitle('Snail labyrinth (step = %d)' % n, x=0.5, y=1.2, fontsize=16)
        display(fig)
        if not show_all:
            time.sleep(1)

        return fig, table, n + 1
