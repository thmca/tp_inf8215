from collections import deque
import heapq
import time
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import copy
import itertools


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
        for i in range(len(state.d)):
            snail_pos = state.pos[i]
            exit_pos = self.exits[i]

        if snail_pos.x != exit_pos.x or snail_pos.y != exit_pos.y:
            return False

        return True

    def init_positions(self, state):
        self.free_pos = np.ones((self.nb_lines, self.nb_columns), dtype=bool)
        # TODO

        for wall in self.walls:
            self.free_pos[wall.x, wall.y] = False

        for snail in state.pos:
            self.free_pos[snail.x, snail.y] = False

        for venom in state.venoms:
            self.free_pos[venom[0], venom[1]] = False

    def check_if_move_in_grid(self, possible_move):
        if possible_move.x < 0 or possible_move.x > (self.nb_lines - 1) or possible_move.y < 0 or possible_move.y > (self.nb_columns -1):
            return False
        return True

    def possible_moves_snail(self, snail_position, exit_position, other_exits):
        moves = []
        # TODO

        if snail_position == exit_position:
            moves.append((0, snail_position))
        else:
            moves.append((0, snail_position))

            movement1 = copy.copy(snail_position)
            movement1.x -= 1

            if self.check_if_move_in_grid(movement1):
                if self.free_pos[movement1.x, movement1.y] and movement1 not in other_exits:
                    moves.append((1, movement1))

            movement2 = copy.copy(snail_position)
            movement2.y += 1

            if self.check_if_move_in_grid(movement2):
                if self.free_pos[movement2.x, movement2.y] and movement2 not in other_exits:
                    moves.append((2, movement2))

            movement3 = copy.copy(snail_position)
            movement3.x += 1

            if self.check_if_move_in_grid(movement3):
                if self.free_pos[movement3.x, movement3.y] and movement3 not in other_exits:
                    moves.append((3, movement3))

            movement4 = copy.copy(snail_position)
            movement4.y -= 1

            if self.check_if_move_in_grid(movement4):
                if self.free_pos[movement4.x, movement4.y] and movement4 not in other_exits:
                    moves.append((4, movement4))

        return moves

    def possible_moves(self, state):
        self.init_positions(state)
        new_states = []
        # TODO
        return new_states

    def solve(self, state):
        to_visit = set()
        fifo = deque([state])
        to_visit.add(state)
        # TODO

        return None

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
        return 0

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