from Position import Position
from State import State
from Labyrinth import Labyrinth



def test1():
    b = True
    s0 = State([Position(0, 0), Position(3, 0)])
    s1 = s0.move([2, 2])
    print(s1.prev == s0)
    b = b and s1.prev == s0
    print(s0.pos[1], " ", s1.pos[1])
    s1 = s1.move([0, 0])
    print(s1 == s1.prev)
    b = b and s1 == s1.prev
    s1 = s1.move([3, 3]).move([4, 4]).move([1, 1])
    print(s1.venoms)
    s2 = s0.move([3, 3]).move([2, 2]).move([1, 1]).move([0, 0]).move([4, 4])
    print(s1 == s2)
    b = b and s1 == s2

    print("\nrésultat correct" if b else "\nmauvais résultat")


def test2():
    b = True
    lb = Labyrinth(5, 7,
                   [Position(0, 4), Position(0, 6)],
                   [Position(0, 3), Position(1, 3), Position(2, 3), Position(2, 2), Position(2, 4)])
    s = State([Position(0, 0), Position(3, 0)])
    s = s.move([3, 3]).move([3, 2]).move([2, 2]).move([3, 2]).move([2, 2]).move([2, 2]).move([2, 2]).move([2, 1]).move(
        [1, 1]).move([1, 1]).move([1, 1]).move([4, 0])
    b = b and lb.success(s)
    print(lb.success(s))

    lb = Labyrinth(8, 8,
                   [Position(0, 4), Position(0, 6), Position(7, 0)],
                   [Position(0, 3), Position(1, 3), Position(2, 3), Position(2, 2), Position(2, 4), Position(6, 2),
                    Position(6, 4), Position(6, 5), Position(6, 6), Position(7, 1), Position(7, 2), Position(7, 3),
                    Position(7, 4)])
    s = State([Position(0, 0), Position(3, 0), Position(7, 7)])
    s = s.move([3, 3, 1]).move([3, 2, 1]).move([2, 2, 4]).move([3, 2, 4]).move([2, 2, 4]).move([2, 2, 4]).move(
        [2, 2, 4]).move([2, 1, 4]).move([1, 1, 4]).move([1, 1, 3]).move([1, 1, 3]).move([4, 0, 0])
    b = b and lb.success(s)
    print(lb.success(s))
    s = s.move([2, 2, 2])
    b = b and not lb.success(s)
    print(lb.success(s))
    print("\nrésultat correct" if b else "\nmauvais résultat")


def test3():
    lb = Labyrinth(5, 7,
                   [Position(0, 4), Position(0, 6)],
                   [Position(0, 3), Position(1, 3), Position(2, 3), Position(2, 2), Position(2, 4)])
    s = State([Position(0, 0), Position(3, 0)])
    lb.init_positions(s)
    b = True
    print(lb.free_pos)
    ans = [[False, True, True, False, True, True, True], [True, True, True, False, True, True, True],
           [True, True, False, False, False, True, True], [False, True, True, True, True, True, True],
           [True, True, True, True, True, True, True]]
    result = lb.free_pos == ans
    for i in range(5):
        for j in range(7):
            b = b and result[i, j]
    print("\n", "résultat correct" if b else "mauvais résultat", "\n")
    s = s.move([2, 2]).move([2, 1]).move([3, 4]).move([4, 0]).move([4, 0])
    lb.init_positions(s)
    b = True
    print(lb.free_pos)
    ans = [[False, False, False, False, True, True, True], [False, False, False, False, True, True, True],
           [False, False, False, False, False, True, True], [False, False, True, True, True, True, True],
           [True, True, True, True, True, True, True]]
    result = lb.free_pos == ans
    for i in range(5):
        for j in range(7):
            b = b and result[i, j]
    print("\n", "résultat correct" if b else "mauvais résultat", "\n")


def test4():
    b = True
    lb = Labyrinth(4, 4,
                   [Position(2, 3), Position(1, 2), Position(0, 0)],
                   [])
    s0 = State([Position(1, 1), Position(2, 1), Position(1, 3)])
    lb.init_positions(s0)
    possible1 = lb.possible_moves_snail(Position(1, 1), Position(2, 3), [Position(1, 2), Position(0, 0)])
    print(possible1)
    ver = True
    for i in possible1:
        if i not in [(1, Position(0, 1)), (4, Position(1, 0)), (0, Position(1, 1))]:
            ver = False
    b = b and len(possible1) == len([(1, Position(0, 1)), (4, Position(1, 0)), (0, Position(1, 1))]) and ver
    possible2 = lb.possible_moves_snail(Position(2, 1), Position(1, 2), [Position(2, 3), Position(0, 0)])
    print(possible2)
    ver = True
    for i in possible2:
        if i not in [(2, Position(2, 2)), (3, Position(3, 1)), (4, Position(2, 0)), (0, Position(2, 1))]:
            ver = False
    b = b and len(possible2) == len(
        [(2, Position(2, 2)), (3, Position(3, 1)), (4, Position(2, 0)), (0, Position(2, 1))]) and ver
    possible3 = lb.possible_moves_snail(Position(1, 3), Position(0, 0), [Position(2, 3), Position(1, 2)])
    print(possible3)
    ver = True
    for i in possible3:
        if i not in [(1, Position(0, 3)), (0, Position(1, 3))]:
            ver = False
    b = b and len(possible3) == len([(1, Position(0, 3)), (0, Position(1, 3))]) and ver
    lb = Labyrinth(5, 7,
                   [Position(0, 4), Position(0, 6)],
                   [Position(0, 3), Position(1, 3), Position(2, 3), Position(2, 2), Position(2, 4)])
    s = State([Position(0, 0), Position(3, 0)])
    s2 = State([Position(0, 1), Position(3, 1)])
    s3 = State([Position(3, 5), Position(4, 4), Position(1, 4)])
    l1 = len(lb.possible_moves(s))
    print(l1)
    b = b and l1 == 11
    l2 = len(lb.possible_moves(s2))
    print(l2)
    b = b and l2 == 19
    lb = Labyrinth(5, 7,
                   [Position(0, 4), Position(0, 6), Position(0, 2)],
                   [Position(0, 3), Position(1, 3), Position(2, 3), Position(2, 2), Position(2, 4)])
    l3 = len(lb.possible_moves(s3))
    print(l3)
    b = b and l3 == 35
    print("\n", "résultat correct" if b else "mauvais résultat", "\n")

def notreTest():
    lb = Labyrinth(4, 4,
                   [Position(0, 3), Position(3, 3)],
                   [])
    s0 = State([Position(0, 0), Position(3, 0)])
    s = lb.solve(s0)
    lb.print_solution(s)

def solve7():
    lb = Labyrinth(4, 4,
                   [Position(2, 3), Position(1, 2), Position(0, 0)],
                   [])
    s0 = State([Position(1, 1), Position(2, 1), Position(1, 3)])
    s = lb.solve(s0)
    lb.print_solution(s)


#     Si vous voulez visualiser les résultats, décommenter la ligne ci-dessous.
#     lb.print_labyrinth(s, show_all=True)

def solve9():
    lb = Labyrinth(5, 5,
                    [Position(3,4), Position(1,3)],
                    [])
    s0 = State([Position(2,0), Position(3,0)])
    s = lb.solve(s0)
    #lb.print_solution(s)
#     Si vous voulez visualiser les résultats, décommenter la ligne ci-dessous.
#     lb.print_labyrinth(s, show_all=True)

def solve12():
    lb = Labyrinth(5, 7,
                [Position(0,4), Position(0,6)],
                [Position(0,3), Position(1,3), Position(2,3), Position(2,2), Position(2,4)])
    s0 = State([Position(0,0), Position(3,0)])
    #s = %time lb.solve(s0)
    #lb.print_solution(s)
#     Si vous voulez visualiser les résultats, décommenter la ligne ci-dessous.
#     lb.print_labyrinth(s, show_all=True)

#test1()
#test2()
#test3()
# test4()

# notreTest()
solve7()
# solve9()
print("\n")
#solve12()
