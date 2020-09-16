from Position import Position
from State import State


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
    s1 = s1.move([3, 3]).move([4, 4])
    # s1 = s1.move([3,3]).move([4,4]).move([1,1])
    print(s1.venoms)
    # s2 = s0.move([3,3]).move([2,2]).move([1,1]).move([0,0]).move([4,4])
    s2 = s0.move([3, 3]).move([2, 2]).move([1, 1]).move([0, 0])
    print(s1 == s2)
    b = b and s1 == s2

    print("\nrésultat correct" if b else "\nmauvais résultat")


test1()
