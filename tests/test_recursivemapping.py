import unittest
from stabcodes.recursivemapping import RecursiveMapping


def testRecursiveMapping():
    empty = RecursiveMapping()
    assert len(empty) == 0

    a = RecursiveMapping([1, 2])
    a1 = RecursiveMapping([])
    empty["X"] = a
    empty["Z"] = a1

    assert len(empty) == 2
    assert len(a1) == 0

    del empty["Z"]
    assert len(empty) == 2
    del empty["X"][0]

    assert len(empty) == 1

    b = RecursiveMapping({"2": [1], "1": [2, 3]})
    c = RecursiveMapping({"1": {"1": [0], "2": [2]}, "2": {"1": [3], "2": [4]}})

    assert len(b) == 3
    assert len(c) == 4

    assert b["1"][0] == 2
    assert b[1] == 3

    assert len(c["1"]) == 2
    assert b[2] == 1
    assert c["2"][1] == 4
    assert 3 in c
    assert 1 not in c
    assert "1" not in c
    assert list(c) == [0, 2, 3, 4]

    b2 = []
    for _ in range(3):
        b2.append(b.pop())

    assert b2 == [1, 3, 2], b2
    assert len(b) == 0

    c2 = []
    for _ in range(3):
        c2.append(c.pop(0))

    assert c2 == [0, 2, 3]
    assert len(c) == 1

    d = RecursiveMapping({1: {1: [1], 2: [2]}, 2: {1: [1]}})
    e = RecursiveMapping({2: {1: [3], 2: [2]}, 3: {3: [3]}})
    f = RecursiveMapping()

    d.update(e)
    assert d[2][1][0] == 3
    assert len(d) == 5
    f.update(e)
    assert e == f
    f.clear()
    assert RecursiveMapping() == f

    # insert
    # index
    # count
    # get
    # append

    regular = RecursiveMapping([0, 1, 2, 3])

    surface = RecursiveMapping({"X": [0, 1, 2, 3],
                                "Z": [2, 3, 4, 5]})

    color = RecursiveMapping({"X": {"blue": [0, 1, 2, 3],
                                    "green": [2, 3, 4, 5],
                                    "red": [1, 2, 5, 6]},
                              "Z": {"blue": [0, 1, 2, 3],
                                    "green": [2, 3, 4, 5],
                                    "red": [1, 2, 5, 6]}})

    assert surface["X"] == regular
    assert color["Z"]["red"][0] == color[1]
    assert color["X"]["green"][0] == color[2]
    assert color["Z"]["red"][3] == color["X"]["red"][3]


if __name__ == "__main__":
    testRecursiveMapping()
