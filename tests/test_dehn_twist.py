import unittest
from copy import deepcopy
from stabcodes.stabilizer_code import SurfaceCode
from stabcodes.pauli import Stabilizer2D, PauliOperator, X, Z

class TestDehnTwists(unittest.TestCase):
    def test_dehn_twist1(self):
        c = SurfaceCode.toric_code(6, 4)
        initial_logical = deepcopy(c.logical_operators)
        c.dehn_twist([0, 6, 12, 18], 24)
        c.check()
        target = {'X': [Stabilizer2D([X(30), X(6), X(25), X(0)], 48),
                        Stabilizer2D([X(1), X(25), X(7), X(26)], 48),
                        Stabilizer2D([X(2), X(26), X(8), X(27)], 48),
                        Stabilizer2D([X(3), X(27), X(9), X(28)], 48),
                        Stabilizer2D([X(4), X(28), X(10), X(29)], 48),
                        Stabilizer2D([X(5), X(29), X(11), X(24)], 48),
                        Stabilizer2D([X(36), X(12), X(31), X(6)], 48),
                        Stabilizer2D([X(7), X(31), X(13), X(32)], 48),
                        Stabilizer2D([X(8), X(32), X(14), X(33)], 48),
                        Stabilizer2D([X(9), X(33), X(15), X(34)], 48),
                        Stabilizer2D([X(10), X(34), X(16), X(35)], 48),
                        Stabilizer2D([X(11), X(35), X(17), X(30)], 48),
                        Stabilizer2D([X(42), X(18), X(37), X(12)], 48),
                        Stabilizer2D([X(13), X(37), X(19), X(38)], 48),
                        Stabilizer2D([X(14), X(38), X(20), X(39)], 48),
                        Stabilizer2D([X(15), X(39), X(21), X(40)], 48),
                        Stabilizer2D([X(16), X(40), X(22), X(41)], 48),
                        Stabilizer2D([X(17), X(41), X(23), X(36)], 48),
                        Stabilizer2D([X(24), X(0), X(43), X(18)], 48),
                        Stabilizer2D([X(19), X(43), X(1), X(44)], 48),
                        Stabilizer2D([X(20), X(44), X(2), X(45)], 48),
                        Stabilizer2D([X(21), X(45), X(3), X(46)], 48),
                        Stabilizer2D([X(22), X(46), X(4), X(47)], 48),
                        Stabilizer2D([X(23), X(47), X(5), X(42)], 48)],
                'Z': [Stabilizer2D([Z(0), Z(24), Z(11), Z(30)], 48),
                        Stabilizer2D([Z(25), Z(6), Z(31), Z(7)], 48),
                        Stabilizer2D([Z(26), Z(7), Z(32), Z(8)], 48),
                        Stabilizer2D([Z(27), Z(8), Z(33), Z(9)], 48),
                        Stabilizer2D([Z(28), Z(9), Z(34), Z(10)], 48),
                        Stabilizer2D([Z(29), Z(10), Z(35), Z(11)], 48),
                        Stabilizer2D([Z(6), Z(30), Z(17), Z(36)], 48),
                        Stabilizer2D([Z(31), Z(12), Z(37), Z(13)], 48),
                        Stabilizer2D([Z(32), Z(13), Z(38), Z(14)], 48),
                        Stabilizer2D([Z(33), Z(14), Z(39), Z(15)], 48),
                        Stabilizer2D([Z(34), Z(15), Z(40), Z(16)], 48),
                        Stabilizer2D([Z(35), Z(16), Z(41), Z(17)], 48),
                        Stabilizer2D([Z(12), Z(36), Z(23), Z(42)], 48),
                        Stabilizer2D([Z(37), Z(18), Z(43), Z(19)], 48),
                        Stabilizer2D([Z(38), Z(19), Z(44), Z(20)], 48),
                        Stabilizer2D([Z(39), Z(20), Z(45), Z(21)], 48),
                        Stabilizer2D([Z(40), Z(21), Z(46), Z(22)], 48),
                        Stabilizer2D([Z(41), Z(22), Z(47), Z(23)], 48),
                        Stabilizer2D([Z(18), Z(42), Z(5), Z(24)], 48),
                        Stabilizer2D([Z(43), Z(0), Z(25), Z(1)], 48),
                        Stabilizer2D([Z(44), Z(1), Z(26), Z(2)], 48),
                        Stabilizer2D([Z(45), Z(2), Z(27), Z(3)], 48),
                        Stabilizer2D([Z(46), Z(3), Z(28), Z(4)], 48),
                        Stabilizer2D([Z(47), Z(4), Z(29), Z(5)], 48)]}

        target_logical = {'X': [PauliOperator([X(0), X(1), X(2), X(3), X(4), X(5), X(24)], 48),
                                PauliOperator([X(24), X(30), X(36), X(42)], 48)],
                        'Z': [PauliOperator([Z(0), Z(6), Z(12), Z(18)], 48),
                                PauliOperator([Z(0), Z(24), Z(25), Z(26), Z(27), Z(28), Z(29)], 48)]}

        for sx in c.stabilizers["X"]:
            if sx in target["X"]:
                break
        else:
            raise ValueError("Unexpected stabilizers found")

        for sz in c.stabilizers["Z"]:
            if sz in target["Z"]:
                break
        else:
            raise ValueError("Unexpected stabilizers found")

        for log in c.logical_operators["X"]:
            if log in target_logical["X"]:
                break
        else:
            raise ValueError("Unexpected stabilizers found")

        for log in c.logical_operators["Z"]:
            if log in target_logical["Z"]:
                break
        else:
            raise ValueError("Unexpected stabilizers found")

        c.dehn_twist([0, 6, 12, 18], 24)
        c.check()
        c.dehn_twist([0, 6, 12, 18], 24)
        c.check()
        c.dehn_twist([0, 6, 12, 18], 24)
        c.check()

        self.assertEqual(initial_logical["X"][0] * initial_logical["X"][1], c.logical_operators["X"][0])
        self.assertEqual(initial_logical["Z"][0] * initial_logical["Z"][1], c.logical_operators["Z"][1])

        c = SurfaceCode.toric_code(6, 4)
        c.dehn_twist([24, 25, 26, 27, 28, 29], 0)
        c.check()
        c.dehn_twist([24, 25, 26, 27, 28, 29], 0)
        c.check()
        c.dehn_twist([24, 25, 26, 27, 28, 29], 0)
        c.check()
        c.dehn_twist([24, 25, 26, 27, 28, 29], 0)
        c.check()
        c.dehn_twist([24, 25, 26, 27, 28, 29], 0)
        c.check()
        c.dehn_twist([24, 25, 26, 27, 28, 29], 0)
        c.check()

        self.assertEqual(initial_logical["X"][0] * initial_logical["X"][1], c.logical_operators["X"][1])
        self.assertEqual(initial_logical["Z"][0] * initial_logical["Z"][1], c.logical_operators["Z"][0])


    def test_dehn_twist2(self):
        c = SurfaceCode.hex_code(4, 2)
        initial_logical = deepcopy(c.logical_operators)


        c.dehn_twist(list(range(16, 20)), 0)
        c.check()
        c.dehn_twist(list(range(16, 20)), 0)
        c.check()
        c.dehn_twist(list(range(16, 20)), 0)
        c.check()
        c.dehn_twist(list(range(16, 20)), 0)
        c.check()
        self.assertEqual(initial_logical["X"][0] * initial_logical["X"][1], c.logical_operators["X"][1])
        self.assertEqual(initial_logical["Z"][0] * initial_logical["Z"][1], c.logical_operators["Z"][0])

        c = SurfaceCode.hex_code(4, 2)
        c.dehn_twist([7, 15], 0)
        c.check()
        c.dehn_twist([7, 15], 0)

        self.assertEqual(initial_logical["X"][0] * initial_logical["X"][1], c.logical_operators["X"][0])
        self.assertEqual(initial_logical["Z"][0] * initial_logical["Z"][1], c.logical_operators["Z"][1])
        c.check()


    def test_dehn_twist3(self):
        c = SurfaceCode.toric_code(6, 4)
        initial_logical = deepcopy(c.logical_operators)

        c.dehn_twist([3, 28, 29, 11, 17, 23, 47, 46], 35)
        c.check()
        c.dehn_twist([3, 28, 29, 11, 17, 23, 47, 46], 35)
        c.check()
        c.dehn_twist([3, 28, 29, 11, 17, 23, 47, 46], 35)
        c.check()
        c.dehn_twist([3, 28, 29, 11, 17, 23, 47, 46], 35)
        c.check()
        c.dehn_twist([3, 28, 29, 11, 17, 23, 47, 46], 35)
        c.check()
        c.dehn_twist([3, 28, 29, 11, 17, 23, 47, 46], 35)
        c.check()
        c.dehn_twist([3, 28, 29, 11, 17, 23, 47, 46], 35)
        c.check()
        c.dehn_twist([3, 28, 29, 11, 17, 23, 47, 46], 35)
        c.check()

        self.assertEqual(initial_logical["Z"][1] * PauliOperator([Z(3), Z(28), Z(29), Z(11), Z(17), Z(23), Z(47), Z(46)], 48), c.logical_operators["Z"][1])
        self.assertEqual(initial_logical["X"][0] * PauliOperator([X(27), X(9), X(10), X(35), X(41), X(22), X(21), X(45)], 48), c.logical_operators["X"][0])

if __name__ == "__main__":
    unittest.main()