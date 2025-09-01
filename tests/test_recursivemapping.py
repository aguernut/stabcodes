import unittest
from stabcodes.recursivemapping import RecursiveMapping

class TestRecursiveMapping(unittest.TestCase):
    def testRecursiveMapping1(self):
        empty = RecursiveMapping()
        self.assertEqual(len(empty), 0)

        a = RecursiveMapping([1, 2])
        a1 = RecursiveMapping([])
        empty["X"] = a
        empty["Z"] = a1

        self.assertEqual(len(empty), 2)
        self.assertEqual(len(a1), 0)

        del empty["Z"]
        self.assertEqual(len(empty), 2)
        del empty["X"][0]

        self.assertEqual(len(empty), 1)

    def testRecursiveMapping2(self):
        b = RecursiveMapping({"2": [1], "1": [2, 3]})
        c = RecursiveMapping({"1": {"1": [0], "2": [2]}, "2": {"1": [3], "2": [4]}})

        self.assertEqual(len(b), 3)
        self.assertEqual(len(c), 4)

        self.assertEqual(b["1"][0], 2)
        self.assertEqual(b[1], 3)

        self.assertEqual(len(c["1"]), 2)
        self.assertEqual(b[2], 1)
        self.assertEqual(c["2"][1], 4)
        self.assertTrue(3 in c)
        self.assertTrue(1 not in c)
        self.assertTrue("1" not in c)
        self.assertEqual(list(c), [0, 2, 3, 4])

        b2 = []
        for _ in range(3):
            b2.append(b.pop())

        self.assertEqual(b2, [1, 3, 2])
        self.assertEqual(len(b),  0)

        c2 = []
        for _ in range(3):
            c2.append(c.pop(0))

        self.assertEqual(c2, [0, 2, 3])
        self.assertEqual(len(c), 1)
        
    def testRecursiveMapping3(self):
        d = RecursiveMapping({1: {1: [1], 2: [2]}, 2: {1: [1]}})
        e = RecursiveMapping({2: {1: [3], 2: [2]}, 3: {3: [3]}})
        f = RecursiveMapping()

        d.update(e)
        self.assertEqual(d[2][1][0], 3)
        self.assertEqual(len(d), 5)
        f.update(e)
        self.assertEqual(e, f)
        f.clear()
        self.assertEqual(RecursiveMapping(), f)

        with self.assertRaises(TypeError):
            f.insert(0, 3)
            
        d.insert(0, 14)
        self.assertEqual(d[1][1], RecursiveMapping([14, 1]))
        d.insert(3, 2)
        self.assertEqual(d[1][2], RecursiveMapping([2, 2]))
        
        self.assertEqual(d.index(14), 0)
        self.assertEqual(d.index(2), 2)
        with self.assertRaises(ValueError):
            d.index(4)
            
        self.assertEqual(d.count(2), 3)
        self.assertEqual(d.count(23), 0)
        
        self.assertEqual(d.get(4, "Failback"), "Failback")
        self.assertEqual(d.get(3, "Failback"), RecursiveMapping({3: RecursiveMapping([3])}))
        self.assertEqual(d.get(2, "Failback"), RecursiveMapping({1: RecursiveMapping([3]), 2: RecursiveMapping([2])}))

        d.append(314)
        self.assertEqual(d[3][3], RecursiveMapping([3, 314]))
        d[1].append(159)
        self.assertEqual(d[1][2], RecursiveMapping([2, 2, 159]))

    def testRecursiveMapping4(self):
        regular = RecursiveMapping([0, 1, 2, 3])

        surface = RecursiveMapping({"X": [0, 1, 2, 3],
                                    "Z": [2, 3, 4, 5]})

        color = RecursiveMapping({"X": {"blue": [0, 1, 2, 3],
                                        "green": [2, 3, 4, 5],
                                        "red": [1, 2, 5, 6]},
                                "Z": {"blue": [0, 1, 2, 3],
                                        "green": [2, 3, 4, 5],
                                        "red": [1, 2, 5, 6]}})

        self.assertEqual(surface["X"], regular)
        self.assertEqual(color["Z"]["red"][0], color[1])
        self.assertEqual(color["X"]["green"][0], color[2])
        self.assertEqual(color["Z"]["red"][3], color["X"]["red"][3])