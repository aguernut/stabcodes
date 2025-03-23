import unittest
from stabcodes.pauli import Pauli


class TestPauliMethods(unittest.Test):
    def test_init(self):
        self.assertEqual()

        with self.assertRaises(ValueError):
            P = Pauli("P", 1)
        with self.assertRaises(ValueError):
            P = Pauli("P", -1)


if __name__ == "__main__":
    unittest.main()
