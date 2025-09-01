import unittest
import doctest
import stabcodes.measure_clock
import stabcodes.pauli
import stabcodes.recursivemapping
import stabcodes.stabgen
import stabcodes.stabilizer_code
import stabcodes.stim_experiment
import stabcodes.tools

def test_doctests():
    tests = unittest.TestSuite()
    tests.addTests(doctest.DocTestSuite(stabcodes.measure_clock, optionflags=doctest.ELLIPSIS))
    tests.addTests(doctest.DocTestSuite(stabcodes.pauli, optionflags=doctest.ELLIPSIS))
    tests.addTests(doctest.DocTestSuite(stabcodes.recursivemapping, optionflags=doctest.ELLIPSIS))
    tests.addTests(doctest.DocTestSuite(stabcodes.stabgen, optionflags=doctest.ELLIPSIS))
    tests.addTests(doctest.DocTestSuite(stabcodes.stabilizer_code, optionflags=doctest.ELLIPSIS))
    tests.addTests(doctest.DocTestSuite(stabcodes.stim_experiment, optionflags=doctest.ELLIPSIS))
    tests.addTests(doctest.DocTestSuite(stabcodes.tools, optionflags=doctest.ELLIPSIS))
    return tests

if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(test_doctests())