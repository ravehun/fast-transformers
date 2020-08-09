import unittest


class SamplingUnitTest(unittest.TestCase):
    def testRolling(self):
        import pandas as pd
        def future(z, window):
            z = z.shift(-window)
            return z.rolling(window).max()

        z = range(10)
        z = pd.Series(z)
        print(future(z,3).fillna(0))


if __name__ == '__main__':
    unittest.main()
