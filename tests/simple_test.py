import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import map_schema

class SimpleTest(unittest.TestCase):
    def test_auth_schema(self):
        print("Running simple test for auth schema")
        #This is needed to avoid FileNotFoundError
        if not os.path.exists("gbad/mapping/source/preprocessed"):
            os.makedirs("gbad/mapping/source/preprocessed")
        map_schema.__init__('auth', 'authority_tailshuf_100.csv')
        print("Finished simple test for auth schema")

if __name__ == '__main__':
    unittest.main()
