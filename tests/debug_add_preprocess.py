import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import map_schema

if __name__ == '__main__':
    if not os.path.exists("gbad/mapping/source/preprocessed"):
        os.makedirs("gbad/mapping/source/preprocessed")
    map_schema.add_preprocess(
        "description_tailshuf_100.csv",
        "gbad/mapping/source/preprocessed/description_tailshuf_100.csv"
    )
