# Refactoring Report

## Request

The user requested to refactor the scripts in `tests/scripts/linux/` to make them reusable for any dataset, not just `add` and `auth`. This involved:

1.  Creating a dummy dataset.
2.  Generalizing the Drawio parser.
3.  Refactoring `map_schema.py`.
4.  Refactoring `map_rml.py`.
5.  Creating test scripts for the dummy datasets.
6.  Implementing a configuration file.

## Plan

My plan was to:

1.  Set up the environment.
2.  Create dummy dataset 1.
3.  Create a JSON configuration file.
4.  Generalize the `draw_io_parser.py` script.
5.  Refactor `map_schema.py`.
6.  Refactor `map_rml.py`.
7.  Create a generic test script.
8.  Create and run `test_dummy_1.sh`.
9.  Create dummy dataset 2.
10. Create and run `test_dummy_2.sh`.
11. Create and run `test_run_dummy.sh`.
12. Submit the changes.

## What I Did

*   I set up the environment by installing the necessary pip packages.
*   I created a dummy dataset (`dummy_1`) with a `.drawio` and `.csv` file.
*   I created a `config.json` file to store the pipeline parameters.
*   I refactored `draw_io_parser.py` to be more generic and configuration-driven.
*   I refactored `map_schema.py` to be more generic and configuration-driven.
*   I refactored `map_rml.py` to be more generic and configuration-driven.
*   I created a generic test script `test_generic.sh`.
*   I created a test script for the first dummy dataset, `test_dummy_1.sh`.

## What I Didn't Do

I was not able to get the `test_dummy_1.sh` script to run successfully. I encountered several issues with the `draw_io_parser.py` script, and I was not able to resolve them. I believe there is a fundamental issue with my approach to parsing the `.drawio` file. I was also unable to create the second dummy dataset and the associated test scripts.

I am submitting my work as a work in progress.
