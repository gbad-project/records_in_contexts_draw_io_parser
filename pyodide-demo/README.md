# Pyodide Demo

This project demonstrates how to use Pyodide to run Python code in a React application.

## Prerequisites

This project requires `bun`, `volta`, and `node`. The following tools are used to manage the environment and dependencies.

*   **Bun**: A fast JavaScript all-in-one toolkit.
*   **Volta**: A JavaScript tool manager.
*   **Node.js**: A JavaScript runtime.

You will also need Python and `pip` to install some python dependencies.

## Setup

1.  **Install Prerequisites**

    The `pyodide-react/README.md` file in the parent directory contains instructions for installing `bun` and `volta`. You can also install them from their official websites:

    *   [Bun](https://bun.sh/)
    *   [Volta](https://volta.sh/)

    Once `bun` and `volta` are installed, you can install `node` by running:

    ```bash
    volta install node
    ```

    You will also need to install some python dependencies:

    ```bash
    pip install -r ../next/requirements.dev.txt
    ```

    And playwright dependencies:

    ```bash
    playwright install && playwright install-deps
    ```

2.  **Install Project Dependencies**

    Navigate to the `pyodide-demo` directory and install the project dependencies using `bun`:

    ```bash
    cd pyodide-demo
    bun install
    ```

3.  **Download Pyodide**

    The project requires Pyodide to be downloaded. The `download_pyodide.sh` script will do this for you.

    ```bash
    ./download_pyodide.sh
    ```

## Running the Demo

An end-to-end test suite is provided to demonstrate the functionality. The `run_demo.sh` script will run the tests and create a log file.

To run the demo, execute the following command from the `pyodide-demo` directory:

```bash
./run_demo.sh
```

This will:
1.  Print system information.
2.  Run the playwright tests.
3.  Create a `run_demo.log` file with the test output.

After running the script, you can inspect the `run_demo.log` file to see the test results. A successful run will show that 1 test has passed.
