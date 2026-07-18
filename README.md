## Environment Setup

### Step 1. Create a virtual environment

Run the command for your operating system in a new terminal:

```bash
# macOS or Linux
python3 -m venv .venv

# Windows
py -m venv .venv
```

If the command above is not available, replace `python3` or `py` with `python`.

### Step 2. Activate the virtual environment

Run the command for your operating system:

```bash
# macOS or Linux
source .venv/bin/activate
```

After activation, `(.venv)` should appear at the beginning of the terminal prompt.

### Step 3. Install the required packages

Install the python packages listed in `requirements.txt`. These packages are required to run the project:

```bash
python -m pip install -r requirements.txt
```