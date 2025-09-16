# Environment Setup Guide
This guide will help you set up the Python virtual environment needed to run all the Jupyter Notebook examples in this book.

## Prerequisites
### Python 3.10 Installation
**Windows:**
1. Download and install Python 3.10 from [python.org](https://www.python.org/downloads/windows)
2. During installation, make sure to check "Add Python to PATH"

**Linux:**
1. Download and install Python 3.10 via apt install.
    ```bash
    sudo apt update
    sudo apt install -y software-properties-common
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install -y python3.10
    sudo apt install python3.10-venv
    curl -sSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.10 get-pip.py
    ```
2. Verify your installation:
    ```bash
    python3 --version
    ```

## Setting Up the Virtual Environment
1. Create the virtual environment:
    ```bash
    python3 -m venv ./onnx_env
    ```
2. Activate the environment:

    **Windows:**
    ```cmd
    .\onnx_env\Scripts\activate
    ```
    **Linux/macOS:**
    ```bash
    source ./onnx_env/bin/activate
    ```
3. Install required packages:
    ```bash
    pip install jupyter==1.1.1 notebook==7.4.4 onnx==1.18.0 onnxruntime==1.22.0 onnx-simplifier==0.4.36
    ```

## Getting the Code and Running Jupyter
1. Clone this repository:
    ```bash
    git clone https://github.com/ava-orange-education/Ultimate-ONNX-for-Optimizing-Deep-Learning-Models.git
    cd Ultimate-ONNX-for-Optimizing-Deep-Learning-Models
    ```
2. Start Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
3. Access the notebooks:
    * The terminal will display a URL starting with http://localhost:8888
    * Copy this URL and paste it into your web browser
    * You'll now have access to all the chapter notebooks

## Next Steps
Once you have Jupyter running, navigate to the Chapter 2 notebook to begin following along with the book content.

**Note:** This environment will work for all chapters in the book. Keep your virtual environment activated whenever you're working with the notebooks.