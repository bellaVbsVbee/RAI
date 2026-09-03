#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Setting up the ACSIncome Fairness Project environment..."

# Check for python3
if ! command -v python3 &> /dev/null
then
    echo "python3 could not be found. Please install Python 3 and try again."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment '.venv'..."
python3 -m venv .venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Register the Jupyter kernel
echo "Installing Jupyter kernel..."
python -m ipykernel install --user --name=rai_fairness --display-name "RAI (Python 3)"

echo "--------------------------------------------------------"
echo "Setup complete! 🎉"
echo "To activate the environment in your terminal, run:"
echo "    source .venv/bin/activate"
echo "Or simply start Jupyter Notebook:"
echo "    jupyter notebook"
echo "And select the 'RAI (Python 3)' kernel for your notebooks."
echo "--------------------------------------------------------"
