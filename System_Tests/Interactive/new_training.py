import pytest
import subprocess
import os

def test_new_training():
    current_directory_path = os.path.dirname(os.path.abspath(__file__))
    input_file_path = os.path.join(current_directory_path, "Input/new_training_test_input")

    with open(input_file_path, "r") as f:
        input_data = f.read()
    
    result = subprocess.run(
        ['python', 'Run.py', '--interactive'],
        input=input_data.encode(), 
        capture_output=True
    )
    
    assert result.returncode == 0, f"Script crashed with return code {result.returncode}"

