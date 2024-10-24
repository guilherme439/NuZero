import pytest
import subprocess

def test_new_training():
    input_file_path = "Input/new_training_test_input"

    with open(input_file_path, "r") as f:
        input_data = f.read()
    
    result = subprocess.run(
        ['python', 'Run.py', '--interactive'],
        input=input_data.encode(), 
        capture_output=True
    )
    
    assert result.returncode == 0, f"Script crashed with return code {result.returncode}"

