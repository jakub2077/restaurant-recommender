# PowerShell script to run a Python file

# Get the current directory
$currentDir = Get-Location

# Specify the Python file name
$pythonFile = "main.py"

# Construct the command to run the Python 
$command = "python $currentDir\$pythonFile"

# Execute the command
Invoke-Expression $command
