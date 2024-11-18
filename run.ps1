# PowerShell script to run a Python file with specific arguments

# Get the current directory
$currentDir = Get-Location

# Specify the Python file name and args
$pythonFile = "main.py"
$n_epochs = 200
$lr = 0.001
$hidden_channels = 64

# Construct the command to run the Python
$command = "python $currentDir\$pythonFile --epochs $n_epochs --lr $lr --hidden_channels $hidden_channels"

# Execute the command
Invoke-Expression $command
