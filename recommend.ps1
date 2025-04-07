# PowerShell script to run a Python file

# Get the current directory
$currentDir = Get-Location

# Specify the Python file name
$pythonFile = "recommend.py"

# params
$device = "cuda:0"
$user_id = 0

# Construct the command to run the Python 
$command = "python $currentDir\$pythonFile --device $device --user_id $user_id"

# Execute the command
Invoke-Expression $command
