$file = "../utils/kmeans.py"
$input_file = "../testcases/testcase01.txt"
$output_file = "../testcases/python_result01.txt"
python $file $input_file $output_file

# the residual arguments are passed to the main program

# if ($?) {
#     .\main $input_file $output_file
#     # delete the executable
#     Remove-Item .\main.exe
#     Remove-Item .\main.exp
#     Remove-Item .\main.lib
# }