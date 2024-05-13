$file = "../utils/kmeans.cpp"
$input_file = "../testcases/testcase01.txt"
$output_file = "../testcases/cpu_result01.txt"
g++ $file -o cpu_main

# the residual arguments are passed to the main program
if ($?) {
    .\cpu_main $input_file $output_file
    # delete the executable
    Remove-Item .\cpu_main.exe
}