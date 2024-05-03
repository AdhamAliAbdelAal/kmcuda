$file = "../utils/kmeans.cpp"
$input_file = "../testcases/testcase01.txt"
$output_file = "../testcases/c++_result01.txt"
g++ $file -o main

# the residual arguments are passed to the main program
if ($?) {
    .\main $input_file $output_file
    # delete the executable
    Remove-Item .\main.exe
}