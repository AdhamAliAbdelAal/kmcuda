$file = "../cuda/kmeans_histogram.cu"
$input_file = "../testcases/testcase01.txt"
$output_file = "../testcases/cuda_histogram_result01.txt"
# $profiling = '../profiling/profiling.txt'
nvcc $file -o main

# the residual arguments are passed to the main program

if ($?) {
    nvprof .\main $input_file $output_file
    # delete the executable
    Remove-Item .\main.exe
    Remove-Item .\main.exp
    Remove-Item .\main.lib
}