$file = "../cuda/kmeans_without_icd.cu"
$input_file = "../testcases/testcase01.txt"
$output_file = "../testcases/cuda_kmeans_without_icd_result01.txt"
# $profiling = '../profiling/profiling.txt'
nvcc $file -o main

# the residual arguments are passed to the main program

if ($?) {
    .\main $input_file $output_file
    # .\main $input_file $output_file
    # delete the executable
    Remove-Item .\main.exe
    Remove-Item .\main.exp
    Remove-Item .\main.lib
}