$numberOfClusters = 10
$file1 = "../testcases/expected01.txt"
$file2 = "../testcases/cuda_result01.txt"
$pythonFile = '../utils/compare_labels.py'

python $pythonFile $file1 $file2 $numberOfClusters 