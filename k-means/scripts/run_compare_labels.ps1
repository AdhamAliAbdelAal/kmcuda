$numberOfClusters = 4
$file1 = "../testcases/cpu_result01.txt"
$file2 = "../testcases/kmeans_hamerly.txt"
$pythonFile = '../utils/compare_labels.py'

python $pythonFile $file1 $file2 $numberOfClusters 

