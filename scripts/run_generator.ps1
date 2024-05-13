$numberOfPoints = 4096
$numberOfFeatures = 128
$numberOfClusters = 16
$maximumNumberOfIterations = 10
$testCaseFile = "../testcases/testcase01.txt"
$testResultFile = "../testcases/expected01.txt"
$pythonFile = '../utils/generator.py'

python $pythonFile $numberOfPoints $numberOfFeatures $numberOfClusters $maximumNumberOfIterations $testCaseFile $testResultFile