$numberOfPoints = 10000
$numberOfFeatures = 256
$numberOfClusters = 32
$maximumNumberOfIterations = 1000
$testCaseFile = "../testcases/testcase01.txt"
$testResultFile = "../testcases/expected01.txt"
$pythonFile = '../utils/generator.py'

python $pythonFile $numberOfPoints $numberOfFeatures $numberOfClusters $maximumNumberOfIterations $testCaseFile $testResultFile