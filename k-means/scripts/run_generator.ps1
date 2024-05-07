$numberOfPoints = 10000
$numberOfFeatures = 2
$numberOfClusters = 4
$maximumNumberOfIterations = 10000
$testCaseFile = "../testcases/testcase01.txt"
$testResultFile = "../testcases/expected01.txt"
$pythonFile = '../utils/generator.py'

python $pythonFile $numberOfPoints $numberOfFeatures $numberOfClusters $maximumNumberOfIterations $testCaseFile $testResultFile