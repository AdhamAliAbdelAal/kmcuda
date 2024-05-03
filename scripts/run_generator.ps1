$numberOfPoints = 10000
$numberOfFeatures = 10
$numberOfClusters = 10
$maximumNumberOfIterations = 10000
$testCaseFile = "../testcases/testcase01.txt"
$testResultFile = "../testcases/expected01.txt"
$pythonFile = '../utils/generator.py'

python $pythonFile $numberOfPoints $numberOfFeatures $numberOfClusters $maximumNumberOfIterations $testCaseFile $testResultFile