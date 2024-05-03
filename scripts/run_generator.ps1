$numberOfPoints = 100
$numberOfFeatures = 2
$numberOfClusters = 4
$maximumNumberOfIterations = 100
$testCaseFile = "../input/testCase01.txt"
$testResultFile = "../input/expected01.txt"
$pythonFile = '../utils/generator.py'

python $pythonFile $numberOfPoints $numberOfFeatures $numberOfClusters $maximumNumberOfIterations $testCaseFile $testResultFile