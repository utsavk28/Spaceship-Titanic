module Config
export BASE_PATH, TRAIN_PATH,TEST_PATH,SUB_PATH,FRACTION,RANDOM_STATE,SHUFFLE

# Necessary Paths
BASE_PATH = "../data"
TRAIN_PATH = "$BASE_PATH/train.csv"
TEST_PATH = "$BASE_PATH/test.csv"
SUB_PATH = "$BASE_PATH/sample_submission.csv"

# Data Partition/Splitting
FRACTION = 0.8
RANDOM_STATE = 42
SHUFFLE = true

end