module Inputs
export MISSING_COLS,MISSING_STRATEGY,CATEGORICAL_COLS

# Target Feature
TARGET = "Transported"

# Unnecessary Columns
UNNECESSARY_COLS = ["PassengerId","Name"]

# Handling Missing Value 
MISSING_STRATEGY = "median"
MISSING = "Z"
MISSING_COLS = ["Age", "FoodCourt", "ShoppingMall", "Spa", "VRDeck" ,"RoomService"]

# Handling Categorical Features
CATEGORICAL_COLS = ["HomePlanet", "CryoSleep","Cabin", "Destination" ,"VIP"]

end