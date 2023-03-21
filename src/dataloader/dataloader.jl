module Dataloader 
export loadData

using CSV
using DataFrames

include("../config.jl")
import .Config:TRAIN_PATH, TEST_PATH, SUB_PATH

include("../preprocess/preprocess.jl")
import .Preprocess:preprocessor

function loadData(preprocess=false)
    train = CSV.File(TRAIN_PATH) |> DataFrame
    test = CSV.File(TEST_PATH) |> DataFrame
    sub = CSV.File(SUB_PATH) |> DataFrame

    if preprocess 
        X_train,y_train,test,sub = preprocessor(train,test,sub)
    else
        X_train = train
        y_train = missing
    end

    X_train,y_train,test,sub
end



end