module Evaluate

include("./predict.jl")
include("../config.jl")
include("../inputs.jl")
include("../model/model.jl")
include("../preprocess/preprocess.jl")

using Random
using Statistics
using CSV
using MLJ

import .Model:setup_model
import .Inputs:TARGET
import .Preprocess:partition_data
import .Config:RANDOM_STATE
import .Predict:predict

function evaluate(X_train,y_train,test,sub)
    train_results = Dict()
    test_results = Dict()
    
    all_models = [
        "DecisionTree",
        "RandomForest",
        # "ExtraTrees",
        # "Logistic",
        # "KNeighbors",
        # "AdaBoost",
        # "GradientBoosting"
    ]

    train_idx, test_idx = partition_data(y_train)
    y_train = y_train[!,TARGET]

    for model_name in all_models 
        model = setup_model(model_name,X_train,y_train)
        eval_results=evaluate!(model,rows=train_idx, resampling=CV(nfolds=10,shuffle=true,rng=RANDOM_STATE), measures=[MLJ.accuracy],operation=predict_mode)
        test_results[model_name] = MLJ.accuracy(predict_mode(model, rows=test_idx), y_train[test_idx])
        train_results[model_name] = MLJ.accuracy(predict_mode(model, rows=train_idx), y_train[train_idx])
        println("$model_name 10 Folds :\n $(eval_results.per_fold)\n\n\n")
        sub = predict(model,test,sub)
        CSV.write("../output/$model_name.csv",sub)
    end

    train_results,test_results
end

end