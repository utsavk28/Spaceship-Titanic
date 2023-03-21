module Model

include("../config.jl")

using Random
using Statistics
using MLJ
# using BetaML
using ScikitLearn
using MLJDecisionTreeInterface

import .Config:RANDOM_STATE

function model_preprocessing()
    Random.seed!(RANDOM_STATE)
end

function load_model(name)
    model = missing
    if name == "DecisionTree" 
        model = @load DecisionTreeClassifier pkg=DecisionTree
    elseif name == "RandomForest"
        model = @load RandomForestClassifier pkg=DecisionTree
    elseif name == "ExtraTrees"
        model = @load ExtraTreesClassifier pkg=ScikitLearn
    elseif name == "Logistic"
        model = @load LogisticClassifier pkg=ScikitLearn 
    elseif name == "KNeighbors"
        model = @load KNeighborsClassifier pkg=ScikitLearn 
    elseif name == "AdaBoost"
        model = @load AdaBoostClassifier pkg=ScikitLearn 
    elseif name == "GradientBoosting"
        model = @load GradientBoostingClassifier pkg=ScikitLearn 
    else 
        throw(DomainError(name,"Invalid Model Name"))
    end
    model()
end

function setup_model(name,X,y)
    model_preprocessing()
    model = load_model(name)
    model = machine(model,X,y)
    model
end

end