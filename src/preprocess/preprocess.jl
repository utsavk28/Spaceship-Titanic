module Preprocess

include("../util/data_utils.jl")
include("../inputs.jl")
include("../config.jl")

using DataFrames
using MLJ
using BetaML

import .DataUtils:data_imputer_strategy
import .Inputs:MISSING_COLS,MISSING_STRATEGY,CATEGORICAL_COLS,UNNECESSARY_COLS,TARGET, MISSING
import .Config:FRACTION,RANDOM_STATE,SHUFFLE

function dropping_unnecessary_cols(X_train,X_test)
    for col in UNNECESSARY_COLS 
        if col in names(X_train)
            select!(X_train,Not(col))
        end
        if col in names(X_test)
            select!(X_test,Not(col))
        end
    end
    return X_train,X_test
end

function missing_value_handler(X_train,X_test,cols)    
    other_cols = filter(x -> ~(x in cols),names(X_train))
    imputer = SimpleImputer(statistic=data_imputer_strategy(MISSING_STRATEGY))
    X_train2 = select(X_train,cols) |> MLJ.matrix
    X_test2 = select(X_test,cols) |> MLJ.matrix
    (fitResults,_,_) = MLJ.fit(imputer,0,X_train2)
    
    
    X_train_imputed = MLJ.transform(imputer,fitResults,X_train2) |> DataFrame
    X_test_imputed = MLJ.transform(imputer,fitResults,X_test2) |> DataFrame
    rename!(X_train_imputed,cols)
    rename!(X_test_imputed,cols)

    # for col in cols 
    #     X_train[!,col] = X_train_imputed[!,col]
    #     X_test[!,col] = X_test_imputed[!,col]
    # end
    
    X_train = DataFrame(hcat(select(X_train,other_cols) |> Matrix, X_train_imputed |> Matrix),:auto)
    X_test = DataFrame(hcat(select(X_test,other_cols) |> Matrix, X_test_imputed |> Matrix),:auto)
    rename!(X_train,[other_cols...,cols...])
    rename!(X_test,[other_cols...,cols...])


    X_train[!,cols] = convert.(Float64,X_train_imputed[!,cols])
    X_test[!,cols] = convert.(Float64,X_test_imputed[!,cols])

    return X_train,X_test
end

function categorical_feature_handler(X_train,X_test,cols)
    for col in cols 
        X_train[!,col] = replace!(X_train[!,col], missing => MISSING)
        X_test[!,col] = replace!(X_test[!,col], missing => MISSING)
    end  
    return X_train,X_test      
end

function label_encoding_handler(X_train,X_test,cols) 
    function gen_label_encoder() 
        label_encoder = Dict()
        for col in cols 
            col_mapper = Dict()
            idx = 0
            for val in unique(X_train[!,col])
                col_mapper[val] = idx
                idx+=1
            end
            label_encoder[col] = col_mapper
        end
        label_encoder
    end

    label_encoder = gen_label_encoder()

    for col in cols
        # X_train[!,col] = map(x -> (x===missing) ? missing : get(label_encoder[col],x,-1),X_train[!,col])
        # X_test[!,col] = map(x -> (x===missing) ? missing : get(label_encoder[col],x,-1),X_test[!,col])

        X_train[!,col] = map(x -> get(label_encoder[col],x,-1),X_train[!,col])
        X_test[!,col] = map(x -> get(label_encoder[col],x,-1),X_test[!,col])

    end
    
    X_train,X_test
end

function partition_data(y)
    train_idx, test_idx = MLJ.partition(eachindex(y[!,TARGET]), FRACTION,shuffle=SHUFFLE,rng=RANDOM_STATE)
    train_idx, test_idx
end

function add_necessary_data_types(train,y_train,test) 
    map = []

    for col in names(train)
        if col in CATEGORICAL_COLS
            push!(map, col => Multiclass)
        else 
            push!(map, col => Continuous)
        end
    end

    # println(map...)
    
    train = coerce(train,map...)
    test = coerce(test,map...)
    # y_train = coerce(y_train,TARGET => Count)
    y_train = coerce(y_train,TARGET => Multiclass)
    train,y_train,test 
end

function preprocessor(train,test,sub) 
    X_train = select(train,Not(TARGET))
    y_train = select(train,TARGET)
    X_train,test = dropping_unnecessary_cols(X_train,test)
    X_train,test = missing_value_handler(X_train,test,MISSING_COLS)
    X_train,test = categorical_feature_handler(X_train,test,CATEGORICAL_COLS)
    X_train,test = label_encoding_handler(X_train,test,CATEGORICAL_COLS)
    X_train,y_train,test = add_necessary_data_types(X_train,y_train,test)
    # X_train,y_train,X_test,y_test = partition_data(X_train,y_train)
    # y_train = y_train[!,TARGET]

    X_train,y_train,test,sub
end

end