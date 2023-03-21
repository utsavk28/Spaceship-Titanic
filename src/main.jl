module SpaceshipTitanic

include("dataloader/dataloader.jl")
include("evaluate/evaluate.jl")

import .Dataloader: loadData
import .Evaluate: evaluate

X_train,y_train,test,sub = loadData(true)
train_results,test_results = evaluate(X_train,y_train,test,sub)
println(train_results,test_results)

end