module Predict

include("../inputs.jl")

using MLJ

import .Inputs:TARGET

function predict(model,test,sub)
    y_sub = MLJ.predict(model,test)
    sub[!,TARGET] = map(x -> x ? "True" : "False",pdf.(y_sub,true) .>= 0.5)
    sub
end

end