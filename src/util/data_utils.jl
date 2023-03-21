module DataUtils
    
using Statistics

function data_imputer_strategy(strategy)
    if (strategy == "mean") 
        return Statistics.mean
    elseif(strategy == "median")
        return Statistics.median
    elseif(strategy == "mode")
        return Statistics.mode
    end
end

end