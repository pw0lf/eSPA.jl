module eSPA

include("utils.jl")
include("eSPAdiscrete.jl")
include("eSPAfuzzy.jl")
include("eSPAplus.jl")
include("GOAL.jl")

export eSPAfuzzy, eSPAdiscrete, eSPAplus, GOAL ,fit!, predict

end # module eSPA
