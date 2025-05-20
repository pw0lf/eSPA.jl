"""
    eSPA.jl

Implementations of eSPA, eSPA+ and GOAl.

# Exports:
- `eSPAdiscrete`
- `eSPAfuzzy`
- `eSPAplus`
- `GOAL`
- `eSPAhybrid`
- `fit!`
- `predict`
"""
module eSPA

include("utils.jl")
include("eSPAdiscrete.jl")
include("eSPAfuzzy.jl")
include("eSPAplus.jl")
include("GOAL.jl")
include("eSPAhybrid.jl")

export eSPAfuzzy, eSPAdiscrete, eSPAplus, GOAL, eSPAhybrid , fit!, predict

end # module eSPA
