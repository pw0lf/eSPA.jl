using Example
using Documenter

DocMeta.setdocmeta!(eSPA, :DocTestSetup, :(using eSPA); recursive=true)

makedocs(;
    modules=[eSPA],
    authors="Philipp Wolf <pwolf01@students.uni-mainz.de> and contributors",
    sitename="eSPA.jl",
    format=Documenter.HTML(;
        canonical="https://pw0lf.github.io/eSPA.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/pw0lf/eSPA.jl",
    devbranch="main",
)
