using SSIMLoss
using Documenter

DocMeta.setdocmeta!(SSIMLoss, :DocTestSetup, :(using SSIMLoss); recursive=true)

makedocs(;
    modules=[SSIMLoss],
    authors="Nikola Janjusevic <nikola@nyu.edu>",
    repo="https://github.com/nikopj/SSIMLoss.jl/blob/{commit}{path}#{line}",
    sitename="SSIMLoss.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://nikopj.github.io/SSIMLoss.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/nikopj/SSIMLoss.jl",
    devbranch="main",
)
