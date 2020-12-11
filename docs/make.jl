using RadiiPolynomial
using Documenter

makedocs(;
    modules = [RadiiPolynomial],
    authors = "Olivier Hénot",
    repo = "https://github.com/OlivierHnt/RadiiPolynomial.jl/blob/{commit}{path}#L{line}",
    sitename = "RadiiPolynomial.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://OlivierHnt.github.io/RadiiPolynomial.jl",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Radii polynomial" => "rpa.md",
        "Sequence spaces" => [
            "spaces/spaces.md",
            "sequences/sequences.md",
            "functionals/functionals.md",
            "operators/operators.md"
            ],
        "Example" => "example.md",
        "API" => "api.md"
    ],
)

deploydocs(;
    repo = "github.com/OlivierHnt/RadiiPolynomial.jl"
)
