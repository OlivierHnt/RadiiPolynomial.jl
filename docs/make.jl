using Documenter, RadiiPolynomial

DocMeta.setdocmeta!(RadiiPolynomial, :DocTestSetup, :(using RadiiPolynomial))

makedocs(;
    modules = [RadiiPolynomial],
    authors = "Olivier Hénot",
    sitename = "RadiiPolynomial.jl",
    format = Documenter.HTML(;
        assets = ["assets/radiipolynomial.css"],
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://olivierhnt.github.io/RadiiPolynomial.jl",
        mathengine = KaTeX(Dict(
            :macros => Dict(
                "\\bydef" => "\\stackrel{\\tiny\\text{def}}{=}",
                "\\bx" => "\\bar{x}",
                "\\tx" => "x^\\star"
            )
        ))
    ),
    pages = [
        "Home" => "index.md",
        "State of the art" => "radii_polynomial_approach.md",
        "Sequence spaces" => [
            "manual/vector_spaces.md",
            "manual/sequences.md",
            "manual/linear_operators.md",
            "manual/norms.md",
            "manual/special_operators.md"
        ],
        "Examples" => [
            "Continuation" => [
                "examples/continuation/cubic_root_cont.md",
                "examples/continuation/cubic_root_pa.md"
            ],
            "Cauchy problems (IVPs)" => [
                "examples/ivp/logistic_ivp.md"
            ],
            "Periodic orbits" => [
                "examples/periodic_orbits/non_autonomous_po.md",
                "examples/periodic_orbits/lorenz_po.md"
            ],
            "Steady states" => [
                "examples/steady_states/nonlinear_diffusion.md"
            ]
        ]
    ],
    warnonly = true
)

deploydocs(;
    repo = "github.com/OlivierHnt/RadiiPolynomial.jl",
    devbranch = "main",
    push_preview = true
)
