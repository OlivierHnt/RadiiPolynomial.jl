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
                "\\bydef" => "\\stackrel{\\tiny\\text{def}}{=}"
            )
        ))
    ),
    pages = [
        "Home" => "index.md",
        "State of the art" => "radii_polynomial_approach.md",
        "Sequence spaces" => [
            "sequence_spaces/vector_spaces.md",
            "sequence_spaces/sequences.md",
            "sequence_spaces/linear_operators.md",
            "sequence_spaces/norms.md",
            "sequence_spaces/special_operators.md"
        ],
        "Examples" => [
            "Finite-dimensional proofs" => [
                "examples/finite_dimensional_proofs/spiderweb.md",
                "examples/finite_dimensional_proofs/fhn_pseudo_arclength.md"
            ],
            "Infinite-dimensional proofs" => [
                "Ordinary differential equations (ODE)" => [
                    "examples/infinite_dimensional_proofs/ode/logistic_ivp.md",
                    "examples/infinite_dimensional_proofs/ode/lorenz_po.md"
                ],
                "Delay differential equations (DDE)" => [
                    "examples/infinite_dimensional_proofs/dde/ikeda_W_u.md"
                ]
            ]
        ]
    ],
    warnonly = true
)

deploydocs(;
    repo = "github.com/OlivierHnt/RadiiPolynomial.jl",
    devbranch = "main"
)
