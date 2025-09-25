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
