using Documenter, DocumenterCodeBlocks
using RadiiPolynomial
using CairoMakie
CairoMakie.activate!(type = "png", px_per_unit = 2)

DocMeta.setdocmeta!(RadiiPolynomial, :DocTestSetup, :(using RadiiPolynomial))

const PAGES = [
    "Home" => "index.md",
    "The radii polynomial approach" => "radii_polynomial_approach.md",
    "Getting started" => [
        "getting_started/first_validation.md",
        "getting_started/lorenz_equilibria.md",
        "getting_started/cubic_root_cont.md",
        "getting_started/logistic_equation.md"
    ],
    "Examples" => [
        "examples/index_examples.md",
        "Steady states" => [
            "examples/steady_states/cahn_hilliard.md",
            "examples/steady_states/nonlinear_diffusion.md"
        ],
        "Periodic orbits" => [
            "examples/periodic_orbits/non_autonomous_po.md",
            "examples/periodic_orbits/lorenz_po.md"
        ],
        "Continuation" => [
            "examples/continuation/cube_root_pa.md",
            "examples/continuation/cahn_hilliard_cont.md"
        ]
    ],
    "Manual" => [
        "manual/vector_spaces.md",
        "manual/sequences.md",
        "manual/linear_operators.md",
        "manual/norms.md",
        "manual/special_operators.md"
    ]
]

makedocs(;
    modules = [RadiiPolynomial],
    authors = "Olivier Hénot",
    sitename = "RadiiPolynomial.jl",
    format = Documenter.HTML(;
        assets = ["assets/radiipolynomial.css"],
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://olivierhnt.github.io/RadiiPolynomial.jl",
        collapselevel = 1,
        mathengine = KaTeX(Dict(
            :macros => Dict(
                "\\bydef" => "\\stackrel{\\tiny\\text{def}}{=}",
                "\\num" => "\\bar{u}",
                "\\exact" => "u_\\star"
            )
        ))
    ),
    pages = PAGES,
    plugins = [CodeBlocks()],
    checkdocs = :exports,
    draft = get(ENV, "RP_DOCS_DRAFT", "false") == "true",
    warnonly = [:missing_docs]
)

deploydocs(;
    repo = "github.com/OlivierHnt/RadiiPolynomial.jl",
    devbranch = "main",
    push_preview = true
)
