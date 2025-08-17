abstract type ConvergenceCriterion end

struct ResidualCriterion <: ConvergenceCriterion end
(res::ResidualCriterion)(nF, nAF, tol, ϵ) = nF ≤ max(tol, sqrt(ϵ)*(1+nF))
Base.show(io::IO, ::ResidualCriterion) = print(io, "|F(x)| ≤ max(tol, √ϵ*(1+|F(x)|)")

struct UpdateCriterion <: ConvergenceCriterion end
(update::UpdateCriterion)(nF, nAF, tol, ϵ) = nAF ≤ max(tol, sqrt(ϵ)*(1+nAF))
Base.show(io::IO, ::UpdateCriterion) = print(io, "|DF(x)\\F(x)| ≤ max(tol, √ϵ*(1+|DF(x)\\F(x)|)")

struct CombinedCriterion <: ConvergenceCriterion end
(comb::CombinedCriterion)(nF, nAF, tol, ϵ) = ResidualCriterion()(nF, nAF, tol, ϵ) & UpdateCriterion()(nF, nAF, tol, ϵ)
Base.show(io::IO, ::CombinedCriterion) = print(io, string(ResidualCriterion(), " && ", UpdateCriterion()))

function newton(F_DF, x0; tol::Real = 1e-12, maxiter::Int = 15, convergence_criterion::ConvergenceCriterion = ResidualCriterion(), verbose::Bool = false)
    F, DF = F_DF(x0)
    x = copy(x0)
    return newton!(x, F, DF; tol = tol, maxiter = maxiter, convergence_criterion = convergence_criterion, verbose = verbose) do F, DF, x
        F_, DF_ = F_DF(x)
        F = _copy_maybeinplace!(F, F_)
        DF = _copy_maybeinplace!(DF, DF_)
        return F, DF
    end
end

_copy_maybeinplace!(x::Number, y) = x = y
_copy_maybeinplace!(x, y) = x .= y

newton!(F_DF!, x0; tol::Real = 1e-12, maxiter::Int = 15, convergence_criterion::ConvergenceCriterion = ResidualCriterion(), verbose::Bool = false) =
    newton!(F_DF!, x0, similar(x0), _similar_linop(x0); tol = tol, maxiter = maxiter, convergence_criterion = convergence_criterion, verbose = verbose)

_similar_linop(x) = similar(x, length(x), length(x))
_similar_linop(x::Sequence) = LinearOperator(space(x), space(x), _similar_linop(coefficients(x)))

function newton!(F_DF!, x, F, DF; tol::Real = 1e-12, maxiter::Int = 15, convergence_criterion::ConvergenceCriterion = ResidualCriterion(), verbose::Bool = false)
    (tol < 0) | (maxiter < 0) && return throw(DomainError((tol, maxiter), "tolerance and maximum number of iterations must be positive"))
    ϵ = eps(real(eltype(x)))

    verbose && _print_header(tol, ϵ, maxiter, convergence_criterion)

    t = time()

    F, DF = F_DF!(F, DF, x)
    nF = norm(F, Inf)
    AF = DF \ F
    nAF = norm(AF, Inf)
    verbose && _print_iter!(t, maxiter, 0, nF, nAF)
    convergence_criterion(nF, nAF, tol, ϵ) && return x, true

    i = 1
    while i ≤ maxiter && isfinite(nF)
        x = _sub_maybeinplace!(x, AF)
        F, DF = F_DF!(F, DF, x)
        nF = norm(F, Inf)
        AF = DF \ F
        nAF = norm(AF, Inf)
        verbose && _print_iter!(t, maxiter, i, nF, nAF)
        convergence_criterion(nF, nAF, tol, ϵ) && return x, true
        i += 1
    end

    return x, false
end

_sub_maybeinplace!(x::Number, y) = x -= y
_sub_maybeinplace!(x, y) = x .-= y

function _print_header(tol, ϵ, maxiter, convergence_criterion)
    println("Newton's method: Inf-norm, tol = $tol, ϵ = $ϵ, maxiter = $maxiter, convergence criterion $convergence_criterion")
    @printf("%s   %s   %s   %s \n",
            center_text("Iteration", 12),
            center_text("|F(x)|", 12),
            center_text("|DF(x)\\F(x)|", 12),
            center_text("ETA (s)", 12))
    println(repeat('-', 12+12+12+12 + 3*3 + 2))
end

center_text(text, width) = lpad(rpad(text, div(width + length(text), 2)), width)

function _print_iter!(t, maxiter, i, nF, nAF)
    color_norm(x) = ifelse(x < 1e-8, "\e[32m", ifelse(x < 1e-4, "\e[33m", "\e[31m")) # green/yellow/red
    reset_color = "\e[0m"

    str_nF  = string(color_norm(nF), @sprintf("%.4e", nF), reset_color)
    str_nAF = string(color_norm(nAF), @sprintf("%.4e", nAF), reset_color)

    elapsed = time() - t
    eta = i > 0 ? elapsed / i * (maxiter - i) : 0.0
    eta_str = @sprintf("%.1e", eta)

    @printf("%s |  %s  |  %s  | %s\n",
        center_text(string(i), 12),
        center_text(str_nF, 12),
        center_text(str_nAF, 12),
        center_text(eta_str, 12))
    flush(stdout)
end
