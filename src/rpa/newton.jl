function newton(F_DF, x₀; tol::Real=1e-12, maxiter::Int=15, verbose::Bool=true)
    tol < 0 || maxiter < 0 && return throw(DomainError)
    verbose && return _newton_verbose(F_DF, x₀, tol, maxiter)
    return _newton_silent(F_DF, x₀, tol, maxiter)
end

function _newton_silent(F_DF, x₀::Number, tol, maxiter)
    F, DF = F_DF(x₀)
    nF = abs(F)
    nF ≤ tol && return x₀, true
    x = x₀
    i = 1
    while i ≤ maxiter && isfinite(nF)
        x -= DF \ F
        F, DF = F_DF(x)
        nF = abs(F)
        nF ≤ tol && return x, true
        i += 1
    end
    return x, false
end

function _newton_verbose(F_DF, x₀::Number, tol, maxiter)
    _display_newton_info(tol, maxiter)
    F, DF = F_DF(x₀)
    nF = abs(F)
    AF = DF \ F
    nAF = abs(AF)
    _display_newton_iteration(0, nF, nAF)
    nF ≤ tol && return x₀, true
    x = x₀
    i = 1
    while i ≤ maxiter && isfinite(nF)
        x -= AF
        F, DF = F_DF(x)
        nF = abs(F)
        AF = DF \ F
        nAF = abs(AF)
        _display_newton_iteration(i, nF, nAF)
        nF ≤ tol && return x, true
        i += 1
    end
    return x, false
end

function _newton_silent(F_DF, x₀, tol, maxiter)
    F, DF = F_DF(x₀)
    nF = norm(F, Inf)
    nF ≤ tol && return x₀, true
    x = copy(x₀)
    i = 1
    while i ≤ maxiter && isfinite(nF)
        x .-= DF \ F
        F, DF = F_DF(x)
        nF = norm(F, Inf)
        nF ≤ tol && return x, true
        i += 1
    end
    return x, false
end

function _newton_verbose(F_DF, x₀, tol, maxiter)
    _display_newton_info(tol, maxiter)
    F, DF = F_DF(x₀)
    nF = norm(F, Inf)
    AF = DF \ F
    nAF = norm(AF, Inf)
    _display_newton_iteration(0, nF, nAF)
    nF ≤ tol && return x₀, true
    x = copy(x₀)
    i = 1
    while i ≤ maxiter && isfinite(nF)
        x .-= AF
        F, DF = F_DF(x)
        nF = norm(F, Inf)
        AF = DF \ F
        nAF = norm(AF, Inf)
        _display_newton_iteration(i, nF, nAF)
        nF ≤ tol && return x, true
        i += 1
    end
    return x, false
end

#

function newton!(F_DF!, x₀, F, DF; tol::Real=1e-12, maxiter::Int=15, verbose::Bool=true)
    tol < 0 || maxiter < 0 && return throw(DomainError)
    verbose && return _newton_verbose!(F_DF!, x₀, F, DF, tol, maxiter)
    return _newton_silent!(F_DF!, x₀, F, DF, tol, maxiter)
end

function newton!(F_DF!, x₀; tol::Real=1e-12, maxiter::Int=15, verbose::Bool=true)
    tol < 0 || maxiter < 0 && return throw(DomainError)
    F = similar(x₀)
    n = length(x₀)
    DF = similar(x₀, n, n)
    verbose && return _newton_verbose!(F_DF!, x₀, F, DF, tol, maxiter)
    return _newton_silent!(F_DF!, x₀, F, DF, tol, maxiter)
end

function newton!(F_DF!, x₀::Sequence; tol::Real=1e-12, maxiter::Int=15, verbose::Bool=true)
    tol < 0 || maxiter < 0 && return throw(DomainError)
    F = similar(x₀)
    s = space(x₀)
    n = length(x₀)
    DF = LinearOperator(s, s, similar(coefficients(x₀), n, n))
    verbose && return _newton_verbose!(F_DF!, x₀, F, DF, tol, maxiter)
    return _newton_silent!(F_DF!, x₀, F, DF, tol, maxiter)
end

function _newton_silent!(F_DF!, x₀, F, DF, tol, maxiter)
    F_DF!(F, DF, x₀)
    nF = norm(F, Inf)
    nF ≤ tol && return x₀, true
    i = 1
    while i ≤ maxiter && isfinite(nF)
        x₀ .-= DF \ F
        F_DF!(F, DF, x₀)
        nF = norm(F, Inf)
        nF ≤ tol && return x₀, true
        i += 1
    end
    return x₀, false
end

function _newton_verbose!(F_DF!, x₀, F, DF, tol, maxiter)
    _display_newton_info(tol, maxiter)
    F_DF!(F, DF, x₀)
    nF = norm(F, Inf)
    AF = DF \ F
    nAF = norm(AF, Inf)
    _display_newton_iteration(0, nF, nAF)
    nF ≤ tol && return x₀, true
    i = 1
    while i ≤ maxiter && isfinite(nF)
        x₀ .-= AF
        F_DF!(F, DF, x₀)
        nF = norm(F, Inf)
        AF = DF \ F
        nAF = norm(AF, Inf)
        _display_newton_iteration(i, nF, nAF)
        nF ≤ tol && return x₀, true
        i += 1
    end
    return x₀, false
end

#

function _display_newton_info(tol, maxiter)
    println("Newton's method: ∞-norm, tol = ", tol, ", maxiter = ", maxiter)
    println("      iteration        |F(x)|              |AF(x)|")
    println("---------------------------------------------------------")
end

_display_newton_iteration(i, nF, nAF) = @printf("%11d %19.4e %19.4e\n", i, nF, nAF)
