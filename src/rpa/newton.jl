function newton(F_DF, x₀; tol::Real=1e-12, maxiter::Int=15, verbose::Bool=true)
    tol < 0 || maxiter < 0 && return throw(DomainError((tol, maxiter), "tolerance and maximum number of iterations must be positive"))
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
    _display_newton_nF(0, nF)
    if nF ≤ tol
        println()
        return x₀, true
    end
    x = x₀
    i = 1
    while i ≤ maxiter && isfinite(nF)
        AF = DF \ F
        nAF = abs(AF)
        _display_newton_nAF(nAF)
        x -= AF
        F, DF = F_DF(x)
        nF = abs(F)
        _display_newton_nF(i, nF)
        if nF ≤ tol
            println()
            return x, true
        end
        i += 1
    end
    println()
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
    _display_newton_nF(0, nF)
    if nF ≤ tol
        println()
        return x₀, true
    end
    x = copy(x₀)
    i = 1
    while i ≤ maxiter && isfinite(nF)
        AF = DF \ F
        nAF = norm(AF, Inf)
        _display_newton_nAF(nAF)
        x .-= AF
        F, DF = F_DF(x)
        nF = norm(F, Inf)
        _display_newton_nF(i, nF)
        if nF ≤ tol
            println()
            return x, true
        end
        i += 1
    end
    println()
    return x, false
end

#

function newton!(F_DF!, x₀, F, DF; tol::Real=1e-12, maxiter::Int=15, verbose::Bool=true)
    tol < 0 || maxiter < 0 && return throw(DomainError((tol, maxiter), "tolerance and maximum number of iterations must be positive"))
    verbose && return _newton_verbose!(F_DF!, x₀, F, DF, tol, maxiter)
    return _newton_silent!(F_DF!, x₀, F, DF, tol, maxiter)
end

function newton!(F_DF!, x₀; tol::Real=1e-12, maxiter::Int=15, verbose::Bool=true)
    tol < 0 || maxiter < 0 && return throw(DomainError((tol, maxiter), "tolerance and maximum number of iterations must be positive"))
    F = similar(x₀)
    n = length(x₀)
    DF = similar(x₀, n, n)
    verbose && return _newton_verbose!(F_DF!, x₀, F, DF, tol, maxiter)
    return _newton_silent!(F_DF!, x₀, F, DF, tol, maxiter)
end

function newton!(F_DF!, x₀::Sequence; tol::Real=1e-12, maxiter::Int=15, verbose::Bool=true)
    tol < 0 || maxiter < 0 && return throw(DomainError((tol, maxiter), "tolerance and maximum number of iterations must be positive"))
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
    _display_newton_nF(0, nF)
    if nF ≤ tol
        println()
        return x₀, true
    end
    i = 1
    while i ≤ maxiter && isfinite(nF)
        AF = DF \ F
        nAF = norm(AF, Inf)
        _display_newton_nAF(nAF)
        x₀ .-= AF
        F_DF!(F, DF, x₀)
        nF = norm(F, Inf)
        _display_newton_nF(i, nF)
        if nF ≤ tol
            println()
            return x₀, true
        end
        i += 1
    end
    println()
    return x₀, false
end

#

function _display_newton_info(tol, maxiter)
    println("Newton's method: Inf-norm, tol = ", tol, ", maxiter = ", maxiter)
    println("      iteration        |F(x)|")
    println("-------------------------------------")
end

_display_newton_nF(i, nF) = @printf("%11d %19.4e", i, nF)

_display_newton_nAF(nAF) = @printf("        |DF(x)\\F(x)| = %10.4e\n", nAF)
