# Generic newton method

function newton(F_DF, x₀; tol::Real=1e-12, maxiter::Int=15, verbose::Bool=true)
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

function _newton_silent(F_DF, x₀, tol, maxiter)
    F, DF = F_DF(x₀)
    nF = norm(F)
    nF ≤ tol && return x₀, true
    x = copy(x₀)
    i = 1
    while i ≤ maxiter && isfinite(nF)
        x .-= ldiv!(lu!(DF), F)
        F, DF = F_DF(x)
        nF = norm(F)
        nF ≤ tol && return x, true
        i += 1
    end
    return x, false
end

function _newton_verbose(F_DF, x₀::Number, tol, maxiter)
    _display_newton_infos(tol, maxiter)
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

function _newton_verbose(F_DF, x₀, tol, maxiter)
    _display_newton_infos(tol, maxiter)
    F, DF = F_DF(x₀)
    nF = norm(F)
    ldiv!(lu!(DF), F)
    nAF = norm(F)
    _display_newton_iteration(0, nF, nAF)
    nF ≤ tol && return x₀, true
    x = copy(x₀)
    i = 1
    while i ≤ maxiter && isfinite(nF)
        x .-= F
        F, DF = F_DF(x)
        nF = norm(F)
        ldiv!(lu!(DF), F)
        nAF = norm(F)
        _display_newton_iteration(i, nF, nAF)
        nF ≤ tol && return x, true
        i += 1
    end
    return x, false
end

# in-place version

function newton!(F_DF!, x₀, F, DF; tol::Real=1e-12, maxiter::Int=15, verbose::Bool=true)
    verbose && return _newton_verbose!(F_DF!, x₀, F, DF, tol, maxiter)
    return _newton_silent!(F_DF!, x₀, F, DF, tol, maxiter)
end

function _newton_silent!(F_DF!, x₀, F, DF, tol, maxiter)
    F_DF!(F, DF, x₀)
    nF = norm(F)
    nF ≤ tol && return x₀, true
    i = 1
    while i ≤ maxiter && isfinite(nF)
        x₀ .-= ldiv!(lu!(DF), F)
        F_DF!(F, DF, x₀)
        nF = norm(F)
        nF ≤ tol && return x₀, true
        i += 1
    end
    return x₀, false
end

function _newton_verbose!(F_DF!, x₀, F, DF, tol, maxiter)
    _display_newton_infos(tol, maxiter)
    F_DF!(F, DF, x₀)
    nF = norm(F)
    ldiv!(lu!(DF), F)
    nAF = norm(F)
    _display_newton_iteration(0, nF, nAF)
    nF ≤ tol && return x₀, true
    i = 1
    while i ≤ maxiter && isfinite(nF)
        x₀ .-= F
        F_DF!(F, DF, x₀)
        nF = norm(F)
        ldiv!(lu!(DF), F)
        nAF = norm(F)
        _display_newton_iteration(i, nF, nAF)
        nF ≤ tol && return x₀, true
        i += 1
    end
    return x₀, false
end

#

function _display_newton_infos(tol, maxiter)
    println(string("\n Newton's method: tol = ", tol, ", maxiter = ", maxiter))
    println("      iteration        |F(x)|             |AF(x)|")
    println("-------------------------------------------------")
end

_display_newton_iteration(i, nF, nAF) = @printf("%11d %19.4e %19.4e\n", i, nF, nAF)
