## Generic newton method

function newton(x₀, F_DF; tol::Real=1e-12, maxiter::Int=15, verbose::Bool=true)
    verbose && return _newton_verbose(x₀, F_DF, tol, maxiter)
    return _newton_silent(x₀, F_DF, tol, maxiter)
end

function _newton_silent(x₀::Number, F_DF, tol, maxiter)
    F, DF = F_DF(x₀)
    abs(F) ≤ tol && return x₀
    x = x₀
    for i ∈ 1:maxiter
        x -= DF \ F
        F, DF = F_DF(x)
        abs(F) ≤ tol && return x
    end
    return x
end

function _newton_silent(x₀, F_DF, tol, maxiter)
    F, DF = F_DF(x₀)
    norm(F) ≤ tol && return x₀
    x = copy(x₀)
    for i ∈ 1:maxiter
        x .-= ldiv!(lu!(DF), F)
        F, DF = F_DF(x)
        norm(F) ≤ tol && return x
    end
    return x
end

function _display_newton_infos(tol, maxiter)
    println(string("\n Newton's method: tol = ", tol, ", maxiter = ", maxiter))
    println("                       |f(x)|             |AF(x)|")
end

function _display_newton_iteration(i, nF, nAF)
    @printf("%11d %19.4e %19.4e\n", i, nF, nAF)
end

function _newton_verbose(x₀::Number, F_DF, tol, maxiter)
    _display_newton_infos(tol, maxiter)
    F, DF = F_DF(x₀)
    nF = abs(F)
    AF = DF \ F
    nAF = abs(AF)
    _display_newton_iteration(0, nF, nAF)
    nF ≤ tol && return x₀
    x = x₀
    for i ∈ 1:maxiter
        x -= AF
        F, DF = F_DF(x)
        nF = abs(F)
        AF = DF \ F
        nAF = abs(AF)
        _display_newton_iteration(i, nF, nAF)
        nF ≤ tol && return x
    end
    return x
end

function _newton_verbose(x₀, F_DF, tol, maxiter)
    _display_newton_infos(tol, maxiter)
    F, DF = F_DF(x₀)
    nF = norm(F)
    ldiv!(lu!(DF), F)
    nAF = norm(F)
    _display_newton_iteration(0, nF, nAF)
    nF ≤ tol && return x₀
    x = copy(x₀)
    for i ∈ 1:maxiter
        x .-= F
        F, DF = F_DF(x)
        nF = norm(F)
        ldiv!(lu!(DF), F)
        nAF = norm(F)
        _display_newton_iteration(i, nF, nAF)
        nF ≤ tol && return x
    end
    return x
end
