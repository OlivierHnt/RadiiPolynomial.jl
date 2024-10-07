function newton(F_DF::Function, x₀::Number; tol::Real=1e-12, maxiter::Int=15, verbose::Bool=true)
    (tol < 0) | (maxiter < 0) && return throw(DomainError((tol, maxiter), "tolerance and maximum number of iterations must be positive"))
    verbose && _display_newton_info(tol, maxiter)
    F, DF = F_DF(x₀)
    nF = abs(F)
    verbose && _display_newton_nF(0, nF)
    if nF ≤ tol
        verbose && println()
        return x₀, true
    end
    x = x₀
    i = 1
    while i ≤ maxiter && isfinite(nF)
        AF = DF \ F
        verbose && _display_newton_nAF(abs(AF))
        x -= AF
        F, DF = F_DF(x)
        nF = abs(F)
        verbose && _display_newton_nF(i, nF)
        if nF ≤ tol
            verbose && println()
            return x, true
        end
        i += 1
    end
    verbose && println()
    return x, false
end

function newton(F_DF::Function, x₀::Union{AbstractVector,Sequence}; tol::Real=1e-12, maxiter::Int=15, verbose::Bool=true)
    (tol < 0) | (maxiter < 0) && return throw(DomainError((tol, maxiter), "tolerance and maximum number of iterations must be positive"))
    verbose && _display_newton_info(tol, maxiter)
    F, DF = F_DF(x₀)
    nF = norm(F, Inf)
    verbose && _display_newton_nF(0, nF)
    if nF ≤ tol
        verbose && println()
        return x₀, true
    end
    x = copy(x₀)
    i = 1
    while i ≤ maxiter && isfinite(nF)
        AF = DF \ F
        verbose && _display_newton_nAF(norm(AF, Inf))
        x .-= AF
        F, DF = F_DF(x)
        nF = norm(F, Inf)
        verbose && _display_newton_nF(i, nF)
        if nF ≤ tol
            verbose && println()
            return x, true
        end
        i += 1
    end
    verbose && println()
    return x, false
end

#

function newton!(F_DF!::Function, x₀::AbstractVector, F::AbstractVector=zeros(eltype(x₀), length(x₀)), DF::AbstractMatrix=zeros(eltype(x₀), length(x₀), length(x₀)); tol::Real=1e-12, maxiter::Int=15, verbose::Bool=true)
    (tol < 0) | (maxiter < 0) && return throw(DomainError((tol, maxiter), "tolerance and maximum number of iterations must be positive"))
    return _newton!(F_DF!, x₀, F, DF, tol, maxiter, verbose)
end

function newton!(F_DF!::Function, x₀::Sequence, F::Sequence=zeros(eltype(x₀), space(x₀)), DF::LinearOperator=zeros(eltype(x₀), space(x₀), space(x₀)); tol::Real=1e-12, maxiter::Int=15, verbose::Bool=true)
    (tol < 0) | (maxiter < 0) && return throw(DomainError((tol, maxiter), "tolerance and maximum number of iterations must be positive"))
    return _newton!(F_DF!, x₀, F, DF, tol, maxiter, verbose)
end

function _newton!(F_DF!, x₀, F, DF, tol, maxiter, verbose)
    verbose && _display_newton_info(tol, maxiter)
    F_DF!(F, DF, x₀)
    nF = norm(F, Inf)
    verbose && _display_newton_nF(0, nF)
    if nF ≤ tol
        verbose && println()
        return x₀, true
    end
    i = 1
    while i ≤ maxiter && isfinite(nF)
        AF = DF \ F
        verbose && _display_newton_nAF(norm(AF, Inf))
        x₀ .-= AF
        F_DF!(F, DF, x₀)
        nF = norm(F, Inf)
        verbose && _display_newton_nF(i, nF)
        if nF ≤ tol
            verbose && println()
            return x₀, true
        end
        i += 1
    end
    verbose && println()
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
