## Generic newton method

function newton(x₀, F_DF; tol::Real=1e-12, maxiter::Int=15, verbose::Bool=true)
    verbose && return _newton_verbose(x₀, F_DF, tol, maxiter)
    return _newton_silent(x₀, F_DF, tol, maxiter)
end

function _newton_verbose(x₀::Number, F_DF, tol, maxiter)
    printstyled("Starting Newton's method ...\n"; color = :blue)
    F, DF = F_DF(x₀)
    norm_F = abs(F)
    printstyled("    → iteration ", 0, ": ||F(x)|| = ", norm_F, "\n"; color = :yellow)
    if norm_F ≤ tol
        printstyled("    ... succes!\n"; color = :green)
        return x₀
    end
    x = x₀
    for i ∈ 1:maxiter
        x -= DF \ F
        F, DF = F_DF(x)
        norm_F = abs(F)
        printstyled("    → iteration ", i, ": ||F(x)|| = ", norm_F, "\n"; color = :yellow)
        if norm_F ≤ tol
            printstyled("    ... succes!\n"; color = :green)
            return x
        end
    end
    printstyled("    ... failure!\n"; color = :red)
    return x
end

function _newton_verbose(x₀::Vector, F_DF, tol, maxiter)
    printstyled("Starting Newton's method ...\n"; color = :blue)
    F, DF = F_DF(x₀)
    norm_F = norm(F, Inf)
    printstyled("    → iteration ", 0, ": ||F(x)|| = ", norm_F, "\n"; color = :yellow)
    if norm_F ≤ tol
        printstyled("    ... succes!\n"; color = :green)
        return x₀
    end
    x = copy(x₀)
    for i ∈ 1:maxiter
        x .-= ldiv!(lu!(DF), F)
        F, DF = F_DF(x)
        norm_F = norm(F, Inf)
        printstyled("    → iteration ", i, ": ||F(x)|| = ", norm_F, "\n"; color = :yellow)
        if norm_F ≤ tol
            printstyled("    ... succes!\n"; color = :green)
            return x
        end
    end
    printstyled("    ... failure!\n"; color = :red)
    return x
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

function _newton_silent(x₀::Vector, F_DF, tol, maxiter)
    F, DF = F_DF(x₀)
    norm(F, Inf) ≤ tol && return x₀
    x = copy(x₀)
    for i ∈ 1:maxiter
        x .-= ldiv!(lu!(DF), F)
        F, DF = F_DF(x)
        norm(F, Inf) ≤ tol && return x
    end
    return x
end
