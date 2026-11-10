# auxiliary radius ν̄ of the Cauchy estimate

function _optimize_decay(nl::Nonlinearity, c, aa, ρ)
    a, ε = sequence(aa), total_error(aa)
    ν = _analyticity_rate(banachspace(aa))
    T = eltype(a)
    function total(args...)
        finite_err, tail_err, pert, _ = _grid_error(nl.f, args...)
        s = _mag(finite_err + tail_err + pert)
        return isfinite(s) ? s : Inf
    end

    # the score is the bound on midpoints; it reads the magnitudes of c to trim the same noise as the bound
    a_mid, c_mag, ν_mag = mid.(a), Sequence(space(c), _mag.(coefficients(c))), _mag.(ν)
    analytic = _analyticity_test(nl, a_mid, _mag(ρ))
    score(μ) = analytic(μ) ? total(a_mid, c_mag, ν_mag, _mag(ε), _mag(ρ), μ, 4) : Inf

    bound(μ) = total(a, c, ν, ε, ρ, _maybe_interval.(T, μ))
    verified(μ) = _isentire(nl) || _check_branch_cut_poles(nl, a, _maybe_interval.(T, μ), ρ)

    ν̄ = _search_radius(score, ν_mag)
    ν̄ === nothing || (ν̄ = _refine_radius(bound, verified, ν_mag, ν̄))
    ν̄ === nothing && return throw(ArgumentError("no auxiliary radius ν̄ > ν = $(ν_mag) keeps `f` analytic within $(ρ) of the contour: analyticity violated"))
    return ν̄
end

# golden section of the bound along the ray μ(t) = ν (ν̄ / ν)^t, t ∈ [1/2, 3/2], below the largest verified t
function _refine_radius(bound, verified, ν, ν̄)
    μ(t) = ν .* (ν̄ ./ ν) .^ t
    t_up = verified(μ(1.5)) ? 1.5 : verified(μ(1.0)) ? 1.0 : _largest_passing(t -> verified(μ(t)), 1.0, 8)
    t_up === nothing && return nothing
    b(t) = bound(μ(t))
    t₀ = min(1.0, t_up)
    return μ(_golden_section(b, min(0.5, t_up / 2), t_up, t₀, b(t₀), 0.03))
end

# largest t ∈ (0, t_fail), within 1/16, passing a test that only fails as t grows; `nothing` if none
# above t_fail / 2^depth
function _largest_passing(test, t_fail, depth = 30)
    t_ok, t_min = zero(t_fail), t_fail / 2^depth
    while t_fail - t_ok > max(t_ok, t_min) / 16
        t = (t_ok + t_fail) / 2
        if test(t)
            t_ok = t
        else
            t_fail = t
        end
    end
    return t_ok > 0 ? t_ok : nothing
end

# analyticity only fails as μ grows: radii below a verified one, or above a refuted one, need no check
function _analyticity_test(nl::Nonlinearity, a::Sequence, ρ)
    _isentire(nl) && return _ -> true
    verified, refuted = Tuple[], Tuple[]
    return function (μ)
        any(v -> all(μ .≤ v), verified) && return true
        any(v -> all(μ .≥ v), refuted) && return false
        isanalytic = _check_branch_cut_poles(nl, a, _maybe_interval.(eltype(a), μ), ρ)
        push!(isanalytic ? verified : refuted, μ)
        return isanalytic
    end
end

# minimum of f over the radii above ν where it is finite, an interval: scan in s = log₂(μ / ν) by steps
# h = 1, or the largest feasible h below, until f rises twice, then golden section around the best s
function _search_radius(f, ν::Real)
    g(s) = f(ν * exp2(s))
    h = 1.0
    best_g = g(h)
    if !isfinite(best_g)
        h = _largest_passing(s -> isfinite(g(s)), h)
        h === nothing && return nothing
        best_g = g(h)
    end
    s, g_s, rises, best_s = h, best_g, 0, h
    while rises < 2 && s < 1024
        s += h
        g_new = g(s)
        isfinite(g_new) || break
        rises = ifelse(g_new > g_s, rises + 1, 0)
        g_s = g_new
        if g_new < best_g
            best_s, best_g = s, g_new
        end
    end
    return ν * exp2(_golden_section(g, max(best_s - h, zero(h)), best_s + h, best_s, best_g, h / 100))
end

function _search_radius(f, ν::Tuple{Any})
    μ = _search_radius(μ -> f((μ,)), ν[1])
    return μ === nothing ? nothing : (μ,)
end

# coordinate descent from a feasible point of the diagonal
function _search_radius(f, ν::NTuple{N}) where {N}
    t = _search_radius(t -> f(ν .* t), one(ν[1]))
    t === nothing && return nothing
    x = ν .* t
    for _ ∈ 1:8
        x_prev = x
        for i ∈ 1:N
            xᵢ = _search_radius(μ -> f(Base.setindex(x, μ, i)), ν[i])
            xᵢ === nothing || (x = Base.setindex(x, xᵢ, i))
        end
        all(abs.(log.(x ./ x_prev)) .< 0.01) && break
    end
    return x
end

# best point seen by a golden section on [lo, hi], given the value at a starting point
function _golden_section(g, lo, hi, best_s, best_g, tol)
    ϕ = (sqrt(5) - 1) / 2
    c, d = hi - ϕ * (hi - lo), lo + ϕ * (hi - lo)
    g_c, g_d = g(c), g(d)
    while true
        if g_c < best_g
            best_s, best_g = c, g_c
        end
        if g_d < best_g
            best_s, best_g = d, g_d
        end
        hi - lo > tol || return best_s
        if g_c ≤ g_d
            hi, d, g_d = d, c, g_c
            c = hi - ϕ * (hi - lo)
            g_c = g(c)
        else
            lo, c, g_c = c, d, g_d
            d = lo + ϕ * (hi - lo)
            g_d = g(d)
        end
    end
end
