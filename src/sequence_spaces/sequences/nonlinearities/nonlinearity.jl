"""
    AnalyticityError(msg)

The exception thrown when the analyticity of a `Nonlinearity` cannot be
verified: the image of a contour, or the disc of the input error around it,
meets one of its `poles` or its `branch_cut`.
"""
struct AnalyticityError <: Exception
    msg :: String
end

    export AnalyticityError

Base.showerror(io::IO, e::AnalyticityError) = print(io, "AnalyticityError: ", e.msg)

"""
    Nonlinearity(f, poles, branch_cut)

A function `f`, together with the set where it fails to be analytic: the `poles`
and the `branch_cut`. Applied to a `Sequence` it evaluates `f` on a grid and
transforms back; applied to an `InfiniteSequence` it also bounds the error of
doing so, which is only possible while the contours it reads `f` off stay clear
of `poles` and of `branch_cut`.

The order of the result is resolved adaptively: `f ∘ a` is computed on spaces of
doubling order until its coefficients reach the rounding level. The keyword
`codomain` fixes the space of the result instead; on a `Sequence`, the keyword
`symmetry` resolves the order within the symmetric space of that group.
"""
struct Nonlinearity{U<:Function,T<:RealOrComplexI,S<:RealOrComplexI}
    f          :: U
    poles      :: Vector{T}
    branch_cut :: S
end

    export Nonlinearity


function Base.:^(a::Union{Sequence{<:SequenceSpace}, InfiniteSequence}, p::Real)
    isinteger(p) && return a ^ Integer(p)
    q = inv(p)
    isinteger(q) && 1 < q ≤ 16 && p == 1 // Int(q) && return _nthroot(a, Int(q))
    return Nonlinearity(x -> x ^ p, Complex{Interval{Float64}}[], interval(-Inf, 0))(a)
end


for f ∈ (:exp, :cos, :sin, :cosh, :sinh)
    @eval begin
        Nonlinearity(::typeof($f)) = Nonlinearity($f, Complex{Interval{Float64}}[], emptyinterval(Float64)) # entire

        Base.$f(a::Sequence{<:SequenceSpace}) = Nonlinearity($f)(a)

        Base.$f(a::InfiniteSequence) = Nonlinearity($f)(a)
    end
end

(nl::Nonlinearity)(a::Sequence{<:SequenceSpace}; codomain = nothing, symmetry = nothing) = _apply(nl.f, a, codomain, symmetry)

function (nl::Nonlinearity)(a::InfiniteSequence; codomain::Union{Nothing,SequenceSpace} = nothing)
    _isexactconstant(a) && return _at_value(nl.f, a, codomain)
    seq_a = _maybe_desym(sequence(a)) # the bound is computed on the full lattice
    ε = total_error(a)

    ν = _analyticity_rate(banachspace(a))
    _validate_rate(space(seq_a), ν)

    ρ = _safe_iszero(ε) ? ε : exact(1) + ε

    _check_branch_cut_poles(nl, seq_a, ν, ρ) ||
        return throw(AnalyticityError(_safe_iszero(ε) ?
            "the image of the ν = $(ν) contour intersects a branch cut or contains at least one pole: analyticity violated" :
            "the input error cannot be propagated: the Cauchy estimate needs f analytic on the disc of radius r⋆ = 1 + total_error(a) = $(ρ) around the image of the ν = $(ν) contour, and that disc meets a branch cut or a pole"))

    T = _image_eltype(nl.f, eltype(a))
    evaluate(s) = _grid_coefficients(nl.f, s, T, seq_a)
    c = codomain === nothing ? _resolve(evaluate, space(seq_a)) : evaluate(codomain)

    ν̄ = _maybe_interval.(eltype(a), _optimize_decay(nl, c, seq_a, ν, ε, ρ))

    finite_err, tail_err, pert, N_v = _grid_error(nl.f, seq_a, c, ν, ε, ρ, ν̄)
    total_err = finite_err + tail_err + pert

    if any(N_v .< 0) # nothing is kept: every index is in the tail
        c = zeros(eltype(c), _oforder(space(c), max.(N_v, 0)))
        finite_err = tail_err
    elseif N_v != Tuple(order(c))
        c = project(c, _oforder(space(c), N_v))
    end

    return InfiniteSequence(c, banachspace(a);
        finite_error = finite_err + pert,
        tail_error   = tail_err + pert,
        total_error  = total_err)
end

# one rate per direction
_analyticity_rate(X::BanachSpace) = Tuple(_analyticity_rate(weight(X)))
_analyticity_rate(w::Tuple{Vararg{Weight}}) = map(_analyticity_rate, w)
_analyticity_rate(w::GeometricWeight) = rate(w)
_analyticity_rate(::IdentityWeight) = exact(1) # valid only for Taylor
_analyticity_rate(w::Weight) = throw(ArgumentError("analyticity check requires a geometric weight, i.e. ω_k = ν^{|k|}; got $(_prettystring(w))"))
