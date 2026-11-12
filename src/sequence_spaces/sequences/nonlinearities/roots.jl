# solution x of b xᵖ = a: division (p = 1) and integer roots (b = 1)

# Newton-Kantorovich around x̄ for T(x) = x - A (b xᵖ - a), A ≈ (p b x̄ᵖ⁻¹)⁻¹: in the binomial expansion of
# p A b (x̄ + y)ᵖ⁻¹, the term j = 0 gives Z₁ and the terms j ≥ 1, for ‖y‖ ≤ R, give Z₂
function _newton_root(x̄_seq::Sequence, A_seq::Sequence, a, b, p::Int, X::BanachSpace, ::Type{T}) where {T}
    x̄ = InfiniteSequence(_maybe_interval(T, x̄_seq), X)
    A = InfiniteSequence(_maybe_interval(T, A_seq), X)
    pAb = exact(p) * (A * b)
    term(m) = m == 0 ? pAb : pAb * x̄ ^ m

    Y = norm(A * (b * x̄ ^ p - a))
    Z₁ = norm(term(p - 1) - exact(1))
    R = p ≤ 2 ? Inf : 3/2 * sup(Y) / (1 - sup(Z₁))
    Z₂ = zero(Y)
    for j ∈ 1:p-1
        Rʲ⁻¹ = j == 1 ? exact(1) : interval(R) ^ exact(j - 1)
        Z₂ += exact(binomial(p - 1, j)) * norm(term(p - 1 - j)) * Rʲ⁻¹
    end
    r, success = interval_of_existence(Y, Z₁, Z₂, R; verbose = false)
    success || return throw(ProofError("the solution of b xᵖ = a could not be proven around its approximation of order $(order(x̄)): Y = $(Y), Z₁ = $(Z₁), Z₂ = $(Z₂)"))

    # the radius is as guaranteed as its bounds, which carry the flags of a and b
    return InfiniteSequence(sequence(x̄), X; total_error = _guaranteed_as(_maybe_interval(T, inf(r)), Y, Z₁, Z₂))
end

_guaranteed_as(x::Interval, bounds::Interval...) =
    IntervalArithmetic._unsafe_interval(bareinterval(x), decoration(x), all(isguaranteed, bounds))
_guaranteed_as(x, bounds...) = x



# division

Base.inv(a::Sequence{<:SequenceSpace}; codomain = nothing, symmetry = nothing) = _apply(inv, a, codomain, symmetry)

function Base.inv(a::InfiniteSequence; codomain = nothing)
    _isexactconstant(a) && return _at_value(inv, a, codomain)
    x̄ = _apply(inv, mid.(sequence(a)), codomain, nothing)
    return _newton_root(x̄, x̄, exact(1), a, 1, banachspace(a), eltype(a))
end

function Base.:/(a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace})
    _isconstant(b) && return a / b[_findindex_constant(space(b))]
    T = promote_type(eltype(a), eltype(b))
    return _noise_trim(_resolve(s -> _grid_coefficients(/, s, T, a, b), union(desymmetrize(space(a)), desymmetrize(space(b)))), NoiseFloor())
end
Base.:/(a::Number, b::Sequence{<:SequenceSpace}) = lmul!(a, inv(b))

function Base.:/(a::InfiniteSequence, b::InfiniteSequence)
    _isexactconstant(b) && return a / sequence(b)[_findindex_constant(space(b))]
    b_mid = mid.(sequence(b))
    x̄ = mid.(sequence(a)) / b_mid
    A = inv(b_mid)
    return _newton_root(x̄, A, a, b, 1, banachspace(a) ∩ banachspace(b), promote_type(eltype(a), eltype(b)))
end
Base.:/(a::Number, b::InfiniteSequence) = lmul!(a, inv(b))

Base.:\(a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace}) = b / a
Base.:\(a::Sequence{<:SequenceSpace}, b::Number) = b / a

Base.:\(a::InfiniteSequence, b::InfiniteSequence) = b / a
Base.:\(a::InfiniteSequence, b::Number) = b / a



# integer roots

# principal root
_root(q::Int, x) = q == 2 ? sqrt(x) : x ^ (1 // q)
_root(q::Int, x::Real) = q == 2 ? sqrt(x) : q == 3 ? cbrt(x) : x ^ (1 // q)

Base.sqrt(a::Union{Sequence{<:SequenceSpace},InfiniteSequence}; kwargs...) = _nthroot(a, 2; kwargs...)
Base.cbrt(a::Union{Sequence{<:SequenceSpace},InfiniteSequence}; kwargs...) = _nthroot(a, 3; kwargs...)

_nthroot(a::Sequence{<:SequenceSpace}, q::Int; codomain = nothing, symmetry = nothing) =
    _apply(Base.Fix1(_root, q), a, codomain, symmetry)

function _nthroot(a::InfiniteSequence, q::Int; codomain = nothing)
    root = Base.Fix1(_root, q)
    _isexactconstant(a) && return _at_value(root, a, codomain)
    a_mid = mid.(sequence(a))
    x̄ = _apply(root, a_mid, codomain, nothing)
    A = _grid_coefficients(x -> inv(q * root(x) ^ (q - 1)), space(x̄), eltype(x̄), a_mid)
    return _newton_root(x̄, A, a, exact(1), q, banachspace(a), eltype(a))
end
