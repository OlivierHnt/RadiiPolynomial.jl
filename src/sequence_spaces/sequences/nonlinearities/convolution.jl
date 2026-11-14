const CONV_ALGORITHM = Ref(:loop) # default

function set_conv_algorithm(algo::Symbol)
    algo ∉ (:fft, :loop) && return throw(ArgumentError("algorithm must be :fft or :loop"))
    CONV_ALGORITHM[] = algo
    return algo
end

# multiplication

"""
    *(a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace})

Compute the discrete convolution (associated with `space(a)` and `space(b)`) of
`a` and `b`.

See also: [`^(::Sequence{<:SequenceSpace}, ::Int)`](@ref).
"""
function Base.:*(a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace})
    space_c = codomain(*, space(a), space(b))
    if CONV_ALGORITHM[] === :fft
        return _mul_via_fft(a, b, space_c, space_c)
    else # CONV_ALGORITHM[] === :loop
        return _mul_via_sum(a, b, space_c)
    end
end

function mul_bar(a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace})
    space_c = codomain(mul_bar, space(a), space(b))
    if CONV_ALGORITHM[] === :fft
        # `mul_bar` truncates to `intersect(s_a, s_b)`, but the FFT must run on
        # the full convolution support (`codomain(*, ...)`) to avoid aliasing
        return _mul_via_fft(a, b, space_c, codomain(*, space(a), space(b)))
    else # CONV_ALGORITHM[] === :loop
        return _mul_via_sum(a, b, space_c)
    end
end

function _mul_via_fft(a::Sequence, b::Sequence, output_space::SequenceSpace, fft_space::SequenceSpace)
    CoefType = promote_type(eltype(a), eltype(b))
    da = _maybe_desym(a)
    db = _maybe_desym(b)
    dc = _grid_coefficients(*, desymmetrize(output_space), CoefType, da, db; grid = fft_space)
    _enforce_zeros!(dc, da, db)
    _noise_trim!(dc, BanachCeiling(da, db), ClampCoefficients())
    return _maybe_sym(dc, output_space)
end

function _mul_via_sum(a::Sequence, b::Sequence, output_space::SequenceSpace)
    CoefType = promote_type(eltype(a), eltype(b))
    c = zeros(CoefType, output_space)
    _conv!(c, a, b)
    return c
end

_maybe_desym(a::Sequence{<:NoSymSpace}) = a
_maybe_desym(a::Sequence{<:SymmetricSpace}) = Projection(desymmetrize(space(a))) * a
_maybe_sym(a::Sequence, ::NoSymSpace) = a
_maybe_sym(a::Sequence, s::SymmetricSpace) = Projection(s) * a

function _conv!(c::Sequence{<:SequenceSpace}, a, b)
    dsa = desymmetrize(space(a))
    dsb = desymmetrize(space(b))
    T = eltype(c)
    @inbounds for k ∈ indices(space(c))
        c[k] += _conv_sum(T, a, b, dsa, dsb, k, _convolution_indices(dsa, dsb, k))
    end
    return c
end

# the sum is accumulated in a local variable, one nested sum per dimension, so
# that each partial sum stays small

function _conv_sum(::Type{T}, a, b, dsa, dsb, k, j_indices) where {T}
    s = zero(T)
    for j ∈ j_indices
        s += _conv_term(a, b, dsa, dsb, k, j)
    end
    return s
end

_conv_sum(::Type{T}, a, b, dsa, dsb, k, j_indices::TensorIndices) where {T} =
    _conv_sum(T, a, b, dsa, dsb, k, j_indices.indices, ())

@inline function _conv_sum(::Type{T}, a, b, dsa, dsb, k, j_indices::Tuple, j_prefix::Tuple) where {T}
    s = zero(T)
    @inbounds inds = j_indices[1]
    for jᵢ ∈ inds
        s += _conv_sum(T, a, b, dsa, dsb, k, Base.tail(j_indices), (j_prefix..., jᵢ))
    end
    return s
end

@inline _conv_sum(::Type, a, b, dsa, dsb, k, ::Tuple{}, j::Tuple) = _conv_term(a, b, dsa, dsb, k, j)

@inline _conv_term(a, b, dsa, dsb, k, j) =
    @inbounds getcoefficient(a, (dsa, _extract_valid_index(dsa, k, j))) * getcoefficient(b, (dsb, _extract_valid_index(dsb, j)))

#

codomain(::typeof(*), s₁::TensorSpace{<:NTuple{N,BaseSpace}}, s₂::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    TensorSpace(map((s₁ᵢ, s₂ᵢ) -> codomain(*, s₁ᵢ, s₂ᵢ), spaces(s₁), spaces(s₂)))

codomain(::typeof(mul_bar), s₁::TensorSpace{<:NTuple{N,BaseSpace}}, s₂::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    TensorSpace(map((s₁ᵢ, s₂ᵢ) -> codomain(mul_bar, s₁ᵢ, s₂ᵢ), spaces(s₁), spaces(s₂)))

_convolution_indices(s₁::TensorSpace{<:NTuple{N,BaseSpace}}, s₂::TensorSpace{<:NTuple{N,BaseSpace}}, α::NTuple{N,Int}) where {N} =
    TensorIndices(map(_convolution_indices, spaces(s₁), spaces(s₂), α))

# Taylor

codomain(::typeof(*), s₁::Taylor, s₂::Taylor) = Taylor(order(s₁) + order(s₂))

codomain(::typeof(mul_bar), s₁::Taylor, s₂::Taylor) = intersect(s₁, s₂)

_convolution_indices(s₁::Taylor, s₂::Taylor, i::Int) = max(i-order(s₁), 0):min(i, order(s₂))

# Fourier

function codomain(::typeof(*), s₁::Fourier{T}, s₂::Fourier{S}) where {T,S}
    ω₁ = frequency(s₁)
    ω₂ = frequency(s₂)
    _safe_isequal(ω₁, ω₂) || return throw(ArgumentError("frequencies must be equal: s₁ has frequency $ω₁, s₂ has frequency $ω₂"))
    R = promote_type(T, S)
    return Fourier(order(s₁) + order(s₂), convert(R, ω₁))
end

codomain(::typeof(mul_bar), s₁::Fourier, s₂::Fourier) = intersect(s₁, s₂)

_convolution_indices(s₁::Fourier, s₂::Fourier, i::Int) = max(i-order(s₁), -order(s₂)):min(i+order(s₁), order(s₂))

# Chebyshev

codomain(::typeof(*), s₁::Chebyshev, s₂::Chebyshev) = Chebyshev(order(s₁) + order(s₂))

codomain(::typeof(mul_bar), s₁::Chebyshev, s₂::Chebyshev) = intersect(s₁, s₂)

_convolution_indices(s₁::Chebyshev, s₂::Chebyshev, i::Int) = max(i-order(s₁), -order(s₂)):min(i+order(s₁), order(s₂))

# Symmetric space

function codomain(::typeof(*), s₁::SymmetricSpace, s₂::SymmetricSpace)
    V = codomain(*, desymmetrize(s₁), desymmetrize(s₂))
    G = _codomain_convolution_symmetry(symmetry(s₁), symmetry(s₂))
    return SymmetricSpace(V, G)
end
codomain(::typeof(*), s₁::SymmetricSpace, s₂::NoSymSpace) = codomain(*, desymmetrize(s₁), s₂)
codomain(::typeof(*), s₁::NoSymSpace, s₂::SymmetricSpace) = codomain(*, s₁, desymmetrize(s₂))

function _codomain_convolution_symmetry(G₁::Group{N,L}, G₂::Group{N,L}) where {N,L}
    idx1 = _by_idx_action(G₁)
    idx2 = _by_idx_action(G₂)

    elems = Set{GroupElement{N,L}}()
    for (key, vals1) ∈ idx1
        haskey(idx2, key) || continue
        vals2 = idx2[key]
        for v1 ∈ vals1, v2 ∈ vals2
            if v1.phase == v2.phase
                push!(elems, GroupElement(key, Cocycle(v1.angle + v2.angle, v1.phase)))
            end
        end
    end

    return unsafe_group!(elems)
end

function _by_idx_action(G::Group{N,L}) where {N,L}
    idx = Dict{LatticeAut{N,L},Vector{Cocycle{N}}}()
    # e = GroupElement(LatticeAut(StaticArrays.SMatrix{N,N,Int}(I)), Cocycle(exact(1), StaticArrays.SVector{N,Rational{Int}}(ntuple(_ -> 0//1, Val(N)))))
    for g ∈ elements(G)
        # g == e && continue
        key = g.lattice_aut
        push!(get!(idx, key, Vector{Cocycle{N}}()), g.cocycle)
    end
    return idx
end





# integer power

"""
    ^(a::Sequence{<:SequenceSpace}, n::Int)

Compute the discrete convolution (associated with `space(a)`) of `a` with itself
`n` times.

See also: [`*(::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace})`](@ref).
"""
function Base.:^(a::Sequence{<:SequenceSpace}, n::Integer)
    n < 0 && return throw(DomainError(n, "^ is only defined for positive integers"))
    n == 0 && return one(a)
    n == 1 && return copy(a)
    if CONV_ALGORITHM[] === :fft
        space_c = codomain(^, space(a), n)
        da = _maybe_desym(a)
        dc = _grid_coefficients(x -> x ^ n, desymmetrize(space_c), eltype(a), da)
        _pow_enforce_zeros!(dc, da, n)
        _noise_trim!(dc, BanachCeiling(da, n), ClampCoefficients())
        return _maybe_sym(dc, space_c)
    else # CONV_ALGORITHM[] === :loop
        return _pow_by_squaring(a, n, _sqr, *)
    end
end

function pow_bar(a::Sequence{<:SequenceSpace}, n::Integer)
    n < 0 && return throw(DomainError(n, "pow_bar is only defined for positive integers"))
    n == 0 && return one(a)
    n == 1 && return copy(a)
    return _pow_by_squaring(a, n, _sqr_bar, mul_bar)
end

function _pow_by_squaring(a, n::Integer, sqr, mul)
    n == 2 && return sqr(a)
    t = trailing_zeros(n) + 1
    n >>= t
    while (t -= 1) > 0
        a = sqr(a)
    end
    c = a
    while n > 0
        t = trailing_zeros(n) + 1
        n >>= t
        while (t -= 1) ≥ 0
            a = sqr(a)
        end
        c = mul(c, a)
    end
    return c
end

function _sqr(a::Sequence{<:SequenceSpace})
    new_space = codomain(^, space(a), 2)
    CoefType = eltype(a)
    c = zeros(CoefType, new_space)
    _add_sqr!(c, a)
    return c
end

function _sqr_bar(a::Sequence{<:SequenceSpace})
    new_space = codomain(pow_bar, space(a), 2)
    CoefType = eltype(a)
    c = zeros(CoefType, new_space)
    _add_sqr!(c, a)
    return c
end

_add_sqr!(c::Sequence, a) = _conv!(c, a, a)

#

function codomain(::typeof(^), s::SequenceSpace, n::Integer)
    n < 0 && return throw(DomainError(n, "^ is only defined for positive integers"))
    n == 0 && return s
    n == 1 && return s
    s² = codomain(*, s, s)
    n == 2 && return s²
    return codomain(*, s², codomain(^, s, n-2))
end

function codomain(::typeof(pow_bar), s::SequenceSpace, n::Integer)
    n < 0 && return throw(DomainError(n, "pow_bar is only defined for positive integers"))
    n == 0 && return s
    n == 1 && return s
    s² = codomain(mul_bar, s, s)
    n == 2 && return s²
    return codomain(mul_bar, s², codomain(pow_bar, s, n-2))
end

# Taylor

function _add_sqr!(c::Sequence{Taylor}, a)
    order_a = order(space(a))
    @inbounds a₀ = a[0]
    @inbounds c[0] += a₀ ^ exact(2)
    @inbounds for i ∈ 1:order(space(c))
        cᵢ = zero(eltype(a))
        i_odd = i%2
        i_end = (i-2+i_odd)÷2
        @inbounds for j ∈ max(i-order_a, 0):i_end
            cᵢ += a[i-j] * a[j]
        end
        if iszero(i_odd)
            a_i½ = a[i÷2]
            c[i] += exact(2) * cᵢ + a_i½ ^ exact(2)
        else
            c[i] += exact(2) * cᵢ
        end
    end
    return c
end

# Fourier

function _add_sqr!(c::Sequence{<:Fourier}, a)
    order_a = order(space(a))
    c₀ = zero(eltype(a))
    @inbounds for j ∈ 1:order_a
        c₀ += a[j] * a[-j]
    end
    @inbounds a₀ = a[0]
    @inbounds c[0] += exact(2) * c₀ + a₀ ^ exact(2)
    @inbounds for i ∈ 1:order(space(c))
        cᵢ = c₋ᵢ = zero(eltype(a))
        i½, i_odd = divrem(i, 2)
        @inbounds for j ∈ i½+1:order_a
            cᵢ += a[i-j] * a[j]
            c₋ᵢ += a[j-i] * a[-j]
        end
        if iszero(i_odd)
            a_i½ = a[i½]
            a_neg_i½ = a[-i½]
            c[i] += exact(2) * cᵢ + a_i½ ^ exact(2)
            c[-i] += exact(2) * c₋ᵢ + a_neg_i½ ^ exact(2)
        else
            c[i] += exact(2) * cᵢ
            c[-i] += exact(2) * c₋ᵢ
        end
    end
    return c
end

# Chebyshev

function _add_sqr!(c::Sequence{Chebyshev}, a)
    order_a = order(space(a))
    c₀ = zero(eltype(a))
    @inbounds for j ∈ 1:order_a
        aⱼ = a[j]
        c₀ += aⱼ ^ exact(2)
    end
    @inbounds a₀ = a[0]
    @inbounds c[0] += exact(2) * c₀ + a₀ ^ exact(2)
    @inbounds for i ∈ 1:order(space(c))
        cᵢ = zero(eltype(a))
        i½, i_odd = divrem(i, 2)
        @inbounds for j ∈ i½+1:order_a
            cᵢ += a[abs(i-j)] * a[j]
        end
        if iszero(i_odd)
            a_i½ = a[i½]
            c[i] += exact(2) * cᵢ + a_i½ ^ exact(2)
        else
            c[i] += exact(2) * cᵢ
        end
    end
    return c
end





#

function Base.:*(a::InfiniteSequence, b::InfiniteSequence)
    c = sequence(a) * sequence(b)

    X = banachspace(a) ∩ banachspace(b)
    cross_bound = norm(sequence(a), X) * total_error(b) +
                  norm(sequence(b), X) * total_error(a) +
                  total_error(a) * total_error(b)

    new_full_norm = norm(a, X) * norm(b, X) # Banach algebra

    return _unsafe_infinite_sequence(c, norm(c, X), cross_bound, cross_bound, cross_bound, new_full_norm, X)
end

# truncated version: the product is cut back to the union of the spaces of the factors,
# and the cut-off coefficients (`spillover`) go into the tail error
# function Base.:*(a::InfiniteSequence, b::InfiniteSequence)
#     full_c = sequence(a) * sequence(b)
#     space_c = space(full_c) ∩ (desymmetrize(space(a)) ∪ desymmetrize(space(b)))
#     c = project(full_c, space_c)
#
#     @inbounds view(full_c, indices(space_c)) .= zero(eltype(full_c)) # keep the tail
#
#     X = banachspace(a) ∩ banachspace(b)
#     cross_bound = norm(sequence(a), X) * total_error(b) +
#                   norm(sequence(b), X) * total_error(a) +
#                   total_error(a) * total_error(b)
#     spillover = norm(full_c, X)
#     new_finite = cross_bound
#     new_tail = spillover + cross_bound
#     new_total = spillover + cross_bound
#
#     new_full_norm = norm(a, X) * norm(b, X) # Banach algebra
#
#     return _unsafe_infinite_sequence(c, norm(c, X), new_finite, new_tail, new_total, new_full_norm, X)
# end

Base.:*(a::InfiniteSequence, b::Sequence) = a * InfiniteSequence(b, banachspace(a))
Base.:*(a::Sequence, b::InfiniteSequence) = InfiniteSequence(a, banachspace(b)) * b

function Base.:^(a::InfiniteSequence, n::Integer)
    n < 0 && return inv(a^(-n))
    n == 0 && return one(a)
    n == 1 && return copy(a)
    n == 2 && return a*a
    # power by squaring
    t = trailing_zeros(n) + 1
    n >>= t
    while (t -= 1) > 0
        a *= a
    end
    c = a
    while n > 0
        t = trailing_zeros(n) + 1
        n >>= t
        while (t -= 1) ≥ 0
            a *= a
        end
        c = c * a
    end
    return c
end





# enforce 0 coefficient outside the known support

# support of the result from those of its factors, per direction
function _enforce_support!(c::Sequence, support, factors::Sequence...)
    C = _no_alloc_reshape(c)
    for (D, sc) ∈ enumerate(_directions(space(c)))
        bounds = map(a -> _nonzero_bounds(_no_alloc_reshape(a), _directions(space(a))[D], Val(D)), factors)
        _zero_outside!(C, sc, support(sc, bounds...)..., Val(D))
    end
    return c
end
_directions(s::BaseSpace) = (s,)
_directions(s::TensorSpace) = spaces(s)

_enforce_zeros!(c::Sequence, a, b) = _enforce_support!(c, (s, a, b) -> _conv_bounds(s, a..., b...), a, b)
_pow_enforce_zeros!(c::Sequence, a, n::Integer) = _enforce_support!(c, (s, a) -> _pow_bounds(s, a..., n), a)

function _nonzero_bounds(C, s, ::Val{D}) where {D}
    pred = i -> any(!iszero, selectdim(C, D, i))
    lo = findfirst(pred, 1:size(C, D))
    lo === nothing && return 0, -1
    hi = findlast(pred, 1:size(C, D))
    offset = first(indices(s)) - 1
    return lo + offset, hi + offset
end
# vectors: 10 times faster without slices
function _nonzero_bounds(C::AbstractVector, s, ::Val{1})
    lo = findfirst(!iszero, C)
    lo === nothing && return 0, -1
    offset = first(indices(s)) - 1
    return lo + offset, findlast(!iszero, C) + offset
end
_conv_bounds(::Taylor, amin, amax, bmin, bmax) = (amin + bmin, amax + bmax)
_conv_bounds(::Fourier, amin, amax, bmin, bmax) = (amin + bmin, amax + bmax)
_conv_bounds(::Chebyshev, amin, amax, bmin, bmax) = (max(0, amin - bmax, bmin - amax), amax + bmax)
function _pow_bounds(s::BaseSpace, amin, amax, n)
    cmin, cmax = amin, amax
    for _ in 2:n
        cmin, cmax = _conv_bounds(s, cmin, cmax, amin, amax)
    end
    return cmin, cmax
end
