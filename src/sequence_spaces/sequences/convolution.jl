const CONV_ALGORITHM = Ref(:sum) # default

function set_conv_algorithm(algo::Symbol)
    algo ∉ (:fft, :sum) && return throw(ArgumentError("algorithm must be :fft or :sum"))
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
    CoefType = promote_type(eltype(a), eltype(b))
    if CONV_ALGORITHM[] === :fft
        da = _maybe_desym(a)
        db = _maybe_desym(b)
        A = to_grid(da, fft_size(space_c))
        B = to_grid(db, fft_size(space_c))
        C = A .* B
        dc = _call_to_seq!(C, desymmetrize(space_c), CoefType)
        _enforce_zeros!(dc, da, db)
        banach_rounding!(dc, da, db)
        c = _maybe_sym(dc, space_c)
    else # CONV_ALGORITHM[] === :sum
        c = zeros(CoefType, space_c)
        _conv!(c, a, b)
    end
    return c
end

_maybe_desym(a::Sequence{<:NoSymSpace}) = a
_maybe_desym(a::Sequence{<:SymmetricSpace}) = Projection(desymmetrize(space(a))) * a
_maybe_sym(a::Sequence, ::NoSymSpace) = a
_maybe_sym(a::Sequence, s::SymmetricSpace) = Projection(s) * a

function mul_bar(a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace})
    space_c = codomain(mul_bar, space(a), space(b))
    CoefType = promote_type(eltype(a), eltype(b))
    if CONV_ALGORITHM[] === :fft
        full_space = codomain(*, space(a), space(b))
        da = _maybe_desym(a)
        db = _maybe_desym(b)
        A = to_grid(da, fft_size(full_space))
        B = to_grid(db, fft_size(full_space))
        C = A .* B
        dc = _call_to_seq!(C, desymmetrize(space_c), CoefType)
        _enforce_zeros!(dc, da, db)
        banach_rounding!(dc, da, db)
        c = _maybe_sym(dc, space_c)
    else # CONV_ALGORITHM[] === :sum
        c = zeros(CoefType, space_c)
        _conv!(c, a, b)
    end
    return c
end

#-
_to_interval(::Type{T}, x) where {T<:Union{Interval,Complex{<:Interval}}} = interval(zero(T), x; format = :midpoint)

_to_interval(::Type{T}, _) where {T} = zero(T)

function banach_rounding_order(bound::T, X::Ell1{GeometricWeight{T}}) where {T<:AbstractFloat}
    (rate(weight(X)) ≤ 1) | isinf(bound) && return typemax(Int)
    v = bound/eps(T)
    v ≤ 1 && return 0
    order = log(v)/log(rate(weight(X)))
    isinf(order) && return typemax(Int)
    return ceil(Int, order)
end

function banach_rounding_order(bound::T,  X::Ell1{AlgebraicWeight{T}}) where {T<:AbstractFloat}
    (rate(weight(X)) == 0) | isinf(bound) && return typemax(Int)
    v = bound/eps(T)
    v ≤ 1 && return 0
    order = exp(log(v)/rate(weight(X)))-1
    isinf(order) && return typemax(Int)
    return ceil(Int, order)
end

for T ∈ (:GeometricWeight, :AlgebraicWeight)
    @eval begin
        function banach_rounding_order(bound_::Real, X::Ell1{<:$T})
            bound, r = promote(float(sup(bound_)), float(sup(rate(weight(X)))))
            return banach_rounding_order(bound, Ell1($T(r)))
        end
    end
end

function banach_rounding_order(bound_::Real, X::Ell1{<:Tuple})
    bound = sup(bound_)
    return map(wᵢ -> banach_rounding_order(bound, Ell1(wᵢ)), weight(X))
end

#

function banach_rounding!(c::Sequence, a::Sequence, b::Sequence)
    X = Ell1(weight(a)) ∩ Ell1(weight(b))
    bound = norm(a, X) * norm(b, X)
    return banach_rounding!(c, bound, X, banach_rounding_order(bound, X))
end

function banach_rounding!(c::Sequence, a::Sequence, n::Integer)
    X = Ell1(weight(a))
    bound = norm(a, X)^n
    return banach_rounding!(c, bound, X, banach_rounding_order(bound, X))
end

function banach_rounding!(c::Sequence, a::Sequence, b::Sequence, X::Ell1)
    bound = norm(a, X) * norm(b, X)
    return banach_rounding!(c, bound, X, banach_rounding_order(bound, X))
end

function banach_rounding!(a::Sequence{TensorSpace{T},<:AbstractVector{S}}, bound::Real, X::Ell1, rounding_order::NTuple{N,Int}) where {N,T<:NTuple{N,BaseSpace},S}
    (inf(bound) ≥ 0) & all(≥(0), rounding_order) || return throw(DomainError((bound, rounding_order), "the bound and the rounding order must be positive"))
    space_a = space(a)
    M = typemax(Int)
    @inbounds for α ∈ indices(space_a)
        if mapreduce((i, ord) -> ifelse(ord == M, 0//1, ifelse(ord == 0, 1//1, abs(i) // max(1, ord))), +, α, rounding_order) ≥ 1
            μᵅ = bound / _getindex(weight(X), space_a, α)
            a[α] = _to_interval(S, sup(μᵅ))
        end
    end
    return a
end

# Taylor

function banach_rounding!(a::Sequence{Taylor,<:AbstractVector{T}}, bound::Real, X::Ell1{<:GeometricWeight}, rounding_order::Int) where {T}
    (inf(bound) ≥ 0) & (rounding_order ≥ 0) || return throw(DomainError((bound, rounding_order), "the bound and the rounding order must be positive"))
    if rounding_order ≤ order(a)
        ν⁻¹ = inv(rate(weight(X)))
        μⁱ = bound / _getindex(weight(X), space(a), rounding_order)
        @inbounds for i ∈ rounding_order:order(a)
            a[i] = _to_interval(T, sup(μⁱ))
            μⁱ *= ν⁻¹
        end
    end
    return a
end

function banach_rounding!(a::Sequence{Taylor,<:AbstractVector{T}}, bound::Real, X::Ell1{<:AlgebraicWeight}, rounding_order::Int) where {T}
    (inf(bound) ≥ 0) & (rounding_order ≥ 0) || return throw(DomainError((bound, rounding_order), "the bound and the rounding order must be positive"))
    space_a = space(a)
    @inbounds for i ∈ rounding_order:order(a)
        μⁱ = bound / _getindex(weight(X), space_a, i)
        a[i] = _to_interval(T, sup(μⁱ))
    end
    return a
end

# Fourier

function banach_rounding!(a::Sequence{<:Fourier,<:AbstractVector{T}}, bound::Real, X::Ell1{<:GeometricWeight}, rounding_order::Int) where {T}
    (inf(bound) ≥ 0) & (rounding_order ≥ 0) || return throw(DomainError((bound, rounding_order), "the bound and the rounding order must be positive"))
    if rounding_order ≤ order(a)
        ν⁻¹ = inv(rate(weight(X)))
        μⁱ = bound / _getindex(weight(X), space(a), rounding_order)
        @inbounds for i ∈ rounding_order:order(a)
            x = _to_interval(T, sup(μⁱ))
            a[i] = x
            a[-i] = x
            μⁱ *= ν⁻¹
        end
    end
    return a
end

function banach_rounding!(a::Sequence{<:Fourier,<:AbstractVector{T}}, bound::Real, X::Ell1{<:AlgebraicWeight}, rounding_order::Int) where {T}
    (inf(bound) ≥ 0) & (rounding_order ≥ 0) || return throw(DomainError((bound, rounding_order), "the bound and the rounding order must be positive"))
    space_a = space(a)
    @inbounds for i ∈ rounding_order:order(a)
        μⁱ = bound / _getindex(weight(X), space_a, i)
        x = _to_interval(T, sup(μⁱ))
        a[i] = x
        a[-i] = x
    end
    return a
end

# Chebyshev

function banach_rounding!(a::Sequence{Chebyshev,<:AbstractVector{T}}, bound::Real, X::Ell1{<:GeometricWeight}, rounding_order::Int) where {T}
    (inf(bound) ≥ 0) & (rounding_order ≥ 0) || return throw(DomainError((bound, rounding_order), "the bound and the rounding order must be positive"))
    if rounding_order ≤ order(a)
        ν⁻¹ = inv(rate(weight(X)))
        μⁱ = bound / _getindex(weight(X), space(a), rounding_order)
        @inbounds for i ∈ rounding_order:order(a)
            a[i] = _to_interval(T, sup(μⁱ))
            μⁱ *= ν⁻¹
        end
    end
    return a
end

function banach_rounding!(a::Sequence{Chebyshev,<:AbstractVector{T}}, bound::Real, X::Ell1{<:AlgebraicWeight}, rounding_order::Int) where {T}
    (inf(bound) ≥ 0) & (rounding_order ≥ 0) || return throw(DomainError((bound, rounding_order), "the bound and the rounding order must be positive"))
    space_a = space(a)
    @inbounds for i ∈ rounding_order:order(a)
        μⁱ = bound / _getindex(weight(X), space_a, i)
        a[i] = _to_interval(T, sup(μⁱ))
    end
    return a
end
#-

#-
_enforce_zeros!(c::Sequence{<:BaseSpace}, a, b) =
    _enforce_zeros!(coefficients(c), coefficients(a), coefficients(b), space(c), space(a), space(b))

_enforce_zeros!(c::Sequence{<:TensorSpace}, a, b) =
    _enforce_zeros!(_no_alloc_reshape(coefficients(c), dimensions(space(c))),
                    _no_alloc_reshape(coefficients(a), dimensions(space(a))),
                    _no_alloc_reshape(coefficients(b), dimensions(space(b))),
                    space(c), space(a), space(b))

_enforce_zeros!(C::AbstractArray{T,N₁}, A, B, space_c::TensorSpace{<:NTuple{N₂,BaseSpace}}, space_a, space_b) where {T,N₁,N₂} =
    @inbounds _enforce_zeros!(_enforce_zeros!(C, A, B, space_c[1], space_a[1], space_b[1], Val(N₁ - N₂ + 1)), A, B, Base.tail(space_c), Base.tail(space_a), Base.tail(space_b))
_enforce_zeros!(C::AbstractArray{T,N}, A, B::AbstractArray, space_c::TensorSpace{<:Tuple{BaseSpace}}, space_a, space_b) where {T,N} =
    @inbounds _enforce_zeros!(C, A, B, space_c[1], space_a[1], space_b[1], Val(N))

function _enforce_zeros!(C, A, B, sc::BaseSpace, sa, sb)
    amin, amax = _nonzero_bounds(A, sa)
    bmin, bmax = _nonzero_bounds(B, sb)
    cmin, _ = _get_order_mul(sc, amin, bmin)
    _, cmax = _get_order_mul(sc, amax, bmax)
    CoefType = eltype(C)
    for i ∈ 1:length(C)
        k = _index_to_math(sc, i)
        if k < cmin || cmax < k
            C[i] = zero(CoefType)
        end
    end
    return C
end
function _enforce_zeros!(C, A, B, sc::BaseSpace, sa, sb, ::Val{D}) where {D}
    amin, amax = _nonzero_bounds(A, sa, Val(D))
    bmin, bmax = _nonzero_bounds(B, sb, Val(D))
    cmin, _ = _get_order_mul(sc, amin, bmin)
    _, cmax = _get_order_mul(sc, amax, bmax)
    CoefType = eltype(C)
    @inbounds for i ∈ 1:size(C, D)
        k = _index_to_math(sc, i)
        if k < cmin || cmax < k
            selectdim(C, D, i) .= zero(CoefType)
        end
    end
    return C
end

function _nonzero_bounds(C, s)
    first_idx = 0
    last_idx  = 0
    found = false
    @inbounds for i ∈ 1:length(C)
        if !iszero(C[i])
            if !found
                first_idx = i
                found = true
            end
            last_idx = i
        end
    end
    !found && return 0, -1 # all zeros
    return _index_to_math(s, first_idx), _index_to_math(s, last_idx)
end
function _nonzero_bounds(C, s, ::Val{D}) where {D}
    first_idx = 0
    last_idx  = 0
    found = false
    @inbounds for i ∈ 1:size(C, D)
        if any(!iszero, selectdim(C, D, i))
            if !found
                first_idx = i
                found = true
            end
            last_idx = i
        end
    end
    !found && return 0, -1 # all zeros
    return _index_to_math(s, first_idx), _index_to_math(s, last_idx)
end

_get_order_mul(::Taylor, i, j) = (i+j, i+j)
_index_to_math(::Taylor, j) = j - 1

_get_order_mul(::Fourier, i, j) = (i+j, i+j)
_index_to_math(s::Fourier, j) = j - (div(dimension(s), 2) + 1)

_get_order_mul(::Chebyshev, i, j) = (abs(i-j), i+j)
_index_to_math(::Chebyshev, j) = j - 1
#-

function _conv!(c::Sequence{<:SequenceSpace}, a, b)
    dsa = desymmetrize(space(a))
    dsb = desymmetrize(space(b))
    @inbounds for k ∈ indices(space(c)), j ∈ _convolution_indices(dsa, dsb, k)
        c[k] += getcoefficient(a, (dsa, _extract_valid_index(dsa, k, j))) * getcoefficient(b, (dsb, _extract_valid_index(dsb, j)))
    end
    return c
end

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

_convolution_indices(s₁::Fourier, s₂::Fourier, i::Int) = intersect(i .- indices(s₁), indices(s₂))

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

function _codomain_convolution_symmetry(G₁::Group{N,T}, G₂::Group{N,T}) where {N,T<:Number}
    idx1 = _by_idx_action(G₁)
    idx2 = _by_idx_action(G₂)

    elems = Set{GroupElement{N,T}}()
    for (key, vals1) ∈ idx1
        haskey(idx2, key) || continue
        vals2 = idx2[key]
        for v1 ∈ vals1, v2 ∈ vals2
            if v1.phase == v2.phase
                push!(elems, GroupElement{N,T}(key, CoefAction{N,T}(v1.amplitude * v2.amplitude, v1.phase)))
            end
        end
    end

    return unsafe_group!(elems)
end

function _by_idx_action(G::Group{N,T}) where {N,T<:Number}
    idx = Dict{IndexAction{N},Vector{CoefAction{N,T}}}()
    # e = GroupElement(IndexAction(StaticArrays.SMatrix{N,N,Int}(I)), CoefAction(exact(1), StaticArrays.SVector{N,Rational{Int}}(ntuple(_ -> 0//1, Val(N)))))
    for g ∈ elements(G)
        # g == e && continue
        key = g.index_action
        push!(get!(idx, key, Vector{CoefAction{N,T}}()), g.coef_action)
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
        A = to_grid(da, fft_size(space_c))
        C = A .^ n
        dc = _call_to_seq!(C, desymmetrize(space_c), eltype(a))
        _pow_enforce_zeros!(dc, da, n)
        banach_rounding!(dc, da, n)
        c = _maybe_sym(dc, space_c)
    else # CONV_ALGORITHM[] === :sum
        n == 2 && return _sqr(a)
        # power by squaring
        t = trailing_zeros(n) + 1
        n >>= t
        while (t -= 1) > 0
            a = _sqr(a)
        end
        c = a
        while n > 0
            t = trailing_zeros(n) + 1
            n >>= t
            while (t -= 1) ≥ 0
                a = _sqr(a)
            end
            c = c * a
        end
    end
    return c
end

function pow_bar(a::Sequence{<:SequenceSpace}, n::Integer)
    n < 0 && return throw(DomainError(n, "pow_bar is only defined for positive integers"))
    n == 0 && return one(a)
    n == 1 && return copy(a)
    n == 2 && return _sqr_bar(a)
    # power by squaring
    t = trailing_zeros(n) + 1
    n >>= t
    while (t -= 1) > 0
        a = _sqr_bar(a)
    end
    c = a
    while n > 0
        t = trailing_zeros(n) + 1
        n >>= t
        while (t -= 1) ≥ 0
            a = _sqr_bar(a)
        end
        c = mul_bar(c, a)
    end
    return c
end

#-
_pow_enforce_zeros!(c::Sequence{<:BaseSpace}, a, n::Integer) =
    _pow_enforce_zeros!(coefficients(c), coefficients(a), space(c), space(a), n)

_pow_enforce_zeros!(c::Sequence{<:TensorSpace}, a, n::Integer) =
    _pow_enforce_zeros!(_no_alloc_reshape(coefficients(c), dimensions(space(c))),
                        _no_alloc_reshape(coefficients(a), dimensions(space(a))),
                        space(c), space(a), n)

_pow_enforce_zeros!(C::AbstractArray{T,N₁}, A, space_c::TensorSpace{<:NTuple{N₂,BaseSpace}}, space_a, n) where {T,N₁,N₂} =
    @inbounds _pow_enforce_zeros!(_pow_enforce_zeros!(C, A, space_c[1], space_a[1], n, Val(N₁ - N₂ + 1)), A, Base.tail(space_c), Base.tail(space_a), n)
_pow_enforce_zeros!(C::AbstractArray{T,N}, A, space_c::TensorSpace{<:Tuple{BaseSpace}}, space_a, n) where {T,N} =
    @inbounds _pow_enforce_zeros!(C, A, space_c[1], space_a[1], n, Val(N))

function _pow_enforce_zeros!(C, A, sc::BaseSpace, sa, n)
    amin, amax = _nonzero_bounds(A, sa, Val(D))
    cmin, _ = _get_order_pow(sc, amin, n)
    _, cmax = _get_order_pow(sc, amax, n)
    CoefType = eltype(C)
    for i ∈ 1:length(C)
        k = _index_to_math(sc, i)
        if k < cmin || cmax < k
            C[i] = zero(CoefType)
        end
    end
    return C
end
function _pow_enforce_zeros!(C, A, sc::BaseSpace, sa, n, ::Val{D}) where {D}
    amin, amax = _nonzero_bounds(A, sa, Val(D))
    cmin, _ = _get_order_pow(sc, amin, n)
    _, cmax = _get_order_pow(sc, amax, n)
    CoefType = eltype(C)
    @inbounds for i ∈ 1:size(C, D)
        k = _index_to_math(sc, i)
        if k < cmin || cmax < k
            selectdim(C, D, i) .= zero(CoefType)
        end
    end
    return C
end

_get_order_pow(::Taylor, i, n) = (i*n, i*n)

_get_order_pow(::Fourier, i, n) = (i*n, i*n)

_get_order_pow(::Chebyshev, i, n) = (ifelse(isodd(n), i % 2, 0), i*n)
#-

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
    full_c = sequence(a) * sequence(b)
    space_c = space(a) ∪ space(b)
    c = project(full_c, space_c)

    @inbounds view(full_c, indices(space_c)) .= zero(eltype(full_c)) # keep the tail

    X = banachspace(a) ∩ banachspace(b)
    new_err = norm(full_c, X) +
              norm(sequence(a), X) * sequence_error(b) +
              norm(sequence(b), X) * sequence_error(a) +
              sequence_error(a) * sequence_error(b)

    new_full_norm = norm(a, X) * norm(b, X) # Banach algebra

    return _unsafe_infinite_sequence(c, norm(c, X), new_err, new_full_norm, X)
end

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
