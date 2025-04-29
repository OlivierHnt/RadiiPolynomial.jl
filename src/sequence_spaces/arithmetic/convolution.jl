_to_interval(::Type{T}, x) where {T<:Union{Interval,Complex{<:Interval}}} = interval(zero(T), x; format = :midpoint)

_to_interval(::Type{T}, _) where {T} = zero(T)

#

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

banach_rounding!(a, bound, X) = banach_rounding!(a, bound, X, banach_rounding_order(bound, X))

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

#

"""
    *(a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace})

Compute the discrete convolution (associated with `space(a)` and `space(b)`) of
`a` and `b`.

See also: [`mul!(::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace}, ::Number, ::Number)`](@ref),
[`^(::Sequence{<:SequenceSpace}, ::Int)`](@ref), [`banach_rounding_mul`](@ref)
and [`banach_rounding_mul!`](@ref) and [`banach_rounding_pow`](@ref).
"""
function Base.:*(a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace})
    new_space = image(*, space(a), space(b))
    CoefType = promote_type(eltype(a), eltype(b))
    c = Sequence(new_space, Vector{CoefType}(undef, dimension(new_space)))
    _mul!(c, a, b, convert(real(CoefType), ExactReal(true)), convert(real(CoefType), ExactReal(false)))
    return c
end

function mul_bar(a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace})
    new_space = image(mul_bar, space(a), space(b))
    CoefType = promote_type(eltype(a), eltype(b))
    c = Sequence(new_space, Vector{CoefType}(undef, dimension(new_space)))
    _mul!(c, a, b, convert(real(CoefType), ExactReal(true)), convert(real(CoefType), ExactReal(false)))
    return c
end

"""
    mul!(c::Sequence{<:SequenceSpace}, a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace}, α::Number, β::Number)

Compute `project(a * b, space(c)) * α + c * β` in-place. The result is stored in
`c` by overwriting it.

Note: `c` must not be aliased with either `a` or `b`.

See also: [`*(::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace})`](@ref),
[`^(::Sequence{<:SequenceSpace}, ::Int)`](@ref), [`banach_rounding_mul`](@ref),
[`banach_rounding_mul!`](@ref) and [`banach_rounding_pow`](@ref).
"""
function mul!(c::Sequence{<:SequenceSpace}, a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace}, α::Number, β::Number)
    space_c = space(c)
    new_space = image(*, space(a), space(b))
    _iscompatible(space_c, new_space) || return throw(ArgumentError("spaces must be compatible: c has space $space_c, a*b has space $new_space"))
    _mul!(c, a, b, α, β)
    return c
end
function _mul!(c::Sequence{<:SequenceSpace}, a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace}, α::Number, β::Number)
    if iszero(β)
        coefficients(c) .= zero(eltype(c))
    elseif !isone(β)
        coefficients(c) .*= β
    end
    _add_mul!(c, a, b, α)
    return c
end

"""
    banach_rounding_mul(a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace}, X::Ell1)

Compute the discrete convolution (associated with `space(a)` and `space(b)`) of
`a` and `b`. A cut-off order is estimated such that the coefficients of the
output beyond this order are rigorously enclosed.

See also: [`banach_rounding_mul!`](@ref), [`banach_rounding_pow`](@ref),
[`*(::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace})`](@ref),
[`mul!(::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace}, ::Number, ::Number)`](@ref)
and [`^(::Sequence{<:SequenceSpace}, ::Int)`](@ref).
"""
function banach_rounding_mul(a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace}, X::Ell1)
    new_space = image(*, space(a), space(b))
    CoefType = promote_type(eltype(a), eltype(b))
    c = Sequence(new_space, zeros(CoefType, dimension(new_space)))
    bound_ab = norm(a, X) * norm(b, X)
    _add_mul!(c, a, b, convert(real(CoefType), ExactReal(true)), bound_ab, X)
    return c
end

function banach_rounding_mul_bar(a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace}, X::Ell1)
    new_space = image(mul_bar, space(a), space(b))
    CoefType = promote_type(eltype(a), eltype(b))
    c = Sequence(new_space, zeros(CoefType, dimension(new_space)))
    bound_ab = norm(a, X) * norm(b, X)
    _add_mul!(c, a, b, convert(real(CoefType), ExactReal(true)), bound_ab, X)
    return c
end

"""
    banach_rounding_mul!(c::Sequence{<:SequenceSpace}, a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace}, X::Ell1)

Compute `project(banach_rounding_mul(a, b, X), space(c))` in-place. The result
is stored in `c` by overwriting it.

Note: `c` must not be aliased with either `a` or `b`.

See also: [`banach_rounding_mul`](@ref), [`banach_rounding_pow`](@ref),
[`*(::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace})`](@ref),
[`mul!(::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace}, ::Number, ::Number)`](@ref)
and [`^(::Sequence{<:SequenceSpace}, ::Int)`](@ref).
"""
function banach_rounding_mul!(c::Sequence{<:SequenceSpace}, a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace}, X::Ell1)
    space_c = space(c)
    new_space = image(*, space(a), space(b))
    _iscompatible(space_c, new_space) || return throw(ArgumentError("spaces must be compatible: c has space $space_c, a*b has space $new_space"))
    coefficients(c) .= zero(eltype(c))
    bound_ab = norm(a, X) * norm(b, X)
    _add_mul!(c, a, b, convert(real(eltype(c)), ExactReal(true)), bound_ab, X)
    return c
end

function _add_mul!(c::Sequence{<:BaseSpace}, a, b, α)
    _add_mul!(coefficients(c), coefficients(a), coefficients(b), α, space(c), space(a), space(b))
    return c
end
function _add_mul!(c::Sequence{<:TensorSpace}, a, b, α)
    space_c = space(c)
    space_a = space(a)
    space_b = space(b)
    C = _no_alloc_reshape(coefficients(c), dimensions(space_c))
    A = _no_alloc_reshape(coefficients(a), dimensions(space_a))
    B = _no_alloc_reshape(coefficients(b), dimensions(space_b))
    _add_mul!(C, A, B, α, space_c, space_a, space_b)
    return c
end
# for Banach rounding
function _add_mul!(c::Sequence{<:BaseSpace}, a, b, α, bound_ab, X)
    rounding_order = banach_rounding_order(bound_ab, X)
    space_c = space(c)
    _add_mul!(coefficients(c), coefficients(a), coefficients(b), α, space_c, space(a), space(b), bound_ab, X, rounding_order)
    return c
end
function _add_mul!(c::Sequence{<:TensorSpace}, a, b, α, bound_ab, X)
    rounding_order = banach_rounding_order(bound_ab, X)
    space_c = space(c)
    space_a = space(a)
    space_b = space(b)
    C = _no_alloc_reshape(coefficients(c), dimensions(space_c))
    A = _no_alloc_reshape(coefficients(a), dimensions(space_a))
    B = _no_alloc_reshape(coefficients(b), dimensions(space_b))
    _add_mul!(C, A, B, α, space_c, space_a, space_b, (), 0, space_c, bound_ab, X, rounding_order)
    return c
end
#

function _add_mul!(C, A, B, α, space_c::BaseSpace, space_a::BaseSpace, space_b::BaseSpace)
    for i ∈ indices(space_c)
        __convolution!(C, A, B, α, space_c, space_a, space_b, i)
    end
    return C
end
function _add_mul!(C, A, B, α, space_c::TensorSpace{<:NTuple{N,BaseSpace}}, space_a::TensorSpace{<:NTuple{N,BaseSpace}}, space_b::TensorSpace{<:NTuple{N,BaseSpace}}) where {N}
    remaining_space_c = Base.front(space_c)
    remaining_space_a = Base.front(space_a)
    remaining_space_b = Base.front(space_b)
    @inbounds current_space_c = space_c[N]
    @inbounds current_space_a = space_a[N]
    @inbounds current_space_b = space_b[N]
    for i ∈ indices(current_space_c)
        _convolution!(C, A, B, α, current_space_c, current_space_a, current_space_b, remaining_space_c, remaining_space_a, remaining_space_b, i)
    end
    return C
end
_add_mul!(C, A, B, α, space_c::TensorSpace{<:Tuple{BaseSpace}}, space_a::TensorSpace{<:Tuple{BaseSpace}}, space_b::TensorSpace{<:Tuple{BaseSpace}}) =
    @inbounds _add_mul!(C, A, B, α, space_c[1], space_a[1], space_b[1])
# for Banach rounding
function _add_mul!(C, A, B, α, space_c::BaseSpace, space_a::BaseSpace, space_b::BaseSpace, bound_ab, X, rounding_order)
    for i ∈ indices(space_c)
        __convolution!(C, A, B, α, space_c, space_a, space_b, i, bound_ab, X, rounding_order)
    end
    return C
end
function _add_mul!(C, A, B, α, space_c::TensorSpace{<:NTuple{N,BaseSpace}}, space_a::TensorSpace{<:NTuple{N,BaseSpace}}, space_b::TensorSpace{<:NTuple{N,BaseSpace}}, t, sum_t, full_space_c, bound_ab, X, rounding_order) where {N}
    remaining_space_c = Base.front(space_c)
    remaining_space_a = Base.front(space_a)
    remaining_space_b = Base.front(space_b)
    @inbounds current_space_c = space_c[N]
    @inbounds current_space_a = space_a[N]
    @inbounds current_space_b = space_b[N]
    for i ∈ indices(current_space_c)
        _convolution!(C, A, B, α, current_space_c, current_space_a, current_space_b, remaining_space_c, remaining_space_a, remaining_space_b, i,
            t, sum_t, full_space_c, bound_ab, X, rounding_order)
    end
    return C
end
function _add_mul!(C, A, B, α, space_c::TensorSpace{<:Tuple{BaseSpace}}, space_a::TensorSpace{<:Tuple{BaseSpace}}, space_b::TensorSpace{<:Tuple{BaseSpace}}, t, sum_t, full_space_c, bound_ab, X, rounding_order)
    @inbounds current_space_c = space_c[1]
    @inbounds current_space_a = space_a[1]
    @inbounds current_space_b = space_b[1]
    for i ∈ indices(current_space_c)
        __convolution!(C, A, B, α, current_space_c, current_space_a, current_space_b, i, t, sum_t, full_space_c, bound_ab, X, rounding_order)
    end
    return C
end
#

function _convolution!(C::AbstractArray{T,N}, A, B, α, current_space_c, current_space_a, current_space_b, remaining_space_c, remaining_space_a, remaining_space_b, i) where {T,N}
    @inbounds Cᵢ = selectdim(C, N, _findposition(i, current_space_c))
    @inbounds for j ∈ _convolution_indices(current_space_a, current_space_b, i)
        x = _inverse_symmetry_action(current_space_c, i) * _symmetry_action(current_space_a, i, j) * _symmetry_action(current_space_b, j)
        if !iszero(x)
            _add_mul!(Cᵢ,
                selectdim(A, N, _findposition(_extract_valid_index(current_space_a, i, j), current_space_a)),
                selectdim(B, N, _findposition(_extract_valid_index(current_space_b, j), current_space_b)),
                ExactReal(x) * α, remaining_space_c, remaining_space_a, remaining_space_b)
        end
    end
    return C
end
# for Banach rounding
function _convolution!(C::AbstractArray{T,N}, A, B, α, current_space_c, current_space_a, current_space_b, remaining_space_c, remaining_space_a, remaining_space_b, i, t, sum_t, full_space_c, bound_ab, X, rounding_order) where {T,N}
    t_ = (i, t...)
    sum_t += abs(i)
    @inbounds Cᵢ = selectdim(C, N, _findposition(i, current_space_c))
    @inbounds for j ∈ _convolution_indices(current_space_a, current_space_b, i)
        x = _inverse_symmetry_action(current_space_c, i) * _symmetry_action(current_space_a, i, j) * _symmetry_action(current_space_b, j)
        if !iszero(x)
            _add_mul!(Cᵢ,
                selectdim(A, N, _findposition(_extract_valid_index(current_space_a, i, j), current_space_a)),
                selectdim(B, N, _findposition(_extract_valid_index(current_space_b, j), current_space_b)),
                ExactReal(x) * α, remaining_space_c, remaining_space_a, remaining_space_b, t_, sum_t, full_space_c, bound_ab, X, rounding_order)
        end
    end
    return C
end
#

function __convolution!(C, A, B, α, space_c, space_a, space_b, i)
    Cᵢ = zero(promote_type(eltype(A), eltype(B)))
    @inbounds @simd for j ∈ _convolution_indices(space_a, space_b, i)
        x = _inverse_symmetry_action(space_c, i) * _symmetry_action(space_a, i, j) * _symmetry_action(space_b, j)
        if !iszero(x)
            Cᵢ += ExactReal(x) * A[_findposition(_extract_valid_index(space_a, i, j), space_a)] * B[_findposition(_extract_valid_index(space_b, j), space_b)]
        end
    end
    @inbounds C[_findposition(i, space_c)] += Cᵢ * α
    return C
end
# for Banach rounding
function __convolution!(C, A, B, α, space_c, space_a, space_b, i, bound_ab, X, rounding_order)
    CoefType = eltype(C)
    if rounding_order ≤ abs(i)
        μⁱ = bound_ab / _getindex(weight(X), space_c, i)
        @inbounds C[_findposition(i, space_c)] += _to_interval(CoefType, sup(α * μⁱ))
    else
        Cᵢ = zero(promote_type(eltype(A), eltype(B)))
        @inbounds @simd for j ∈ _convolution_indices(space_a, space_b, i)
            x = _inverse_symmetry_action(space_c, i) * _symmetry_action(space_a, i, j) * _symmetry_action(space_b, j)
            if !iszero(x)
                Cᵢ += ExactReal(x) * A[_findposition(_extract_valid_index(space_a, i, j), space_a)] * B[_findposition(_extract_valid_index(space_b, j), space_b)]
            end
        end
        @inbounds C[_findposition(i, space_c)] += Cᵢ * α
    end
    return C
end
function __convolution!(C, A, B, α, space_c, space_a, space_b, i, t, sum_t, full_space_c, bound_ab, X, rounding_order)
    CoefType = eltype(C)
    sum_t += abs(i)
    if any(≤(sum_t), rounding_order)
        μⁱ = bound_ab / _getindex(weight(X), full_space_c, (i, t...))
        @inbounds C[_findposition(i, space_c)] += _to_interval(CoefType, sup(α * μⁱ))
    else
        Cᵢ = zero(promote_type(eltype(A), eltype(B)))
        @inbounds @simd for j ∈ _convolution_indices(space_a, space_b, i)
            x = _inverse_symmetry_action(space_c, i) * _symmetry_action(space_a, i, j) * _symmetry_action(space_b, j)
            if !iszero(x)
                Cᵢ += ExactReal(x) * A[_findposition(_extract_valid_index(space_a, i, j), space_a)] * B[_findposition(_extract_valid_index(space_b, j), space_b)]
            end
        end
        @inbounds C[_findposition(i, space_c)] += Cᵢ * α
    end
    return C
end
#

_convolution_indices(s₁::TensorSpace{<:NTuple{N,BaseSpace}}, s₂::TensorSpace{<:NTuple{N,BaseSpace}}, α::NTuple{N,Int}) where {N} =
    TensorIndices(map(_convolution_indices, spaces(s₁), spaces(s₂), α))

_symmetry_action(s::TensorSpace{<:NTuple{N,BaseSpace}}, α::NTuple{N,Int}, β::NTuple{N,Int}) where {N} =
    @inbounds _symmetry_action(s[1], α[1], β[1]) * _symmetry_action(Base.tail(s), Base.tail(α), Base.tail(β))
_symmetry_action(s::TensorSpace{<:Tuple{BaseSpace}}, α::Tuple{Int}, β::Tuple{Int}) =
    @inbounds _symmetry_action(s[1], α[1], β[1])
_symmetry_action(s::TensorSpace{<:NTuple{N,BaseSpace}}, α::NTuple{N,Int}) where {N} =
    @inbounds _symmetry_action(s[1], α[1]) * _symmetry_action(Base.tail(s), Base.tail(α))
_symmetry_action(s::TensorSpace{<:Tuple{BaseSpace}}, α::Tuple{Int}) = @inbounds _symmetry_action(s[1], α[1])
_inverse_symmetry_action(s::TensorSpace{<:NTuple{N,BaseSpace}}, α::NTuple{N,Int}) where {N} =
    @inbounds _inverse_symmetry_action(s[1], α[1]) * _inverse_symmetry_action(Base.tail(s), Base.tail(α))
_inverse_symmetry_action(s::TensorSpace{<:Tuple{BaseSpace}}, α::Tuple{Int}) = @inbounds _inverse_symmetry_action(s[1], α[1])

_extract_valid_index(s::TensorSpace{<:NTuple{N,BaseSpace}}, α::NTuple{N,Int}, β::NTuple{N,Int}) where {N} =
    @inbounds (_extract_valid_index(s[1], α[1], β[1]), _extract_valid_index(Base.tail(s), Base.tail(α), Base.tail(β))...)
_extract_valid_index(s::TensorSpace{<:Tuple{BaseSpace}}, α::Tuple{Int}, β::Tuple{Int}) =
    @inbounds (_extract_valid_index(s[1], α[1], β[1]),)
_extract_valid_index(s::TensorSpace{<:NTuple{N,BaseSpace}}, α::NTuple{N,Int}) where {N} =
    @inbounds (_extract_valid_index(s[1], α[1]), _extract_valid_index(Base.tail(s), Base.tail(α))...)
_extract_valid_index(s::TensorSpace{<:Tuple{BaseSpace}}, α::Tuple{Int}) =
    @inbounds (_extract_valid_index(s[1], α[1]),)

# Taylor

function __convolution!(C, A, B, α, ::Taylor, space_a::Taylor, space_b::Taylor, i)
    Cᵢ = zero(promote_type(eltype(A), eltype(B)))
    @inbounds @simd for j ∈ max(i-order(space_a), 0):min(i, order(space_b)) # _convolution_indices(space_a, space_b, i)
        Cᵢ += A[i-j+1] * B[j+1]
    end
    @inbounds C[i+1] += Cᵢ * α
    return C
end
function _convolution!(C::AbstractArray{T,N}, A, B, α, ::Taylor, current_space_a::Taylor, current_space_b::Taylor, remaining_space_c, remaining_space_a, remaining_space_b, i) where {T,N}
    @inbounds Cᵢ = selectdim(C, N, i+1)
    @inbounds for j ∈ max(i-order(current_space_a), 0):min(i, order(current_space_b)) # _convolution_indices(current_space_a, current_space_b, i)
        _add_mul!(Cᵢ,
            selectdim(A, N, i-j+1),
            selectdim(B, N, j+1),
            α, remaining_space_c, remaining_space_a, remaining_space_b)
    end
    return C
end
# for Banach rounding
function __convolution!(C, A, B, α, space_c::Taylor, space_a::Taylor, space_b::Taylor, i, bound_ab, X, rounding_order)
    CoefType = eltype(C)
    if rounding_order ≤ i
        μⁱ = bound_ab / _getindex(weight(X), space_c, i)
        @inbounds C[i+1] += _to_interval(CoefType, sup(α * μⁱ))
    else
        Cᵢ = zero(promote_type(eltype(A), eltype(B)))
        @inbounds @simd for j ∈ max(i-order(space_a), 0):min(i, order(space_b)) # _convolution_indices(space_a, space_b, i)
            Cᵢ += A[i-j+1] * B[j+1]
        end
        @inbounds C[i+1] += Cᵢ * α
    end
    return C
end
function __convolution!(C, A, B, α, ::Taylor, space_a::Taylor, space_b::Taylor, i, t, sum_t, full_space_c, bound_ab, X, rounding_order)
    CoefType = eltype(C)
    sum_t += i
    if any(≤(sum_t), rounding_order)
        μⁱ = bound_ab / _getindex(weight(X), full_space_c, (i, t...))
        @inbounds C[i+1] += _to_interval(CoefType, sup(α * μⁱ))
    else
        Cᵢ = zero(promote_type(eltype(A), eltype(B)))
        @inbounds @simd for j ∈ max(i-order(space_a), 0):min(i, order(space_b)) # _convolution_indices(space_a, space_b, i)
            Cᵢ += A[i-j+1] * B[j+1]
        end
        @inbounds C[i+1] += Cᵢ * α
    end
    return C
end
function _convolution!(C::AbstractArray{T,N}, A, B, α, ::Taylor, current_space_a::Taylor, current_space_b::Taylor, remaining_space_c, remaining_space_a, remaining_space_b, i, t, sum_t, full_space_c, bound_ab, X, rounding_order) where {T,N}
    t_ = (i, t...)
    sum_t += i
    @inbounds Cᵢ = selectdim(C, N, i+1)
    @inbounds for j ∈ max(i-order(current_space_a), 0):min(i, order(current_space_b)) # _convolution_indices(current_space_a, current_space_b, i)
        _add_mul!(Cᵢ,
            selectdim(A, N, i-j+1),
            selectdim(B, N, j+1),
            α, remaining_space_c, remaining_space_a, remaining_space_b, t_, sum_t, full_space_c, bound_ab, X, rounding_order)
    end
    return C
end
#

_convolution_indices(s₁::Taylor, s₂::Taylor, i::Int) = max(i-order(s₁), 0):min(i, order(s₂))

_symmetry_action(::Taylor, ::Int, ::Int) = 1
_symmetry_action(::Taylor, ::Int) = 1
_inverse_symmetry_action(::Taylor, ::Int) = 1

_extract_valid_index(::Taylor, i::Int, j::Int) = i-j
_extract_valid_index(::Taylor, i::Int) = i

# Fourier

_convolution_indices(s₁::Fourier, s₂::Fourier, i::Int) = intersect(i .- indices(s₁), indices(s₂))

_symmetry_action(::Fourier, ::Int, ::Int) = 1
_symmetry_action(::Fourier, ::Int) = 1
_inverse_symmetry_action(::Fourier, ::Int) = 1

_extract_valid_index(::Fourier, i::Int, j::Int) = i-j
_extract_valid_index(::Fourier, i::Int) = i

# Chebyshev

function __convolution!(C, A, B, α, ::Chebyshev, space_a::Chebyshev, space_b::Chebyshev, i)
    order_a = order(space_a)
    order_b = order(space_b)
    Cᵢ = zero(promote_type(eltype(A), eltype(B)))
    @inbounds @simd for j ∈ max(i-order_a, -order_b):min(i+order_a, order_b) # _convolution_indices(space_a, space_b, i)
        Cᵢ += A[abs(i-j)+1] * B[abs(j)+1]
    end
    @inbounds C[i+1] += Cᵢ * α
    return C
end
function _convolution!(C::AbstractArray{T,N}, A, B, α, ::Chebyshev, current_space_a::Chebyshev, current_space_b::Chebyshev, remaining_space_c, remaining_space_a, remaining_space_b, i) where {T,N}
    order_a = order(current_space_a)
    order_b = order(current_space_b)
    @inbounds Cᵢ = selectdim(C, N, i+1)
    @inbounds for j ∈ max(i-order_a, -order_b):min(i+order_a, order_b) # _convolution_indices(current_space_a, current_space_b, i)
        _add_mul!(Cᵢ,
            selectdim(A, N, abs(i-j)+1),
            selectdim(B, N, abs(j)+1),
            α, remaining_space_c, remaining_space_a, remaining_space_b)
    end
    return C
end
# for Banach rounding
function __convolution!(C, A, B, α, space_c::Chebyshev, space_a::Chebyshev, space_b::Chebyshev, i, bound_ab, X, rounding_order)
    order_a = order(space_a)
    order_b = order(space_b)
    CoefType = eltype(C)
    if rounding_order ≤ i
        μⁱ = bound_ab / _getindex(weight(X), space_c, i)
        @inbounds C[i+1] += _to_interval(CoefType, sup(α * μⁱ))
    else
        Cᵢ = zero(promote_type(eltype(A), eltype(B)))
        @inbounds @simd for j ∈ max(i-order_a, -order_b):min(i+order_a, order_b) # _convolution_indices(space_a, space_b, i)
            Cᵢ += A[abs(i-j)+1] * B[abs(j)+1]
        end
        @inbounds C[i+1] += Cᵢ * α
    end
    return C
end
function __convolution!(C, A, B, α, ::Chebyshev, space_a::Chebyshev, space_b::Chebyshev, i, t, sum_t, full_space_c, bound_ab, X, rounding_order)
    order_a = order(space_a)
    order_b = order(space_b)
    CoefType = eltype(C)
    sum_t += i
    if any(≤(sum_t), rounding_order)
        μⁱ = bound_ab / _getindex(weight(X), full_space_c, (i, t...))
        @inbounds C[i+1] += _to_interval(CoefType, sup(α * μⁱ))
    else
        Cᵢ = zero(promote_type(eltype(A), eltype(B)))
        @inbounds @simd for j ∈ max(i-order_a, -order_b):min(i+order_a, order_b) # _convolution_indices(space_a, space_b, i)
            Cᵢ += A[abs(i-j)+1] * B[abs(j)+1]
        end
        @inbounds C[i+1] += Cᵢ * α
    end
    return C
end
function _convolution!(C::AbstractArray{T,N}, A, B, α, ::Chebyshev, current_space_a::Chebyshev, current_space_b::Chebyshev, remaining_space_c, remaining_space_a, remaining_space_b, i, t, sum_t, full_space_c, bound_ab, X, rounding_order) where {T,N}
    order_a = order(current_space_a)
    order_b = order(current_space_b)
    t_ = (i, t...)
    sum_t += i
    @inbounds Cᵢ = selectdim(C, N, i+1)
    @inbounds for j ∈ max(i-order_a, -order_b):min(i+order_a, order_b) # _convolution_indices(current_space_a, current_space_b, i)
        _add_mul!(Cᵢ,
            selectdim(A, N, abs(i-j)+1),
            selectdim(B, N, abs(j)+1),
            α, remaining_space_c, remaining_space_a, remaining_space_b, t_, sum_t, full_space_c, bound_ab, X, rounding_order)
    end
    return C
end
#

_convolution_indices(s₁::Chebyshev, s₂::Chebyshev, i::Int) = max(i-order(s₁), -order(s₂)):min(i+order(s₁), order(s₂))

_symmetry_action(::Chebyshev, ::Int, ::Int) = 1
_symmetry_action(::Chebyshev, ::Int) = 1
_inverse_symmetry_action(::Chebyshev, ::Int) = 1

_extract_valid_index(::Chebyshev, i::Int, j::Int) = abs(i-j)
_extract_valid_index(::Chebyshev, i::Int) = abs(i)

#

"""
    ^(a::Sequence{<:SequenceSpace}, n::Int)

Compute the discrete convolution (associated with `space(a)`) of `a` with itself
`n` times.

See also: [`*(::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace})`](@ref),
[`mul!(::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace}, ::Number, ::Number)`](@ref),
[`banach_rounding_mul`](@ref), [`banach_rounding_mul!`](@ref) and [`banach_rounding_pow`](@ref).
"""
function Base.:^(a::Sequence{<:SequenceSpace}, n::Int)
    n < 0 && return throw(DomainError(n, "^ is only defined for positive integers"))
    n == 0 && return one(a)
    n == 1 && return copy(a)
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
    return c
end

function pow_bar(a::Sequence{<:SequenceSpace}, n::Int)
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

function _sqr(a::Sequence{<:SequenceSpace})
    new_space = image(^, space(a), 2)
    CoefType = eltype(a)
    c = Sequence(new_space, zeros(CoefType, dimension(new_space)))
    _add_sqr!(c, a)
    return c
end

function _sqr_bar(a::Sequence{<:SequenceSpace})
    new_space = image(pow_bar, space(a), 2)
    CoefType = eltype(a)
    c = Sequence(new_space, zeros(CoefType, dimension(new_space)))
    _add_sqr!(c, a)
    return c
end

"""
    banach_rounding_pow(a::Sequence{<:SequenceSpace}, n::Int, X::Ell1)

Compute the discrete convolution (associated with `space(a)`) of `a` with itself
`n` times. A cut-off order is estimated such that the coefficients of the output
beyond this order are rigorously enclosed.

See also: [`banach_rounding_mul`](@ref), [`banach_rounding_mul!`](@ref),
[`*(::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace})`](@ref),
[`mul!(::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace}, ::Number, ::Number)`](@ref)
and [`^(::Sequence{<:SequenceSpace}, ::Int)`](@ref).
"""
function banach_rounding_pow(a::Sequence{<:SequenceSpace}, n::Int, X::Ell1)
    n < 0 && return throw(DomainError(n, "banach_rounding_pow is only defined for positive integers"))
    n == 0 && return one(a)
    n == 1 && return copy(a)
    n == 2 && return _banach_rounding_sqr(a, X)
    # power by squaring
    t = trailing_zeros(n) + 1
    n >>= t
    while (t -= 1) > 0
        a = _banach_rounding_sqr(a, X)
    end
    c = a
    while n > 0
        t = trailing_zeros(n) + 1
        n >>= t
        while (t -= 1) ≥ 0
            a = _banach_rounding_sqr(a, X)
        end
        c = banach_rounding_mul(c, a, X)
    end
    return c
end

function banach_rounding_pow_bar(a::Sequence{<:SequenceSpace}, n::Int, X::Ell1)
    n < 0 && return throw(DomainError(n, "banach_rounding_pow_bar is only defined for positive integers"))
    n == 0 && return one(a)
    n == 1 && return copy(a)
    n == 2 && return _banach_rounding_sqr_bar(a, X)
    # power by squaring
    t = trailing_zeros(n) + 1
    n >>= t
    while (t -= 1) > 0
        a = _banach_rounding_sqr_bar(a, X)
    end
    c = a
    while n > 0
        t = trailing_zeros(n) + 1
        n >>= t
        while (t -= 1) ≥ 0
            a = _banach_rounding_sqr_bar(a, X)
        end
        c = banach_rounding_mul_bar(c, a, X)
    end
    return c
end

function _banach_rounding_sqr(a::Sequence{<:SequenceSpace}, X::Ell1)
    new_space = image(^, space(a), 2)
    CoefType = eltype(a)
    c = Sequence(new_space, zeros(CoefType, dimension(new_space)))
    norm_a = norm(a, X)
    bound_a² = norm_a * norm_a
    _add_sqr!(c, a, bound_a², X)
    return c
end

function _banach_rounding_sqr_bar(a::Sequence{<:SequenceSpace}, X::Ell1)
    new_space = image(pow_bar, space(a), 2)
    CoefType = eltype(a)
    c = Sequence(new_space, zeros(CoefType, dimension(new_space)))
    norm_a = norm(a, X)
    bound_a² = norm_a * norm_a
    _add_sqr!(c, a, bound_a², X)
    return c
end

_add_sqr!(c::Sequence, a) = _add_mul!(c, a, a, convert(real(eltype(c)), ExactReal(true)))
_add_sqr!(c::Sequence, a, bound_a², X) = _add_mul!(c, a, a, convert(real(eltype(c)), ExactReal(true)), bound_a², X)

# Taylor

function _add_sqr!(c::Sequence{Taylor}, a)
    order_a = order(space(a))
    @inbounds a₀ = a[0]
    @inbounds c[0] += a₀ ^ ExactReal(2)
    @inbounds for i ∈ 1:order(space(c))
        cᵢ = zero(eltype(a))
        i_odd = i%2
        i_end = (i-2+i_odd)÷2
        @inbounds for j ∈ max(i-order_a, 0):i_end
            cᵢ += a[i-j] * a[j]
        end
        if iszero(i_odd)
            a_i½ = a[i÷2]
            c[i] += ExactReal(2) * cᵢ + a_i½ ^ ExactReal(2)
        else
            c[i] += ExactReal(2) * cᵢ
        end
    end
    return c
end

function _add_sqr!(c::Sequence{Taylor}, a, bound_a², X)
    rounding_order = banach_rounding_order(bound_a², X)
    if rounding_order > 0
        order_a = order(space(a))
        @inbounds a₀ = a[0]
        @inbounds c[0] += a₀ ^ ExactReal(2)
        @inbounds for i ∈ 1:min(order(space(c)), rounding_order-1)
            cᵢ = zero(eltype(a))
            i_odd = i%2
            i_end = (i-2+i_odd)÷2
            @inbounds for j ∈ max(i-order_a, 0):i_end
                cᵢ += a[i-j] * a[j]
            end
            if iszero(i_odd)
                a_i½ = a[i÷2]
                c[i] += ExactReal(2) * cᵢ + a_i½ ^ ExactReal(2)
            else
                c[i] += ExactReal(2) * cᵢ
            end
        end
    end
    banach_rounding!(c, bound_a², X, rounding_order)
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
    @inbounds c[0] += ExactReal(2) * c₀ + a₀ ^ ExactReal(2)
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
            c[i] += ExactReal(2) * cᵢ + a_i½ ^ ExactReal(2)
            c[-i] += ExactReal(2) * c₋ᵢ + a_neg_i½ ^ ExactReal(2)
        else
            c[i] += ExactReal(2) * cᵢ
            c[-i] += ExactReal(2) * c₋ᵢ
        end
    end
    return c
end

function _add_sqr!(c::Sequence{<:Fourier}, a, bound_a², X)
    rounding_order = banach_rounding_order(bound_a², X)
    if rounding_order > 0
        order_a = order(space(a))
        c₀ = zero(eltype(a))
        @inbounds for j ∈ 1:order_a
            c₀ += a[j] * a[-j]
        end
        @inbounds a₀ = a[0]
        @inbounds c[0] += ExactReal(2) * c₀ + a₀ ^ ExactReal(2)
        @inbounds for i ∈ 1:min(order(space(c)), rounding_order-1)
            cᵢ = c₋ᵢ = zero(eltype(a))
            i½, i_odd = divrem(i, 2)
            @inbounds for j ∈ i½+1:order_a
                cᵢ += a[i-j] * a[j]
                c₋ᵢ += a[j-i] * a[-j]
            end
            if iszero(i_odd)
                a_i½ = a[i½]
                a_neg_i½ = a[-i½]
                c[i] += ExactReal(2) * cᵢ + a_i½ ^ ExactReal(2)
                c[-i] += ExactReal(2) * c₋ᵢ + a_neg_i½ ^ ExactReal(2)
            else
                c[i] += ExactReal(2) * cᵢ
                c[-i] += ExactReal(2) * c₋ᵢ
            end
        end
    end
    banach_rounding!(c, bound_a², X, rounding_order)
    return c
end

# Chebyshev

function _add_sqr!(c::Sequence{Chebyshev}, a)
    order_a = order(space(a))
    c₀ = zero(eltype(a))
    @inbounds for j ∈ 1:order_a
        aⱼ = a[j]
        c₀ += aⱼ ^ ExactReal(2)
    end
    @inbounds a₀ = a[0]
    @inbounds c[0] += ExactReal(2) * c₀ + a₀ ^ ExactReal(2)
    @inbounds for i ∈ 1:order(space(c))
        cᵢ = zero(eltype(a))
        i½, i_odd = divrem(i, 2)
        @inbounds for j ∈ i½+1:order_a
            cᵢ += a[abs(i-j)] * a[j]
        end
        if iszero(i_odd)
            a_i½ = a[i½]
            c[i] += ExactReal(2) * cᵢ + a_i½ ^ ExactReal(2)
        else
            c[i] += ExactReal(2) * cᵢ
        end
    end
    return c
end

function _add_sqr!(c::Sequence{Chebyshev}, a, bound_a², X)
    rounding_order = banach_rounding_order(bound_a², X)
    if rounding_order > 0
        order_a = order(space(a))
        c₀ = zero(eltype(a))
        @inbounds for j ∈ 1:order_a
            aⱼ = a[j]
            c₀ += aⱼ ^ ExactReal(2)
        end
        @inbounds a₀ = a[0]
        @inbounds c[0] += ExactReal(2) * c₀ + a₀ ^ ExactReal(2)
        @inbounds for i ∈ 1:min(order(space(c)), rounding_order-1)
            cᵢ = zero(eltype(a))
            i½, i_odd = divrem(i, 2)
            @inbounds for j ∈ i½+1:order_a
                cᵢ += a[abs(i-j)] * a[j]
            end
            if iszero(i_odd)
                a_i½ = a[i½]
                c[i] += ExactReal(2) * cᵢ + a_i½ ^ ExactReal(2)
            else
                c[i] += ExactReal(2) * cᵢ
            end
        end
    end
    banach_rounding!(c, bound_a², X, rounding_order)
    return c
end
