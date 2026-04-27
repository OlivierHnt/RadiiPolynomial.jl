"""
    AbstractLinearOperator

Abstract type for all linear operators.
"""
abstract type AbstractLinearOperator end

Base.broadcastable(A::AbstractLinearOperator) = Ref(A)

# order, frequency

order(A::AbstractLinearOperator) = (order(domain(A)), order(codomain(A)))
order(A::AbstractLinearOperator, i::Int, j::Int) = (order(domain(A), j), order(codomain(A), i))

frequency(A::AbstractLinearOperator) = (frequency(domain(A)), frequency(codomain(A)))
frequency(A::AbstractLinearOperator, i::Int, j::Int) = (frequency(domain(A), j), frequency(codomain(A), i))

# utilities

function Base.firstindex(A::AbstractLinearOperator, i::Int)
    i == 1 && return _firstindex(codomain(A))
    i == 2 && return _firstindex(domain(A))
    return 1
end

function Base.lastindex(A::AbstractLinearOperator, i::Int)
    i == 1 && return _lastindex(codomain(A))
    i == 2 && return _lastindex(domain(A))
    return 1
end

Base.length(A::AbstractLinearOperator) = length(coefficients(A))

Base.size(A::AbstractLinearOperator) = size(coefficients(A))
Base.size(A::AbstractLinearOperator, i::Int) = size(coefficients(A), i)

Base.iterate(A::AbstractLinearOperator) = iterate(coefficients(A))
Base.iterate(A::AbstractLinearOperator, i) = iterate(coefficients(A), i)

#

domain(A::AbstractLinearOperator, s::VectorSpace) = throw(DomainError((A, s), "cannot infer a domain"))
domain(::AbstractLinearOperator, ::EmptySpace) = EmptySpace()

_coeftype(A::AbstractLinearOperator, dom::VectorSpace) = _coeftype(A, dom, Float64)

function _coeftype(A::AbstractLinearOperator, dom::VectorSpace, ::Type{T}) where {T}
    codom = codomain(A, dom)
    i, j = first(indices(codom)), first(indices(dom))
    x = getcoefficient(A, (codom, i), (dom, j), T)
    CoefType = typeof(x)
    return promote_type(CoefType, T)
end

#

Base.getindex(A::AbstractLinearOperator, (codom, i)::Tuple{VectorSpace,Any}, (dom, j)::Tuple{VectorSpace,Any}) =
    getcoefficient(A, (codom, i), (dom, j), _coeftype(A, dom))

getcoefficient(A::AbstractLinearOperator, (codom, i)::Tuple{VectorSpace,Any}, (dom, j)::Tuple{VectorSpace,Any}, ::Type{T}) where {T} =
    getcoefficient(A, (codom, i), (dom, j)) # getindex(A, (codom, i), (dom, j))

#

abstract type AbstractDiagonalOperator <: AbstractLinearOperator end

domain(::AbstractDiagonalOperator, s::VectorSpace) = s

codomain(::AbstractDiagonalOperator, s::VectorSpace) = s

"""
    LinearOperator{T<:VectorSpace,S<:VectorSpace,R<:AbstractMatrix} <: AbstractLinearOperator

Compactly supported linear operator with effective domain and codomain.

Fields:
- `domain :: T`
- `codomain :: S`
- `coefficients :: R`

Constructors:
- `LinearOperator(::VectorSpace, ::VectorSpace, ::AbstractMatrix)`
- `LinearOperator(coefficients::AbstractMatrix)`: equivalent to `LinearOperator(ScalarSpace()^size(coefficients, 2), ScalarSpace()^size(coefficients, 1), coefficients)`

# Examples

```jldoctest
julia> LinearOperator(Taylor(1), Taylor(1), [1 2 ; 3 4])
LinearOperator : Taylor(1) → Taylor(1) with coefficients Matrix{Int64}:
 1  2
 3  4

julia> LinearOperator(Taylor(2), ScalarSpace(), [1.0 0.5 0.25])
LinearOperator : Taylor(2) → 𝕂 with coefficients Matrix{Float64}:
 1.0  0.5  0.25

julia> LinearOperator([1 2 3 ; 4 5 6])
LinearOperator : 𝕂³ → 𝕂² with coefficients Matrix{Int64}:
 1  2  3
 4  5  6
```
"""
struct LinearOperator{T<:VectorSpace,S<:VectorSpace,R<:AbstractMatrix} <: AbstractLinearOperator
    domain :: T
    codomain :: S
    coefficients :: R
    function LinearOperator{T,S,R}(domain::T, codomain::S, coefficients::R) where {T<:VectorSpace,S<:VectorSpace,R<:AbstractMatrix}
        sz₁, sz₂ = size(coefficients)
        (Base.OneTo(sz₁) == Base.axes(coefficients, 1)) & (Base.OneTo(sz₂) == Base.axes(coefficients, 2)) || return throw(ArgumentError("offset matrices are not supported"))
        dimension_domain = dimension(domain)
        dimension_codomain = dimension(codomain)
        (dimension_codomain == sz₁) & (dimension_domain == sz₂) || return throw(DimensionMismatch("dimensions must match: codomain and domain have dimensions $((dimension_codomain, dimension_domain)), coefficients has size $((sz₁, sz₂))"))
        return new{T,S,R}(domain, codomain, coefficients)
    end
end

LinearOperator(domain::T, codomain::S, coefficients::R) where {T<:VectorSpace,S<:VectorSpace,R<:AbstractMatrix} =
    LinearOperator{T,S,R}(domain, codomain, coefficients)

LinearOperator(coefficient::Number) = LinearOperator(ScalarSpace(), ScalarSpace(), [coefficient;;])
LinearOperator(coefficients::AbstractMatrix) =
    LinearOperator(ScalarSpace()^size(coefficients, 2), ScalarSpace()^size(coefficients, 1), coefficients)

LinearOperator(a::Sequence) = LinearOperator(ScalarSpace(), space(a), reshape(coefficients(a), length(a), 1))

Sequence(A::LinearOperator) = Sequence(codomain(A), vec(coefficients(A)))

domain(A::LinearOperator) = A.domain
function domain(A::LinearOperator, s::VectorSpace)
    _iscompatible(_promote_space(codomain(A), s)...) || return throw(ArgumentError("spaces must be compatible"))
    return domain(A)
end

codomain(A::LinearOperator) = A.codomain
function codomain(A::LinearOperator, s::VectorSpace)
    _iscompatible(_promote_space(domain(A), s)...) || return throw(ArgumentError("spaces must be compatible"))
    return codomain(A)
end

coefficients(A::LinearOperator) = A.coefficients

# to allow a[...] .= f.(...)
Base.@propagate_inbounds Base.Broadcast.dotview(A::LinearOperator, α, β) = view(A, α, β)

# utilities

Base.eltype(A::LinearOperator) = eltype(coefficients(A))
Base.eltype(::Type{<:LinearOperator{<:VectorSpace,<:VectorSpace,T}}) where {T<:AbstractMatrix} = eltype(T)
_coeftype(A::LinearOperator, ::VectorSpace) = eltype(A)
_coeftype(A::LinearOperator, ::VectorSpace, ::Type{T}) where {T} = promote_type(eltype(A), T)

Base.:(==)(A::LinearOperator, B::LinearOperator) =
    codomain(A) == codomain(B) && domain(A) == domain(B) && coefficients(A) == coefficients(B)

Base.iszero(A::LinearOperator) = iszero(coefficients(A))

Base.isapprox(A::LinearOperator, B::LinearOperator; kwargs...) =
    codomain(A) == codomain(B) && domain(A) == domain(B) && isapprox(coefficients(A), coefficients(B); kwargs...)

Base.copy(A::LinearOperator) = LinearOperator(domain(A), codomain(A), copy(coefficients(A)))

Base.similar(A::LinearOperator) = LinearOperator(domain(A), codomain(A), similar(coefficients(A)))
Base.similar(A::LinearOperator, ::Type{T}) where {T} = LinearOperator(domain(A), codomain(A), similar(coefficients(A), T))

Base.zeros(s₁::VectorSpace, s₂::VectorSpace) = LinearOperator(s₁, s₂, zeros(dimension(s₂), dimension(s₁)))
Base.zeros(::Type{T}, s₁::VectorSpace, s₂::VectorSpace) where {T} = LinearOperator(s₁, s₂, zeros(T, dimension(s₂), dimension(s₁)))

Base.ones(s₁::VectorSpace, s₂::VectorSpace) = LinearOperator(s₁, s₂, ones(dimension(s₂), dimension(s₁)))
Base.ones(::Type{T}, s₁::VectorSpace, s₂::VectorSpace) where {T} = LinearOperator(s₁, s₂, ones(T, dimension(s₂), dimension(s₁)))

Base.fill(value, s₁::VectorSpace, s₂::VectorSpace) = LinearOperator(s₁, s₂, fill(value, dimension(s₂), dimension(s₁)))

function Base.fill!(A::LinearOperator, value)
    fill!(coefficients(A), value)
    return A
end

IntervalArithmetic._infer_numtype(A::LinearOperator) = numtype(eltype(A))
IntervalArithmetic._interval_infsup(::Type{T}, A::LinearOperator, B::LinearOperator, d::IntervalArithmetic.Decoration) where {T<:IntervalArithmetic.NumTypes} =
    LinearOperator(IntervalArithmetic._interval_infsup(T, domain(A), domain(B), d), IntervalArithmetic._interval_infsup(T, codomain(A), codomain(B), d), IntervalArithmetic._interval_infsup(T, coefficients(A), coefficients(B), d))

Base.reverse(A::LinearOperator; dims = :) = LinearOperator(domain(A), codomain(A), reverse(coefficients(A); dims = dims))

Base.reverse!(A::LinearOperator; dims = :) = LinearOperator(domain(A), codomain(A), reverse!(coefficients(A); dims = dims))

Base.zero(A::LinearOperator) = zeros(eltype(A), domain(A), codomain(A))
Base.zero(::Type{LinearOperator{T,S,R}}) where {T<:VectorSpace,S<:VectorSpace,R<:AbstractMatrix} = zeros(eltype(R), _zero_space(T), _zero_space(S))

function Base.one(A::LinearOperator)
    CoefType = eltype(A)
    C = zero(A)
    @inbounds for α ∈ indices(domain(A) ∩ codomain(A))
        C[α,α] = one(CoefType)
    end
    return C
end
Base.one(::Type{LinearOperator{T,S,R}}) where {T<:VectorSpace,S<:VectorSpace,R<:AbstractMatrix} = ones(eltype(R), _zero_space(T), _zero_space(S))

Base.float(A::LinearOperator) = LinearOperator(_float_space(domain(A)), _float_space(codomain(A)), float.(coefficients(A)))
Base.big(A::LinearOperator) = LinearOperator(_big_space(domain(A)), _big_space(codomain(A)), big.(coefficients(A)))
IntervalArithmetic.mid(A::LinearOperator) = LinearOperator(_mid_space(domain(A)), _mid_space(codomain(A)), mid.(coefficients(A)))

for f ∈ (:complex, :real, :imag, :conj)
    @eval Base.$f(A::LinearOperator) = LinearOperator(domain(A), codomain(A), $f.(coefficients(A)))
end
Base.conj!(A::LinearOperator) = LinearOperator(domain(A), codomain(A), conj!(coefficients(A)))
Base.complex(A::LinearOperator, B::LinearOperator) =
    LinearOperator(codomain(+, domain(A), domain(B)), codomain(+, codomain(A), codomain(B)), complex.(coefficients(A), coefficients(B)))
Base.complex(::Type{LinearOperator{T,S,R}}) where {T<:VectorSpace,S<:VectorSpace,R<:AbstractMatrix} = LinearOperator{T,S,Matrix{complex(eltype(R))}}

#

Base.transpose(A::LinearOperator) = LinearOperator(codomain(A), domain(A), transpose(coefficients(A)))
Base.adjoint(A::LinearOperator) = LinearOperator(codomain(A), domain(A), adjoint(coefficients(A)))

Base.transpose(a::Sequence) = transpose(LinearOperator(a))
Base.adjoint(a::Sequence) = adjoint(LinearOperator(a))

# getindex, view

Base.@propagate_inbounds function Base.getindex(A::LinearOperator, α, β)
    domain_A = domain(A)
    codomain_A = codomain(A)
    @boundscheck(
        (_checkbounds_indices(α, codomain_A) & _checkbounds_indices(β, domain_A)) ||
        throw(BoundsError((indices(codomain_A), indices(domain_A)), (α, β)))
        )
    return getindex(coefficients(A), _findposition(α, codomain_A), _findposition(β, domain_A))
end

Base.@propagate_inbounds function Base.view(A::LinearOperator, α, β)
    domain_A = domain(A)
    codomain_A = codomain(A)
    @boundscheck(
        (_checkbounds_indices(α, codomain_A) & _checkbounds_indices(β, domain_A)) ||
        throw(BoundsError((indices(codomain_A), indices(domain_A)), (α, β)))
        )
    return view(coefficients(A), _findposition(α, codomain_A), _findposition(β, domain_A))
end

Base.getindex(::LinearOperator, ::VectorSpace, β) = error()
Base.getindex(::LinearOperator, α, ::VectorSpace) = error()
Base.@propagate_inbounds function Base.getindex(A::LinearOperator, α::VectorSpace, β::VectorSpace)
    domain_A = domain(A)
    codomain_A = codomain(A)
    @boundscheck(
        (_checkbounds_indices(α, codomain_A) & _checkbounds_indices(β, domain_A)) ||
        throw(BoundsError((codomain_A, domain_A), (α, β)))
        )
    return LinearOperator(β, α, getindex(coefficients(A), _findposition(α, codomain_A), _findposition(β, domain_A))) # project(A, β, α)
end

Base.view(::LinearOperator, ::VectorSpace, β) = error()
Base.view(::LinearOperator, α, ::VectorSpace) = error()
Base.@propagate_inbounds function Base.view(A::LinearOperator, α::VectorSpace, β::VectorSpace)
    domain_A = domain(A)
    codomain_A = codomain(A)
    @boundscheck(
        (_checkbounds_indices(α, codomain_A) & _checkbounds_indices(β, domain_A)) ||
        throw(BoundsError((codomain_A, domain_A), (α, β)))
        )
    return LinearOperator(β, α, view(coefficients(A), _findposition(α, codomain_A), _findposition(β, domain_A)))
end

# setindex!

Base.@propagate_inbounds function Base.setindex!(A::LinearOperator, x, α, β)
    domain_A = domain(A)
    codomain_A = codomain(A)
    @boundscheck(
        (_checkbounds_indices(α, codomain_A) & _checkbounds_indices(β, domain_A)) ||
        throw(BoundsError((indices(codomain_A), indices(domain_A)), (α, β)))
        )
    setindex!(coefficients(A), x, _findposition(α, codomain_A), _findposition(β, domain_A))
    return A
end

# assume compatibility

getcoefficient(A::LinearOperator, (codom, α)::Tuple{VectorSpace,Any}, (dom, β)::Tuple{VectorSpace,Any}) =
    _getcoefficient(A, α, β)

function getcoefficient(A::LinearOperator{<:SymmetricSpace,<:SymmetricSpace}, (codom, α)::Tuple{SymmetricSpace,Any}, (dom, β)::Tuple{SymmetricSpace,Any})
    (symmetry(domain(A)) == symmetry(dom)) & (symmetry(codomain(A)) == symmetry(codom)) || return throw(DomainError(((symmetry(domain(A)), symmetry(dom)), (symmetry(codomain(A)), symmetry(codom))), "symmetries must be equal"))
    return _getcoefficient(A, α, β)
end

function _getcoefficient(A::LinearOperator, α, β)
    _checkbounds_indices(α, codomain(A)) & _checkbounds_indices(β, domain(A)) || return zero(eltype(A))
    return A[α,β]
end

getcoefficient(A::LinearOperator{<:NoSymSpace,<:NoSymSpace}, (codom, α)::Tuple{SymmetricSpace,Any}, (dom, β)::Tuple{SymmetricSpace,Any}) =
    _sym_getcoefficient(A, dom, codom, α, β)

function _sym_getcoefficient(A::LinearOperator, dom::SymmetricSpace, codom::SymmetricSpace, α, β)
    v = zero(eltype(A))
    orbit_α = _orbit(symmetry(codom), α)
    for l ∈ _orbit(symmetry(dom), β), k ∈ orbit_α
        _checkbounds_indices(k, desymmetrize(codom)) || continue
        _, factor_k = _unsafe_get_representative_and_action(codom, k)
        _checkbounds_indices(l, desymmetrize(dom)) || continue
        _, factor_l = _unsafe_get_representative_and_action(dom, l)
        v += factor_l * (A[k,l] / factor_k) / exact(length(orbit_α))
    end
    return v
end

getcoefficient(A::LinearOperator{<:SymmetricSpace,<:SymmetricSpace}, (codom, α)::Tuple{NoSymSpace,Any}, (dom, β)::Tuple{NoSymSpace,Any}) =
    _desym_getcoefficient(A, α, β)

function _desym_getcoefficient(A::LinearOperator{<:SymmetricSpace,<:SymmetricSpace}, α, β)
    CoefType = complex(eltype(A))
    _checkbounds_indices(α, desymmetrize(codomain(A))) & _checkbounds_indices(β, desymmetrize(domain(A))) || return zero(CoefType)
    k0, factor_α = _unsafe_get_representative_and_action(codomain(A), α)
    l0, factor_β = _unsafe_get_representative_and_action(domain(A), β)
    _checkbounds_indices(k0, codomain(A)) & _checkbounds_indices(l0, domain(A)) || return zero(CoefType)
    return @inbounds convert(CoefType, factor_α * (A[k0,l0] / factor_β) / exact(length(_orbit(symmetry(domain(A)), l0))))
end

getcoefficient(A::LinearOperator{<:SymmetricSpace,<:VectorSpace}, (codom, α)::Tuple{VectorSpace,Any}, (dom, β)::Tuple{NoSymSpace,Any}) =
    _desym_getcoefficient(A, α, β)

function _desym_getcoefficient(A::LinearOperator{<:SymmetricSpace,<:VectorSpace}, α, β)
    CoefType = complex(eltype(A))
    _checkbounds_indices(α, codomain(A)) & _checkbounds_indices(β, desymmetrize(domain(A))) || return zero(CoefType)
    l0, factor_β = _unsafe_get_representative_and_action(domain(A), β)
    _checkbounds_indices(l0, domain(A)) || return zero(CoefType)
    return @inbounds convert(CoefType, (A[α,l0] / factor_β) / exact(length(_orbit(symmetry(domain(A)), l0))))
end

getcoefficient(A::LinearOperator{<:VectorSpace,<:SymmetricSpace}, (codom, α)::Tuple{NoSymSpace,Any}, (dom, β)::Tuple{VectorSpace,Any}) =
    _desym_getcoefficient(A, α, β)

function _desym_getcoefficient(A::LinearOperator{<:VectorSpace,<:SymmetricSpace}, α, β)
    CoefType = complex(eltype(A))
    _checkbounds_indices(α, codomain(A)) & _checkbounds_indices(β, domain(A)) || return zero(CoefType)
    k0, factor_α = _unsafe_get_representative_and_action(codomain(A), α)
    _checkbounds_indices(k0, codomain(A)) || return zero(CoefType)
    return @inbounds convert(CoefType, factor_α * A[k0,β])
end

#

"""
    eachblock(A::LinearOperator{<:CartesianSpace,<:CartesianSpace})

Create a generator whose iterates yield each [`LinearOperator`](@ref) composing the block operator.

# Examples

```jldoctest
julia> A = LinearOperator(Taylor(1)^2, Taylor(1)^2, [i+j for i = 1:4, j = 1:4])
LinearOperator : Taylor(1)² → Taylor(1)² with coefficients Matrix{Int64}:
 2  3  4  5
 3  4  5  6
 4  5  6  7
 5  6  7  8

julia> m = eachblock(A)
Base.Generator{Base.Iterators.ProductIterator{Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}}, RadiiPolynomial.var"#eachblock##2#eachblock##3"{LinearOperator{CartesianPower{Taylor}, CartesianPower{Taylor}, Matrix{Int64}}}}(RadiiPolynomial.var"#eachblock##2#eachblock##3"{LinearOperator{CartesianPower{Taylor}, CartesianPower{Taylor}, Matrix{Int64}}}(LinearOperator(Taylor(1)², Taylor(1)², [2 3 4 5; 3 4 5 6; 4 5 6 7; 5 6 7 8])), Base.Iterators.ProductIterator{Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}}((Base.OneTo(2), Base.OneTo(2))))

julia> [v for v = m]
2×2 Matrix{LinearOperator{Taylor, Taylor, SubArray{Int64, 2, Matrix{Int64}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false}}}:
 [2 3; 3 4]  [4 5; 5 6]
 [4 5; 5 6]  [6 7; 7 8]
```
"""
eachblock(A::LinearOperator{<:CartesianSpace,<:CartesianSpace}) =
    (@inbounds(block(A, i, j)) for i ∈ Base.OneTo(nspaces(codomain(A))), j ∈ Base.OneTo(nspaces(domain(A))))
eachblock(A::LinearOperator{<:CartesianSpace,<:VectorSpace}) =
    (@inbounds(block(A, j)) for i ∈ Base.OneTo(1), j ∈ Base.OneTo(nspaces(domain(A))))
eachblock(A::LinearOperator{<:VectorSpace,<:CartesianSpace}) =
    (@inbounds(block(A, i)) for i ∈ Base.OneTo(nspaces(codomain(A))), j ∈ Base.OneTo(1))

"""
    block(a::LinearOperator{<:CartesianSpace,{<:CartesianSpace})

Return the collection of blocks composing the linear operator.

# Examples

```jldoctest
julia> A = LinearOperator(Taylor(1)^2, Taylor(1)^2, [i+j for i = 1:4, j = 1:4])
LinearOperator : Taylor(1)² → Taylor(1)² with coefficients Matrix{Int64}:
 2  3  4  5
 3  4  5  6
 4  5  6  7
 5  6  7  8

julia> block(A)
2×2 Matrix{LinearOperator{Taylor, Taylor, SubArray{Int64, 2, Matrix{Int64}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false}}}:
 [2 3; 3 4]  [4 5; 5 6]
 [4 5; 5 6]  [6 7; 7 8]
```
"""
block(A::LinearOperator{<:CartesianSpace,<:CartesianSpace}) = collect(eachblock(A))
block(A::LinearOperator{<:CartesianSpace,<:VectorSpace}) = collect(eachblock(A))
block(A::LinearOperator{<:VectorSpace,<:CartesianSpace}) = collect(eachblock(A))

"""
    block(A::LinearOperator{<:CartesianSpace,<:CartesianSpace}, i, j)

Return the ``(i,j)``-th [`LinearOperator`](@ref) composing the block operator.

# Examples

```jldoctest
julia> A = LinearOperator(Taylor(1)^2, Taylor(1)^2, [i+j for i = 1:4, j = 1:4])
LinearOperator : Taylor(1)² → Taylor(1)² with coefficients Matrix{Int64}:
 2  3  4  5
 3  4  5  6
 4  5  6  7
 5  6  7  8

julia> block(A, 1, 1)
LinearOperator : Taylor(1) → Taylor(1) with coefficients SubArray{Int64, 2, Matrix{Int64}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false}:
 2  3
 3  4

julia> block(A, 1, 2)
LinearOperator : Taylor(1) → Taylor(1) with coefficients SubArray{Int64, 2, Matrix{Int64}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false}:
 4  5
 5  6
```
"""
Base.@propagate_inbounds block(A::LinearOperator{<:CartesianSpace,<:CartesianSpace}, i, j) =
    LinearOperator(domain(A)[j], codomain(A)[i], view(coefficients(A), _component_findposition(i, codomain(A)), _component_findposition(j, domain(A))))

Base.@propagate_inbounds block(A::LinearOperator{<:CartesianSpace,<:VectorSpace}, j) =
    LinearOperator(domain(A)[j], codomain(A), view(coefficients(A), :, _component_findposition(j, domain(A))))

Base.@propagate_inbounds block(A::LinearOperator{<:VectorSpace,<:CartesianSpace}, i) =
    LinearOperator(domain(A), codomain(A)[i], view(coefficients(A), _component_findposition(i, codomain(A)), :))

# Base.@propagate_inbounds function block(A::LinearOperator{<:CartesianSpace,<:CartesianSpace}, k)
#     n = nspaces(codomain(A))
#     j, i = divrem(k, n)
#     if iszero(i)
#         return block(A, n, j)
#     else
#         return block(A, i, 1+j)
#     end
# end

# Base.eachcol(A::LinearOperator{<:CartesianSpace,<:CartesianSpace}) =
#     (@inbounds(block(A, :, j)) for j ∈ Base.OneTo(nspaces(domain(A))))
# Base.eachrow(A::LinearOperator{<:CartesianSpace,<:CartesianSpace}) =
#     (@inbounds(block(A, i, :)) for i ∈ Base.OneTo(nspaces(codomain(A))))

# promotion

# Base.convert(::Type{LinearOperator{T₁,S₁,R₁}}, A::LinearOperator{T₂,S₂,R₂}) where {T₁,S₁,R₁,T₂,S₂,R₂} =
#     LinearOperator{T₁,S₁,R₁}(convert(T₁, domain(A)), convert(S₁, codomain(A)), convert(R₁, coefficients(A)))

# Base.promote_rule(::Type{LinearOperator{T₁,S₁,R₁}}, ::Type{LinearOperator{T₂,S₂,R₂}}) where {T₁,S₁,R₁,T₂,S₂,R₂} =
#     LinearOperator{promote_type(T₁, T₂), promote_type(S₁, S₂), promote_type(R₁, R₂)}

# show

function Base.show(io::IO, ::MIME"text/plain", A::LinearOperator)
    println(io, "LinearOperator : ", _prettystring(domain(A), true), " → ", _prettystring(codomain(A), true), " with coefficients ", typeof(coefficients(A)), ":")
    return Base.print_array(io, coefficients(A))
end

function Base.show(io::IO, A::LinearOperator)
    get(io, :compact, false) && return show(io, coefficients(A))
    return print(io, "LinearOperator(", domain(A), ", ", codomain(A), ", ", coefficients(A), ")")
end





#

struct UniformScalingOperator{T<:Number} <: AbstractLinearOperator
    λ :: T
end

UniformScalingOperator() = UniformScalingOperator(true)
UniformScalingOperator(J::UniformScaling) = UniformScalingOperator(J.λ)

domain(::UniformScalingOperator, s::VectorSpace) = s
domain(::UniformScalingOperator, ::EmptySpace) = EmptySpace() # needed to resolve method ambiguity
domain(J::UniformScaling, s::VectorSpace) = domain(UniformScalingOperator(J), s)

codomain(::UniformScalingOperator, s::VectorSpace) = s
codomain(J::UniformScaling, s::VectorSpace) = codomain(UniformScalingOperator(J), s)

Base.eltype(::UniformScalingOperator{T}) where {T<:Number} = T
Base.eltype(::Type{UniformScalingOperator{T}}) where {T<:Number} = T
_coeftype(A::UniformScalingOperator, ::VectorSpace) = eltype(A)
_coeftype(A::UniformScalingOperator, ::VectorSpace, ::Type{T}) where {T} = promote_type(eltype(A), T)

Base.zero(J::UniformScalingOperator) = UniformScalingOperator(zero(J.λ))
Base.zero(::Type{T}) where {T<:UniformScalingOperator} = UniformScalingOperator(zero(eltype(T)))
Base.one(J::UniformScalingOperator) = UniformScalingOperator(one(J.λ))
Base.one(::Type{T}) where {T<:UniformScalingOperator} = UniformScalingOperator(one(eltype(T)))

Base.zero(::AbstractLinearOperator) = UniformScalingOperator(exact(false))
Base.zero(::Type{<:AbstractLinearOperator}) = UniformScalingOperator(exact(false))
Base.one(::AbstractLinearOperator) = UniformScalingOperator(exact(true))
Base.one(::Type{<:AbstractLinearOperator}) = UniformScalingOperator(exact(true))

IntervalArithmetic._infer_numtype(J::UniformScalingOperator) = numtype(eltype(J))
IntervalArithmetic._interval_infsup(::Type{T}, J::UniformScalingOperator, H::UniformScalingOperator, d::IntervalArithmetic.Decoration) where {T<:IntervalArithmetic.NumTypes} =
    UniformScalingOperator(IntervalArithmetic._interval_infsup(T, J.λ, H.λ, d))

Base.promote_rule(::Type{<:UniformScaling}, ::Type{<:AbstractLinearOperator}) = AbstractLinearOperator
Base.promote_rule(::Type{<:AbstractLinearOperator}, ::Type{<:UniformScaling}) = AbstractLinearOperator
Base.promote_rule(::Type{UniformScaling{T}}, ::Type{UniformScalingOperator{S}}) where {T<:Number,S<:Number} = UniformScalingOperator{promote_type(T,S)}
Base.promote_rule(::Type{UniformScalingOperator{T}}, ::Type{UniformScaling{S}}) where {T<:Number,S<:Number} = UniformScalingOperator{promote_type(T,S)}

Base.convert(::Type{AbstractLinearOperator}, J::UniformScaling) = UniformScalingOperator(J.λ)
Base.convert(::Type{UniformScalingOperator{T}}, J::UniformScaling{S}) where {T<:Number,S<:Number} = UniformScalingOperator(convert(promote_type(T,S), J.λ))





# Generic operators used to compose other operators

struct Add{T<:AbstractLinearOperator,S<:AbstractLinearOperator} <: AbstractLinearOperator
    A :: T
    B :: S
end

struct Negate{T<:AbstractLinearOperator} <: AbstractLinearOperator
    A :: T
end

domain(S::Add, s::VectorSpace) = _union(domain(S.A, s), domain(S.B, s))
domain(S::Negate, s::VectorSpace) = domain(S.A, s)
domain(::Add, ::EmptySpace) = EmptySpace() # needed to resolve method ambiguity
domain(::Negate, ::EmptySpace) = EmptySpace() # needed to resolve method ambiguity
# same as `union` but propagate empty space
_union(::EmptySpace, ::VectorSpace) = EmptySpace()
_union(::VectorSpace, ::EmptySpace) = EmptySpace()
_union(::EmptySpace, ::EmptySpace) = EmptySpace()
_union(s₁::VectorSpace, s₂::VectorSpace) = s₁ ∪ s₂

codomain(S::Add, s::VectorSpace) = codomain(S.A, s) ∪ codomain(S.B, s)
codomain(S::Negate, s::VectorSpace) = codomain(S.A, s)

_coeftype(S::Add, s::VectorSpace, ::Type{T}) where {T} = promote_type(_coeftype(S.A, s, T), _coeftype(S.B, s, T))
_coeftype(S::Negate, s::VectorSpace, ::Type{T}) where {T} = _coeftype(S.A, s, T)

Base.show(io::IO, ::MIME"text/plain", S::Add) = print(io, S.A, " + ", S.B)
Base.show(io::IO, ::MIME"text/plain", S::Add{<:AbstractLinearOperator,<:Negate}) = print(io, S.A, " - ", S.B.A)
Base.show(io::IO, ::MIME"text/plain", S::Negate) = print(io, "-", S.A)

#

struct ComposedOperator{T<:AbstractLinearOperator,S<:AbstractLinearOperator} <: AbstractLinearOperator
    outer :: T
    inner :: S
end

Base.:∘(A::AbstractLinearOperator, B::AbstractLinearOperator) = ComposedOperator(A, B)
Base.:∘(A::AbstractLinearOperator, J::UniformScaling) = ComposedOperator(A, UniformScalingOperator(J))
Base.:∘(J::UniformScaling, A::AbstractLinearOperator) = ComposedOperator(UniformScalingOperator(J), A)

domain(A::ComposedOperator, s::VectorSpace) = domain(A.inner, domain(A.outer, s))

codomain(A::ComposedOperator, s::VectorSpace) = codomain(A.outer, codomain(A.inner, s))

_coeftype(S::ComposedOperator, s::VectorSpace, ::Type{T}) where {T} = _coeftype(S.outer, codomain(S.inner, s), _coeftype(S.inner, s, T))
