"""
    AbstractLinearOperator

Abstract type for all linear operators.
"""
abstract type AbstractLinearOperator end

# order, frequency

order(A::AbstractLinearOperator) = (order(domain(A)), order(codomain(A)))
order(A::AbstractLinearOperator, i::Int, j::Int) = (order(domain(A), j), order(codomain(A), i))

frequency(A::AbstractLinearOperator) = (frequency(domain(A)), frequency(codomain(A)))
frequency(A::AbstractLinearOperator, i::Int, j::Int) = (frequency(domain(A), j), frequency(codomain(A), i))

Base.:*(A::AbstractLinearOperator, B::AbstractLinearOperator) = A ∘ B

Base.:+(A::AbstractLinearOperator) = A

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

# getindex, view

Base.@propagate_inbounds function Base.getindex(A::AbstractLinearOperator, α, β)
    domain_A = domain(A)
    codomain_A = codomain(A)
    @boundscheck(
        (_checkbounds_indices(α, codomain_A) & _checkbounds_indices(β, domain_A)) ||
        throw(BoundsError((indices(codomain_A), indices(domain_A)), (α, β)))
        )
    return getindex(coefficients(A), _findposition(α, codomain_A), _findposition(β, domain_A))
end

Base.@propagate_inbounds function Base.view(A::AbstractLinearOperator, α, β)
    domain_A = domain(A)
    codomain_A = codomain(A)
    @boundscheck(
        (_checkbounds_indices(α, codomain_A) & _checkbounds_indices(β, domain_A)) ||
        throw(BoundsError((indices(codomain_A), indices(domain_A)), (α, β)))
        )
    return view(coefficients(A), _findposition(α, codomain_A), _findposition(β, domain_A))
end

#

"""
    LinearOperator{T<:VectorSpace,S<:VectorSpace,R<:AbstractMatrix} <: AbstractLinearOperator

Compactly supported linear operator with effective domain and codomain.

Fields:
- `domain :: T`
- `codomain :: S`
- `coefficients :: R`

Constructors:
- `LinearOperator(::VectorSpace, ::VectorSpace, ::AbstractMatrix)`
- `LinearOperator(coefficients::AbstractMatrix)`: equivalent to `LinearOperator(ParameterSpace()^size(coefficients, 2), ParameterSpace()^size(coefficients, 1), coefficients)`

# Examples

```jldoctest
julia> LinearOperator(Taylor(1), Taylor(1), [1 2 ; 3 4])
LinearOperator : Taylor(1) → Taylor(1) with coefficients Matrix{Int64}:
 1  2
 3  4

julia> LinearOperator(Taylor(2), ParameterSpace(), [1.0 0.5 0.25])
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

LinearOperator(coefficient::Number) = LinearOperator(ParameterSpace(), ParameterSpace(), [coefficient;;])
LinearOperator(coefficients::AbstractMatrix) =
    LinearOperator(ParameterSpace()^size(coefficients, 2), ParameterSpace()^size(coefficients, 1), coefficients)

LinearOperator(a::Sequence) = LinearOperator(ParameterSpace(), space(a), reshape(coefficients(a), length(a), 1))

Sequence(A::LinearOperator) = Sequence(codomain(A), vec(coefficients(A)))

domain(A::LinearOperator) = A.domain

codomain(A::LinearOperator) = A.codomain
function codomain(A::LinearOperator, s::VectorSpace)
    _iscompatible(domain(A), s) || return throw(ArgumentError("spaces must be compatible"))
    return codomain(A)
end

coefficients(A::LinearOperator) = A.coefficients

# utilities

Base.eltype(A::LinearOperator) = eltype(coefficients(A))
Base.eltype(::Type{<:LinearOperator{<:VectorSpace,<:VectorSpace,T}}) where {T<:AbstractMatrix} = eltype(T)

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

IntervalArithmetic.interval(::Type{T}, A::LinearOperator, d::IntervalArithmetic.Decoration = com; format::Symbol = :infsup) where {T} =
    LinearOperator(interval(T, domain(A)), interval(T, codomain(A)), interval(T, coefficients(A), d; format = format))
IntervalArithmetic.interval(A::LinearOperator, d::IntervalArithmetic.Decoration = com; format::Symbol = :infsup) =
    LinearOperator(interval(domain(A)), interval(codomain(A)), interval(coefficients(A), d; format = format))
IntervalArithmetic.interval(::Type{T}, A::LinearOperator, d::AbstractMatrix{IntervalArithmetic.Decoration}; format::Symbol = :infsup) where {T} =
    LinearOperator(interval(T, domain(A)), interval(T, codomain(A)), interval(T, coefficients(A), d; format = format))
IntervalArithmetic.interval(A::LinearOperator, d::AbstractMatrix{IntervalArithmetic.Decoration}; format::Symbol = :infsup) =
    LinearOperator(interval(domain(A)), interval(codomain(A)), interval(coefficients(A), d; format = format))

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

#

Base.eachcol(A::LinearOperator{<:CartesianSpace,<:CartesianSpace}) =
    (@inbounds(component(A, :, j)) for j ∈ Base.OneTo(nspaces(domain(A))))
Base.eachrow(A::LinearOperator{<:CartesianSpace,<:CartesianSpace}) =
    (@inbounds(component(A, i, :)) for i ∈ Base.OneTo(nspaces(codomain(A))))

eachcomponent(A::LinearOperator{<:CartesianSpace,<:CartesianSpace}) =
    (@inbounds(component(A, i, j)) for i ∈ Base.OneTo(nspaces(codomain(A))), j ∈ Base.OneTo(nspaces(domain(A))))
eachcomponent(A::LinearOperator{<:CartesianSpace,<:VectorSpace}) =
    (@inbounds(component(A, j)) for i ∈ Base.OneTo(1), j ∈ Base.OneTo(nspaces(domain(A))))
eachcomponent(A::LinearOperator{<:VectorSpace,<:CartesianSpace}) =
    (@inbounds(component(A, i)) for i ∈ Base.OneTo(nspaces(codomain(A))), j ∈ Base.OneTo(1))

Base.@propagate_inbounds component(A::LinearOperator{<:CartesianSpace,<:CartesianSpace}, i, j) =
    LinearOperator(domain(A)[j], codomain(A)[i], view(coefficients(A), _component_findposition(i, codomain(A)), _component_findposition(j, domain(A))))

Base.@propagate_inbounds component(A::LinearOperator{<:CartesianSpace,<:VectorSpace}, j) =
    LinearOperator(domain(A)[j], codomain(A), view(coefficients(A), :, _component_findposition(j, domain(A))))

Base.@propagate_inbounds component(A::LinearOperator{<:VectorSpace,<:CartesianSpace}, i) =
    LinearOperator(domain(A), codomain(A)[i], view(coefficients(A), _component_findposition(i, codomain(A)), :))

Base.@propagate_inbounds function component(A::LinearOperator{<:CartesianSpace,<:CartesianSpace}, k)
    n = nspaces(codomain(A))
    j, i = divrem(k, n)
    if iszero(i)
        return component(A, n, j)
    else
        return component(A, i, 1+j)
    end
end

# promotion

Base.convert(::Type{LinearOperator{T₁,S₁,R₁}}, A::LinearOperator{T₂,S₂,R₂}) where {T₁,S₁,R₁,T₂,S₂,R₂} =
    LinearOperator{T₁,S₁,R₁}(convert(T₁, domain(A)), convert(S₁, codomain(A)), convert(R₁, coefficients(A)))

Base.promote_rule(::Type{LinearOperator{T₁,S₁,R₁}}, ::Type{LinearOperator{T₂,S₂,R₂}}) where {T₁,S₁,R₁,T₂,S₂,R₂} =
    LinearOperator{promote_type(T₁, T₂), promote_type(S₁, S₂), promote_type(R₁, R₂)}

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

struct Add{T<:AbstractLinearOperator,S<:AbstractLinearOperator} <: AbstractLinearOperator
    A :: T
    B :: S
end

struct Negate{T<:AbstractLinearOperator} <: AbstractLinearOperator
    A :: T
end

Base.:+(A::AbstractLinearOperator, B::AbstractLinearOperator) = Add(A, B)
Base.:-(A::AbstractLinearOperator) = Negate(A)
Base.:-(A::AbstractLinearOperator, B::AbstractLinearOperator) = A + (-B)

Base.:-(A::Negate) = A.A
Base.:-(A::Add) = -A.A + (-A.B)

codomain(S::Add, s::VectorSpace) = codomain(S.A, s) ∪ codomain(S.B, s)
codomain(S::Negate, s::VectorSpace) = codomain(S.A, s)

Base.show(io::IO, ::MIME"text/plain", S::Add) = print(io, S.A, " + ", S.B)
Base.show(io::IO, ::MIME"text/plain", S::Add{<:AbstractLinearOperator,<:Negate}) = print(io, S.A, " - ", S.B.A)
Base.show(io::IO, ::MIME"text/plain", S::Negate) = print(io, "-", S.A)

#

struct UniformScalingOperator{T<:UniformScaling} <: AbstractLinearOperator
    J :: T
end

UniformScalingOperator(λ::Number=true) = UniformScalingOperator(UniformScaling(λ))

Base.:*(J::UniformScalingOperator, b::AbstractSequence) = J.J.λ * b

codomain(::UniformScalingOperator, s::VectorSpace) = s
codomain(J::UniformScaling, s::VectorSpace) = codomain(UniformScalingOperator(J), s)

Base.eltype(J::UniformScalingOperator) = eltype(J.J)
Base.eltype(::Type{UniformScalingOperator{T}}) where {T<:UniformScaling} = eltype(T)

Base.zero(J::UniformScalingOperator) = UniformScalingOperator(zero(J.J))
Base.zero(::Type{T}) where {T<:UniformScalingOperator} = UniformScalingOperator(zero(eltype(T)))
Base.one(J::UniformScalingOperator) = UniformScalingOperator(one(J.J))
Base.one(::Type{T}) where {T<:UniformScalingOperator} = UniformScalingOperator(one(eltype(T)))

Base.:-(J::UniformScalingOperator) = UniformScalingOperator(-J.J)

Base.:+(A::AbstractLinearOperator, J::UniformScaling) = A + UniformScalingOperator(J)
Base.:+(J::UniformScaling, A::AbstractLinearOperator) = UniformScalingOperator(J) + A
Base.:-(A::AbstractLinearOperator, J::UniformScaling) = A - UniformScalingOperator(J)
Base.:-(J::UniformScaling, A::AbstractLinearOperator) = UniformScalingOperator(J) - A
Base.:*(J::UniformScaling, A::AbstractLinearOperator) = UniformScalingOperator(J) * A
Base.:*(A::AbstractLinearOperator, J::UniformScaling) = A * UniformScalingOperator(J)

#

struct ComposedOperator{T<:AbstractLinearOperator,S<:AbstractLinearOperator} <: AbstractLinearOperator
    outer :: T
    inner :: S
end

Base.:∘(A::AbstractLinearOperator, B::AbstractLinearOperator) = ComposedOperator(A, B)
Base.:∘(A::AbstractLinearOperator, J::UniformScaling) = ComposedOperator(A, UniformScalingOperator(J))
Base.:∘(J::UniformScaling, A::AbstractLinearOperator) = ComposedOperator(UniformScalingOperator(J), A)
