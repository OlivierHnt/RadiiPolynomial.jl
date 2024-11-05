"""
    LinearOperator{T<:VectorSpace,S<:VectorSpace,R<:AbstractMatrix}

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
struct LinearOperator{T<:VectorSpace,S<:VectorSpace,R<:AbstractMatrix}
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

LinearOperator(coefficients::AbstractMatrix) =
    LinearOperator(ParameterSpace()^size(coefficients, 2), ParameterSpace()^size(coefficients, 1), coefficients)

LinearOperator(a::Sequence) = LinearOperator(ParameterSpace(), space(a), reshape(coefficients(a), length(a), 1))

Sequence(A::LinearOperator) = Sequence(codomain(A), vec(coefficients(A)))

domain(A::LinearOperator) = A.domain

codomain(A::LinearOperator) = A.codomain

coefficients(A::LinearOperator) = A.coefficients

# order, frequency

order(A::LinearOperator) = (order(domain(A)), order(codomain(A)))
order(A::LinearOperator, i::Int, j::Int) = (order(domain(A), j), order(codomain(A), i))

frequency(A::LinearOperator) = (frequency(domain(A)), frequency(codomain(A)))
frequency(A::LinearOperator, i::Int, j::Int) = (frequency(domain(A), j), frequency(codomain(A), i))

# utilities

function Base.firstindex(A::LinearOperator, i::Int)
    i == 1 && return _firstindex(codomain(A))
    i == 2 && return _firstindex(domain(A))
    return 1
end

function Base.lastindex(A::LinearOperator, i::Int)
    i == 1 && return _lastindex(codomain(A))
    i == 2 && return _lastindex(domain(A))
    return 1
end

Base.length(A::LinearOperator) = length(coefficients(A))

Base.size(A::LinearOperator) = size(coefficients(A))
Base.size(A::LinearOperator, i::Int) = size(coefficients(A), i)

Base.iterate(A::LinearOperator) = iterate(coefficients(A))
Base.iterate(A::LinearOperator, i) = iterate(coefficients(A), i)

Base.eltype(A::LinearOperator) = eltype(coefficients(A))
Base.eltype(::Type{<:LinearOperator{<:VectorSpace,<:VectorSpace,T}}) where {T<:AbstractMatrix} = eltype(T)

# getindex, view, setindex!

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

# ==, iszero, isapprox

Base.:(==)(A::LinearOperator, B::LinearOperator) =
    codomain(A) == codomain(B) && domain(A) == domain(B) && coefficients(A) == coefficients(B)

Base.iszero(A::LinearOperator) = iszero(coefficients(A))

Base.isapprox(A::LinearOperator, B::LinearOperator; kwargs...) =
    codomain(A) == codomain(B) && domain(A) == domain(B) && isapprox(coefficients(A), coefficients(B); kwargs...)

# copy, similar, zeros, ones, fill, fill!

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

# zero, one

function Base.zero(A::LinearOperator)
    domain_A = domain(A)
    codomain_A = codomain(A)
    CoefType = eltype(A)
    C = LinearOperator(domain_A, codomain_A, Matrix{CoefType}(undef, dimension(codomain_A), dimension(domain_A)))
    coefficients(C) .= zero(CoefType)
    return C
end

function Base.one(A::LinearOperator)
    CoefType = eltype(A)
    C = zero(A)
    @inbounds for α ∈ indices(domain(A) ∩ codomain(A))
        C[α,α] = one(CoefType)
    end
    return C
end

# float, complex, real, imag, conj, conj!

for f ∈ (:float, :complex, :real, :imag, :conj, :conj!)
    @eval Base.$f(A::LinearOperator) = LinearOperator(domain(A), codomain(A), $f(coefficients(A)))
end

#

adjoint(A::LinearOperator) = LinearOperator(codomain(A), domain(A), adjoint(coefficients(A)))
adjoint(a::Sequence) = adjoint(LinearOperator(a))

# promotion

Base.convert(::Type{LinearOperator{T₁,S₁,R₁}}, A::LinearOperator{T₂,S₂,R₂}) where {T₁,S₁,R₁,T₂,S₂,R₂} =
    LinearOperator{T₁,S₁,R₁}(convert(T₁, domain(A)), convert(S₁, codomain(A)), convert(R₁, coefficients(A)))

Base.promote_rule(::Type{LinearOperator{T₁,S₁,R₁}}, ::Type{LinearOperator{T₂,S₂,R₂}}) where {T₁,S₁,R₁,T₂,S₂,R₂} =
    LinearOperator{promote_type(T₁, T₂), promote_type(S₁, S₂), promote_type(R₁, R₂)}

# Cartesian spaces

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

# show

function Base.show(io::IO, ::MIME"text/plain", A::LinearOperator)
    println(io, "LinearOperator : ", _prettystring(domain(A)), " → ", _prettystring(codomain(A)), " with coefficients ", typeof(coefficients(A)), ":")
    return Base.print_array(io, coefficients(A))
end

function Base.show(io::IO, A::LinearOperator)
    get(io, :compact, false) && return show(io, coefficients(A))
    return print(io, "LinearOperator(", domain(A), ", ", codomain(A), ", ", coefficients(A), ")")
end
