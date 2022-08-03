"""
    LinearOperator{T<:VectorSpace,S<:VectorSpace,R<:AbstractMatrix}

Compactly supported linear operator with effective domain and codomain.

Fields:
- `domain :: T`
- `codomain :: S`
- `coefficients :: R`
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
Base.eltype(::Type{LinearOperator{T,S,R}}) where {T,S,R} = eltype(R)

# getindex, view, setindex!

Base.@propagate_inbounds Base.getindex(A::LinearOperator, α, β) =
    getindex(coefficients(A), _findposition(α, codomain(A)), _findposition(β, domain(A)))

Base.@propagate_inbounds Base.view(A::LinearOperator, α, β) =
    view(coefficients(A), _findposition(α, codomain(A)), _findposition(β, domain(A)))

Base.@propagate_inbounds Base.setindex!(A::LinearOperator, x, α, β) =
    setindex!(coefficients(A), x, _findposition(α, codomain(A)), _findposition(β, domain(A)))

# ==, iszero, isapprox

Base.:(==)(A::LinearOperator, B::LinearOperator) =
    codomain(A) == codomain(B) && domain(A) == domain(B) && coefficients(A) == coefficients(B)

Base.iszero(A::LinearOperator) = iszero(coefficients(A))

Base.isapprox(A::LinearOperator, B::LinearOperator; kwargs...) =
    codomain(A) == codomain(B) && domain(A) == domain(B) && isapprox(coefficients(A), coefficients(B); kwargs...)

# copy, similar

Base.copy(A::LinearOperator) = LinearOperator(domain(A), codomain(A), copy(coefficients(A)))

Base.similar(A::LinearOperator) = LinearOperator(domain(A), codomain(A), similar(coefficients(A)))
Base.similar(A::LinearOperator, ::Type{T}) where {T} = LinearOperator(domain(A), codomain(A), similar(coefficients(A), T))

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

# promotion

Base.convert(::Type{T}, A::T) where {T<:LinearOperator} = A
Base.convert(::Type{LinearOperator{T₁,S₁,R₁}}, A::LinearOperator{T₂,S₂,R₂}) where {T₁,S₁,R₁,T₂,S₂,R₂} =
    LinearOperator{T₁,S₁,R₁}(convert(T₁, domain(A)), convert(S₁, codomain(A)), convert(R₁, coefficients(A)))

Base.promote_rule(::Type{T}, ::Type{T}) where {T<:LinearOperator} = T
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

# show

function Base.show(io::IO, ::MIME"text/plain", A::LinearOperator)
    println(io, "LinearOperator : ", string_space(domain(A)), " → ", string_space(codomain(A)), " with coefficients ", typeof(coefficients(A)), ":")
    Base.print_array(io, coefficients(A))
end
