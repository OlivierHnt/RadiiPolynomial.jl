"""
    Functional{T<:SequenceSpace,S<:AbstractVector}

Compactly supported functional with effective domain.

Fields:
- `domain :: T`
- `coefficients :: S`
"""
struct Functional{T<:SequenceSpace,S<:AbstractVector}
    domain :: T
    coefficients :: S
    function Functional{T,S}(domain::T,coefficients::S) where {T<:SequenceSpace,S<:AbstractVector}
        @assert length(domain) == length(coefficients)
        return new{T,S}(domain, coefficients)
    end
end

Functional(domain::T, coefficients::S) where {T<:SequenceSpace,S<:AbstractVector} =
    Functional{T,S}(domain, coefficients)

## utilities

Base.firstindex(A::Functional) = firstindex(A.domain)

Base.lastindex(A::Functional) = lastindex(A.domain)

Base.eachindex(A::Functional) = eachindex(A.domain)

Base.length(A::Functional) = length(A.domain)

Base.size(A::Functional) = tuple(length(A.domain)) # necessary for broadcasting

Base.iterate(A::Functional) = iterate(A.coefficients)
Base.iterate(A::Functional, i) = iterate(A.coefficients, i)

Base.eltype(A::Functional) = eltype(A.coefficients)
Base.eltype(::Type{Functional{T,S}}) where {T<:SequenceSpace,S<:AbstractVector} = eltype(S)

## domain, coefficients, order, frequency

domain(A::Functional) = A.domain
coefficients(A::Functional) = A.coefficients
order(A::Functional) = order(A.domain)
order(A::Functional{<:TensorSpace}, i::Int) = order(A.domain, i)
frequency(A::Functional) = frequency(A.domain)
frequency(A::Functional{<:TensorSpace}, i::Int) = frequency(A.domain, i)

## getindex, view, setindex!

for f ∈ (:getindex, :view)
    @eval begin
        Base.@propagate_inbounds function Base.$f(A::Functional, α)
            @boundscheck(!isindexof(α, A.domain) && throw(BoundsError(A.domain, α)))
            return $f(A.coefficients, _findindex(α, A.domain))
        end

        Base.@propagate_inbounds function Base.$f(A::Functional{TensorSpace{T}}, α::NTuple{N,Int}) where {N,T<:NTuple{N,UnivariateSpace}}
            @boundscheck(!isindexof(α, A.domain) && throw(BoundsError(A.domain, α)))
            return $f(A.coefficients, _findindex(α, A.domain))
        end

        Base.@propagate_inbounds function Base.$f(A::Functional{TensorSpace{T}}, α::Tuple) where {N,T<:NTuple{N,UnivariateSpace}}
            @boundscheck(!isindexof(α, A.domain) && throw(BoundsError(A.domain, α)))
            A_ = reshape(a.coefficients, size(A.domain))
            return $f(A_, _findindex(α, A.domain)...)
        end
    end
end

Base.@propagate_inbounds function Base.setindex!(A::Functional, x, α)
    @boundscheck(!isindexof(α, A.domain) && throw(BoundsError(A.domain, α)))
    return setindex!(A.coefficients, x, _findindex(α, A.domain))
end

Base.@propagate_inbounds function Base.setindex!(A::Functional{TensorSpace{T}}, x, α::NTuple{N,Int}) where {N,T<:NTuple{N,UnivariateSpace}}
    @boundscheck(!isindexof(α, A.domain) && throw(BoundsError(A.domain, α)))
    return setindex!(A.coefficients, x, _findindex(α, A.domain))
end

Base.@propagate_inbounds function Base.setindex!(A::Functional{TensorSpace{T}}, x, α::Tuple) where {N,T<:NTuple{N,UnivariateSpace}}
    @boundscheck(!isindexof(α, A.domain) && throw(BoundsError(A.domain, α)))
    A_ = reshape(A.coefficients, size(A.domain))
    return setindex!(A_, x, _findindex(α, A.domain)...)
end

## project

"""
    project(A::Functional, domain::SequenceSpace)

Return a [`Functional`](@ref) representing `A` with effective `domain`.
"""
function project(A::Functional, domain::SequenceSpace)
    CoefType = eltype(A)
    C = Functional(space, Vector{CoefType}(undef, length(domain)))
    if A.domain == domain
        @. C.coefficients = A.coefficients
        return C
    elseif A.domain ⊆ domain
        C.coefficients .= zero(CoefType)
        @inbounds for α ∈ eachindex(A.domain)
            C[α] = A[α]
        end
        return c
    elseif domain ⊆ A.domain
        @inbounds for α ∈ eachindex(domain)
            C[α] = A[α]
        end
        return C
    else
        C.coefficients .= zero(CoefType)
        @inbounds for α ∈ eachindex(A.domain ∩ domain)
            C[α] = A[α]
        end
        return C
    end
end

## selectdim

"""
    selectdim(a::Functional{TensorSpace{T}}, dim::Int, i::Int) where {N,T<:NTuple{N,UnivariateSpace}}

Return a [`Functional`](@ref) with effective domain `A.domain[1:dim-1] ⊗ A.domain[dim+1:N]` and whose coefficients are a view of all the data `A.coefficients` where the index for dimension `dim` equals `i`.
"""
Base.@propagate_inbounds function Base.selectdim(A::Functional{TensorSpace{T}}, dim::Int, i::Int) where {N,T<:NTuple{N,UnivariateSpace}}
    @boundscheck(!isindexof(i, A.domain[dim]) && throw(BoundsError(A.domain[dim], i)))
    A_ = reshape(A.coefficients, size(A.domain))
    _A_ = selectdim(A_, dim, _findindex(i, A.domain[dim]))
    return Functional(A.domain[1:dim-1] ⊗ A.domain[dim+1:N], vec(_A_))
end

## permutedims

Base.permutedims(A::Functional{<:TensorSpace}, σ::AbstractVector{Int}) =
    Functional(A.domain[σ], vec(permutedims(reshape(A.coefficients, size(A.domain)), σ)))

## ==, iszero, isapprox

Base.:(==)(A::Functional, B::Functional) = A.domain == B.domain && A.coefficients == B.coefficients

Base.iszero(A::Functional) = iszero(A.coefficients)

Base.isapprox(A::Functional, B::Functional; kwargs...) = A.domain == B.domain && isapprox(A.coefficients, B.coefficients; kwargs...)

## copy, similar

Base.copy(A::Functional) = Functional(A.domain, copy(A.coefficients))

Base.similar(A::Functional) = Functional(A.domain, similar(A.coefficients))

## zero, one

Base.zero(A::Functional) = Functional(A.domain, zero.(A.coefficients))

function Base.one(A::Functional)
    B = zero(A)
    @inbounds B[_constant_index(A.domain)] = one(eltype(A))
    return B
end

## float, real, complex, conj, imag, abs, abs2

for f ∈ (:float, :real, :complex, :conj, :imag)
    @eval Base.$f(A::Functional) = Functional(A.domain, $f(A.coefficients))
end

# Base.abs(A::Functional) = Functional(A.domain, abs.(A.coefficients))
# Base.abs2(A::Functional) = Functional(A.domain, abs2.(A.coefficients))

## promotion

Base.convert(::Type{Functional{T₁,S₁}}, A::Functional{T₂,S₂}) where {T₁<:SequenceSpace,S₁<:AbstractVector,T₂<:SequenceSpace,S₂<:AbstractVector} =
    Functional{T₁,S₁}(convert(T₁, A.domain), convert(S₁, A.coefficients))

Base.promote_rule(::Type{Functional{T₁,S₁}}, ::Type{Functional{T₂,S₂}}) where {T₁<:SequenceSpace,S₁<:AbstractVector,T₂<:SequenceSpace,S₂<:AbstractVector} =
    Functional{promote_type(T₁,T₂),promote_type(S₁,S₂)}

## opnorm

function LinearAlgebra.opnorm(A::Functional{Taylor}, ν::Real=1)
    @assert ν > 0
    ν⁻¹ = inv(ν)
    ν⁻ⁱ = one(ν⁻¹)
    @inbounds s = abs(A[0]) * ν⁻ⁱ
    @inbounds for i ∈ 1:order(A)
        ν⁻ⁱ *= ν⁻¹
        s = max(s, abs(A[i]) * ν⁻ⁱ)
    end
    return s
end

function LinearAlgebra.opnorm(A::Functional{<:Fourier}, ν::Real=1)
    @assert ν > 0
    ν⁻¹ = inv(ν)
    ν⁻ⁱ = one(ν⁻¹)
    @inbounds s = abs(A[0]) * ν⁻ⁱ
    @inbounds for i ∈ 1:order(A)
        ν⁻ⁱ *= ν⁻¹
        s = max(s, abs(A[i]) * ν⁻ⁱ, abs(A[-i]) * ν⁻ⁱ)
    end
    return s
end

function LinearAlgebra.opnorm(A::Functional{Chebyshev}, ν::Real=1)
    @assert ν > 0
    ν⁻¹ = inv(ν)
    ν⁻ⁱ = one(ν⁻¹)
    @inbounds s = abs(A[0]) * ν⁻ⁱ
    @inbounds for i ∈ 1:order(A)
        ν⁻ⁱ *= ν⁻¹
        s = max(s, abs(A[i]) * ν⁻ⁱ)
    end
    return s
end

LinearAlgebra.opnorm(A::Functional{TensorSpace{T}}, ν::Real=1) where {N,T<:NTuple{N,UnivariateSpace}} =
    opnorm(A, ntuple(i -> ν, N))

function LinearAlgebra.opnorm(A::Functional{TensorSpace{T}}, ν::NTuple{N,Real}) where {N,T<:NTuple{N,UnivariateSpace}}
    @assert all(νᵢ -> νᵢ > 0, ν)
    A_ = reshape(A.coefficients, size(A.domain))
    return @inbounds _apply_opnorm(A.domain, A_, ν)[1]
end

_apply_opnorm(space::TensorSpace{<:NTuple{N₁,UnivariateSpace}}, A::AbstractArray{T,N₂}, ν::NTuple{N₁,Real}) where {N₁,T,N₂} =
    @inbounds _opnorm(space[1], Val(N₂-N₁+1), _apply_opnorm(Base.tail(space), A, Base.tail(ν)), ν[1])

_apply_opnorm(space::TensorSpace{<:NTuple{2,UnivariateSpace}}, A::AbstractArray{T,N}, ν::NTuple{2,Real}) where {T,N} =
    @inbounds _opnorm(space[1], Val(N-1), _opnorm(space[2], Val(N), A, ν[2]), ν[1])

function _opnorm(space::Taylor, ::Val{D}, A, ν) where {D}
    ν⁻¹ = inv(ν)
    ν⁻ⁱ = one(ν⁻¹)
    @inbounds s = abs.(_selectdim(space, A, D, 0:0)) .* ν⁻ⁱ
    @inbounds for i ∈ 1:order(space)
        ν⁻ⁱ *= ν⁻¹
        Aᵢ = _selectdim(space, A, D, i:i)
        @. s = max(s, abs(Aᵢ) * ν⁻ⁱ)
    end
    return s
end

function _opnorm(space::Fourier, ::Val{D}, A, ν) where {D}
    ν⁻¹ = inv(ν)
    ν⁻ⁱ = one(ν⁻¹)
    @inbounds s = abs.(_selectdim(space, A, D, 0:0)) .* ν⁻ⁱ
    @inbounds for i ∈ 1:order(space)
        ν⁻ⁱ *= ν⁻¹
        Aᵢ = _selectdim(space, A, D, i:i)
        A₋ᵢ = _selectdim(space, A, D, -i:-i)
        @. s = max(s, abs(A₋ᵢ) * ν⁻ⁱ, abs(Aᵢ) * ν⁻ⁱ)
    end
    return s
end

function _opnorm(space::Chebyshev, ::Val{D}, A, ν) where {D}
    ν⁻¹ = inv(ν)
    ν⁻ⁱ = one(ν⁻¹)
    @inbounds s = abs.(_selectdim(space, A, D, 0:0)) .* ν⁻ⁱ
    @inbounds for i ∈ 1:order(space)
        ν⁻ⁱ *= ν⁻¹
        Aᵢ = _selectdim(space, A, D, i:i)
        @. s = max(s, abs(Aᵢ) * ν⁻ⁱ)
    end
    return s
end

## action

(A::Functional)(b::Sequence) = *(A, b)

function Base.:*(A::Functional, b::Sequence)
    s = zero(promote_type(eltype(A), eltype(b)))
    if A.domain == b.space
        for (Aᵢ,bᵢ) ∈ zip(A.coefficients, b.coefficients)
            s += Aᵢ*bᵢ
        end
        return s
    elseif A.domain ⊆ b.space
        @inbounds for α ∈ eachindex(A.domain)
            s += A[α]*b[α]
        end
        return s
    elseif b.space ⊆ A.domain
        @inbounds for α ∈ eachindex(b.space)
            s += A[α]*b[α]
        end
        return s
    else
        @inbounds for α ∈ eachindex(A.domain ∩ b.space)
            s += A[α]*b[α]
        end
        return s
    end
end

## specific functional

struct Evaluation{T}
    value :: T
end

(ℰ::Evaluation)(a::Sequence) = evaluate(a, ℰ.value)
Base.:*(ℰ::Evaluation, a::Sequence) = evaluate(a, ℰ.value)

function Functional(domain::Taylor, ℰ::Evaluation{T}) where {T}
    A = Functional(domain, Vector{T}(undef, length(domain)))
    @inbounds A[0] = one(T)
    @inbounds for i ∈ 1:order(domain)
        A[i] = ℰ.value ^ i
    end
    return A
end

function Functional(domain::Fourier, ℰ::Evaluation)
    iωx = im*domain.frequency*ℰ.value
    CoefType = float(typeof(iωx))
    A = Functional(domain, Vector{CoefType}(undef, length(domain)))
    @inbounds A[0] = zero(CoefType)
    @inbounds for j ∈ 1:order(domain)
        eiωxj = exp(iωx*j)
        A[j] = eiωxj
        A[-j] = conj(eiωxj)
    end
    return A
end

function Functional(domain::Chebyshev, ℰ::Evaluation{T}) where {T}
    A = Functional(domain, Vector{T}(undef, length(domain)))
    ord = order(domain)
    @inbounds A[0] = one(T)
    ord == 0 && return A
    x2 = 2ℰ.value
    @inbounds A[1] = x2
    ord == 1 && return A
    @inbounds A[2] = x2*A[1] - 2A[0]
    @inbounds for j ∈ 3:ord
        A[j] = x2*A[j-1] - A[j-2]
    end
    return A
end
