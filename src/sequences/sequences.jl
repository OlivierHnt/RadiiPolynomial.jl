"""
    Sequence{T<:SequenceSpace,S<:AbstractVector}

Compactly supported sequence of the given space.

Fields:
- `space :: T`
- `coefficients :: S`
"""
struct Sequence{T<:SequenceSpace,S<:AbstractVector}
    space :: T
    coefficients :: S
    function Sequence{T,S}(space::T, coefficients::S) where {T<:SequenceSpace,S<:AbstractVector}
        @assert length(space) == length(coefficients)
        return new{T,S}(space, coefficients)
    end
end

Sequence(space::T, coefficients::S) where {T<:SequenceSpace,S<:AbstractVector} =
    Sequence{T,S}(space, coefficients)

## utilities

Base.firstindex(a::Sequence) = firstindex(a.space)

Base.lastindex(a::Sequence) = lastindex(a.space)

Base.eachindex(a::Sequence) = eachindex(a.space)

Base.length(a::Sequence) = length(a.space)

Base.size(a::Sequence) = tuple(length(a.space)) # necessary for broadcasting

Base.iterate(a::Sequence) = iterate(a.coefficients)
Base.iterate(a::Sequence, i::Int) = iterate(a.coefficients, i)

Base.eltype(a::Sequence) = eltype(a.coefficients)
Base.eltype(::Type{Sequence{T,S}}) where {T<:SequenceSpace,S<:AbstractVector} = eltype(S)

## space, coefficients, order

space(a::Sequence) = a.space
coefficients(a::Sequence) = a.coefficients
order(a::Sequence) = order(a.space)

## permutedims

Base.permutedims(a::Sequence{T}, σ::Vector{Int}) where {T<:TensorSpace} =
    Sequence(a.space[σ], vec(permutedims(reshape(a.coefficients, size(a.space)), σ)))

## project

"""
    project(a::Sequence, space::SequenceSpace)

Return a [`Sequence`](@ref) representing `a` in `space`.
"""
function project(a::Sequence, space::SequenceSpace)
    CoefType = eltype(a)
    c = Sequence(space, Vector{CoefType}(undef, length(space)))
    if a.space == space
        @. c.coefficients = a.coefficients
        return c
    elseif a.space ⊆ space
        c.coefficients .= zero(CoefType)
        @inbounds for α ∈ eachindex(a.space)
            c[α] = a[α]
        end
        return c
    elseif space ⊆ a.space
        @inbounds for α ∈ eachindex(space)
            c[α] = a[α]
        end
        return c
    else
        c.coefficients .= zero(CoefType)
        @inbounds for α ∈ eachindex(a.space ∩ space)
            c[α] = a[α]
        end
        return c
    end
end

## selectdim

"""
    selectdim(a::Sequence{TensorSpace{T}}, dim::Int, i::Int) where {N,T<:NTuple{N,UnivariateSpace}}

Return a [`Sequence`](@ref) in the space `a.space[1:dim-1] ⊗ a.space[dim+1:N]` and whose coefficients are a view of all the data `a.coefficients` where the index for dimension `dim` equals `i`.
"""
Base.@propagate_inbounds function Base.selectdim(a::Sequence{TensorSpace{T}}, dim::Int, i::Int) where {N,T<:NTuple{N,UnivariateSpace}}
    @boundscheck(!isindexof(i, a.space[dim]) && throw(BoundsError(a.space[dim], i)))
    A = reshape(a.coefficients, size(a.space))
    A_ = selectdim(A, dim, _findindex(i, a.space[dim]))
    return Sequence(a.space[1:dim-1] ⊗ a.space[dim+1:N], vec(A_))
end

## getindex, view, setindex!

for f ∈ (:getindex, :view)
    @eval begin
        Base.@propagate_inbounds function Base.$f(a::Sequence, α)
            @boundscheck(!isindexof(α, a.space) && throw(BoundsError(a.space, α)))
            return $f(a.coefficients, _findindex(α, a.space))
        end

        Base.@propagate_inbounds function Base.$f(a::Sequence{TensorSpace{T}}, α::NTuple{N,Int}) where {N,T<:NTuple{N,UnivariateSpace}}
            @boundscheck(!isindexof(α, a.space) && throw(BoundsError(a.space, α)))
            return $f(a.coefficients, _findindex(α, a.space))
        end

        Base.@propagate_inbounds function Base.$f(a::Sequence{TensorSpace{T}}, α::Tuple) where {N,T<:NTuple{N,UnivariateSpace}}
            @boundscheck(!isindexof(α, a.space) && throw(BoundsError(a.space, α)))
            A = reshape(a.coefficients, size(a.space))
            return $f(A, _findindex(α, a.space)...)
        end
    end
end

Base.@propagate_inbounds function Base.setindex!(a::Sequence, x, α)
    @boundscheck(!isindexof(α, a.space) && throw(BoundsError(a.space, α)))
    return setindex!(a.coefficients, x, _findindex(α, a.space))
end

Base.@propagate_inbounds function Base.setindex!(a::Sequence{TensorSpace{T}}, x, α::NTuple{N,Int}) where {N,T<:NTuple{N,UnivariateSpace}}
    @boundscheck(!isindexof(α, a.space) && throw(BoundsError(a.space, α)))
    return setindex!(a.coefficients, x, _findindex(α, a.space))
end

Base.@propagate_inbounds function Base.setindex!(a::Sequence{TensorSpace{T}}, x, α::Tuple) where {N,T<:NTuple{N,UnivariateSpace}}
    @boundscheck(!isindexof(α, a.space) && throw(BoundsError(a.space, α)))
    A = reshape(a.coefficients, size(a.space))
    return setindex!(A, x, _findindex(α, a.space)...)
end

## ==, iszero, isapprox

Base.:(==)(a::Sequence, b::Sequence) = a.space == b.space && a.coefficients == b.coefficients

Base.iszero(a::Sequence) = iszero(a.coefficients)

Base.isapprox(a::Sequence, b::Sequence; kwargs...) = a.space == b.space && isapprox(a.coefficients, b.coefficients; kwargs...)

## copy, similiar

Base.copy(a::Sequence) = Sequence(a.space, copy(a.coefficients))

Base.similar(a::Sequence) = Sequence(a.space, similar(a.coefficients))

## zero, one

Base.zero(a::Sequence) = Sequence(a.space, zero.(a.coefficients))

function Base.one(a::Sequence)
    b = zero(a)
    @inbounds b[_constant_index(a.space)] = one(eltype(a))
    return b
end

## float, real, complex, conj, imag, abs, abs2

for f ∈ (:float, :real, :complex, :conj, :imag)
    @eval Base.$f(a::Sequence) = Sequence(a.space, $f(a.coefficients))
end

# Base.abs(a::Sequence) = Sequence(a.space, abs.(a.coefficients))
# Base.abs2(a::Sequence) = Sequence(a.space, abs2.(a.coefficients))

## promotion

Base.convert(::Type{Sequence{T₁,S₁}}, a::Sequence{T₂,S₂}) where {T₁<:SequenceSpace,S₁<:AbstractVector,T₂<:SequenceSpace,S₂<:AbstractVector} =
    Sequence{T₁,S₁}(convert(T₁, a.space), convert(S₁, a.coefficients))

Base.promote_rule(::Type{Sequence{T₁,S₁}}, ::Type{Sequence{T₂,S₂}}) where {T₁<:SequenceSpace,S₁<:AbstractVector,T₂<:SequenceSpace,S₂<:AbstractVector} =
    Sequence{promote_type(T₁,T₂),promote_type(S₁,S₂)}

## shifting

"""
    shift(a::Sequence{<:Fourier}, from, to)

Return the shifted [`Fourier`](@ref) sequence representing a shifted Fourier series centered at `to` and was originally centered at `from`.
"""
function shift(a::Sequence{<:Fourier}, from, to)
    iωΔτ = im*a.space.frequency*(to-from)
    CoefType = float(promote_type(eltype(a), typeof(iωΔτ)))
    c = Sequence(a.space, Vector{CoefType}(undef, length(a.space)))
    @inbounds c[0] = a[0]
    @inbounds for j ∈ 1:order(a.space)
        eiωΔτj = exp(iωΔτ*j)
        c[-i] = a[-i]*conj(eiωΔτj)
        c[i] = a[i]*eiωΔτj
    end
    return c
end

## rescaling

function rescale(a::Sequence{<:UnivariateSpace}, γ)
    CoefType = promote_type(eltype(a), typeof(γ))
    c = Sequence(a.space, Vector{CoefType}(undef, length(a.space)))
    @. c.coefficients = a.coefficients
    return rescale!(c, γ)
end

function rescale(a::Sequence{<:TensorSpace}, γ; dims=:)
    CoefType = promote_type(eltype(a), eltype(γ))
    c = Sequence(a.space, Vector{CoefType}(undef, length(a.space)))
    @. c.coefficients = a.coefficients
    return rescale!(c, γ; dims = dims)
end

function rescale!(a::Sequence{Taylor}, γ)
    isone(γ) && return a
    @inbounds for i ∈ 1:order(a)
        a[i] *= γ^i
    end
    return a
end

function rescale!(a::Sequence{<:TensorSpace}, γ; dims=:)
    A = reshape(a.coefficients, size(a.space))
    _rescale!(a.space, dims, A, γ)
    return a
end

_rescale!(space, dims::Int, A, γ) = @inbounds _rescale!(space[dims], Val(dims), A, γ)
_rescale!(space, dims::Tuple{Int}, A, γ) = @inbounds _rescale!(space[dims[1]], Val(dims[1]), A, γ)
_rescale!(space, ::Colon, A, γ) = _apply_rescale!(space, A, γ)
_rescale!(space, dims::NTuple{N,Int}, A, γ) where {N} = @inbounds _apply_rescale!(space[[dims...]], dims, A, γ)
_rescale!(space, dims::Vector{Int}, A, γ) = @inbounds _apply_rescale!(space[dims], (dims...,), A, γ)

_apply_rescale!(space::TensorSpace{<:NTuple{N₁,UnivariateSpace}}, A::Array{T,N₂}, γ::NTuple{N₁,Any}) where {N₁,T,N₂} =
    @inbounds _rescale!(space[1], Val(N₂-N₁+1), _apply_rescale!(space[2:N₁], A, γ[2:N₁]), γ[1])

_apply_rescale!(space::TensorSpace{<:NTuple{2,UnivariateSpace}}, A::Array{T,N}, γ::NTuple{2,Any}) where {T,N} =
    @inbounds _rescale!(space[1], Val(N-1), _rescale!(space[2], Val(N), A, γ[2]), γ[1])

_apply_rescale!(space::TensorSpace{<:NTuple{N₁,UnivariateSpace}}, dims::NTuple{N₁,Int}, A::Array{T,N₂}, γ::NTuple{N₁,Any}) where {N₁,T,N₂} =
    @inbounds _rescale!(space[1], Val(dims[1]), _apply_rescale!(space[2:N₁], dims[2:N₁], A, γ[2:N₁]), γ[1])

_apply_rescale!(space::TensorSpace{<:NTuple{2,UnivariateSpace}}, dims::NTuple{2,Int}, A::Array{T,N}, γ::NTuple{2,Any}) where {T,N} =
    @inbounds _rescale!(space[1], Val(dims[1]), _rescale!(space[2], Val(dims[2]), A, γ[2]), γ[1])

function _rescale!(space::Taylor, ::Val{D}, A::Array, γ) where {D}
    isone(γ) && return A
    @inbounds for i ∈ 1:order(space)
        A_view = selectdim(A, D, i+1)
        γⁱ = γ^i
        @. A_view *= γⁱ
    end
    return A
end

## rounding via Banach algebra

"""
    banach_algebra_rounding!(a::Sequence, ν::Vector)

Return a rounded [`Sequence`](@ref) according to the decay `ν` (c.f. [Computing Discrete Convolutions with Verified Accuracy Via Banach Algebras and the FFT](https://link.springer.com/article/10.21136/AM.2018.0082-18)).
"""
function banach_algebra_rounding!(a::Sequence, ν::Vector)
    @assert length(a.coefficients) == length(ν)
    @inbounds for i ∈ eachindex(a.coefficients)
        ν[i] ≥ sup(abs(a.coefficients[i])) && continue
        a.coefficients[i] = @interval(-ν[i], ν[i])
    end
    return a
end

## norm

function LinearAlgebra.norm(a::Sequence{Taylor}, ν::Real=1)
    @assert ν > 0
    ord = order(a.space)
    @inbounds s = convert(promote_type(real(eltype(a)), typeof(ν)), abs(a[ord]))
    @inbounds for i ∈ ord-1:-1:0
        s = muladd(s, ν, abs(a[i]))
    end
    return s
end

function LinearAlgebra.norm(a::Sequence{<:Fourier}, ν::Real=1)
    @assert ν > 0
    ord = order(a.space)
    @inbounds s = convert(promote_type(real(eltype(a)), typeof(ν)), abs(a[-ord])+abs(a[ord]))
    @inbounds for i ∈ ord-1:-1:1
        s = muladd(s, ν, abs(a[-i])+abs(a[i]))
    end
    @inbounds s = muladd(s, ν, abs(a[0]))
    return s
end

function LinearAlgebra.norm(a::Sequence{Chebyshev}, ν::Real=1)
    @assert ν > 0
    ord = order(a.space)
    @inbounds s = convert(promote_type(real(eltype(a)), typeof(ν)), abs(a[ord]))
    @inbounds for i ∈ ord-1:-1:0
        s = muladd(s, ν, abs(a[i]))
    end
    return s
end

LinearAlgebra.norm(a::Sequence{TensorSpace{T}}) where {N,T<:NTuple{N,UnivariateSpace}} =
    norm(a, ntuple(i -> 1, N))

function LinearAlgebra.norm(a::Sequence{TensorSpace{T}}, ν::NTuple{N,Real}) where {N,T<:NTuple{N,UnivariateSpace}}
    @assert all(νᵢ -> νᵢ > 0, ν)
    A = reshape(a.coefficients, size(a.space))
    return @inbounds _apply_norm(a.space, A, ν)[1]
end

_apply_norm(space::TensorSpace{<:NTuple{N₁,UnivariateSpace}}, A::Array{T,N₂}, ν::NTuple{N₁,Real}) where {N₁,T,N₂} =
    @inbounds _norm(space[1], Val(N₂-N₁+1), _apply_norm(space[2:N₁], A, ν[2:N₁]), ν[1])

_apply_norm(space::TensorSpace{<:NTuple{2,UnivariateSpace}}, A::Array{T,N}, ν::NTuple{2,Real}) where {T,N} =
    @inbounds _norm(space[1], Val(N-1), _norm(space[2], Val(N), A, ν[2]), ν[1])

function _norm(space::Taylor, ::Val{D}, A::Array{T,N}, ν::S) where {D,T,N,S<:Real}
    ord = order(space)
    @inbounds s = convert(Array{promote_type(real(T), S),N}, abs.(selectdim(A, D, ord+1:ord+1)))
    @inbounds for i ∈ ord:-1:1
        Aᵢ = selectdim(A, D, i:i)
        @. s = muladd(s, ν, abs(Aᵢ))
    end
    return s
end

function _norm(space::Fourier, ::Val{D}, A::Array{T,N}, ν::S) where {D,T,N,S<:Real}
    ord = order(space)
    idx₀ = ord+1
    @inbounds s = convert(Array{promote_type(real(T), S),N}, abs.(selectdim(A, D, 1:1))+abs.(selectdim(A, D, idx₀+ord:idx₀+ord)))
    @inbounds for i ∈ ord-1:-1:1
        idxᵢ = idx₀+i
        idx₋ᵢ = idx₀-i
        A₋ᵢ = selectdim(A, D, idx₋ᵢ:idx₋ᵢ)
        Aᵢ = selectdim(A, D, idxᵢ:idxᵢ)
        @. s = muladd(s, ν, abs(A₋ᵢ)+abs(Aᵢ))
    end
    @inbounds A₀ = selectdim(A, D, idx₀:idx₀)
    @. s = muladd(s, ν, abs(A₀))
    return s
end

function _norm(space::Chebyshev, ::Val{D}, A::Array{T,N}, ν::S) where {D,T,N,S<:Real}
    ord = order(space)
    @inbounds s = convert(Array{promote_type(real(T), S),N}, abs.(selectdim(A, D, ord+1:ord+1)))
    @inbounds for i ∈ ord:-1:1
        Aᵢ = selectdim(A, D, i:i)
        @. s = muladd(s, ν, abs(Aᵢ))
    end
    return s
end

## show

Base.show(io::IO, a::Sequence) = print(io, pretty_string(a.space) * ":\n\n" * pretty_string(a))

function pretty_string(a::Sequence{T}) where {T<:SequenceSpace}
    indices = eachindex(a.space)
    @inbounds strout = string(" (", a[indices[1]], ") ") * string_basis_symbol(T, indices[1])
    if length(a.space) ≤ 10
        @inbounds for i ∈ 2:length(a.space)
            strout *= "\n" * string(" (", a[indices[i]], ") ") * string_basis_symbol(T, indices[i])
        end
    else
        @inbounds for i ∈ 2:10
            strout *= "\n" * string(" (", a[indices[i]], ") ") * string_basis_symbol(T, indices[i])
        end
        strout *= "\n ⋮"
        @inbounds for i ∈ max(12,length(a.space)-9):length(a.space)
            strout *= "\n" * string(" (", a[indices[i]], ") ") * string_basis_symbol(T, indices[i])
        end
    end
    return strout
end

string_basis_symbol(::Type{<:UnivariateSpace}, i::Int) = i < 0 ? string("𝜙₋", subscriptify(-i)) : string("𝜙", subscriptify(i))

function string_basis_symbol(::Type{<:TensorSpace}, α::NTuple{N,Int}) where {N}
    @inbounds s = α[1] < 0 ? string("𝜙⁽¹⁾₋", subscriptify(-α[1])) : string("𝜙⁽¹⁾", subscriptify(α[1]))
    @inbounds for i ∈ 2:N
        s *= " ⊗ "
        s *= α[i] < 0 ? string("𝜙⁽", superscriptify(i), "⁾₋", subscriptify(-α[i])) : string("𝜙⁽", superscriptify(i), "⁾", subscriptify(α[i]))
    end
    return s
end

const subscript_digits = ["₀","₁","₂","₃","₄","₅","₆","₇","₈","₉"]
const superscript_digits = ["⁰","¹","²","³","⁴","⁵","⁶","⁷","⁸","⁹"]
subscriptify(n::Int) = join([subscript_digits[i+1] for i ∈ reverse(digits(n))])
superscriptify(n::Int) = join([superscript_digits[i+1] for i ∈ reverse(digits(n))])
