struct Operator{T<:SequenceSpace,S<:SequenceSpace,R<:AbstractMatrix}
    domain :: T
    range :: S
    coefficients :: R
    function Operator{T,S,R}(domain::T, range::S, coefficients::R) where {T<:SequenceSpace,S<:SequenceSpace,R<:AbstractMatrix}
        @assert (length(range), length(domain)) == size(coefficients)
        return new{T,S,R}(domain, range, coefficients)
    end
end

Operator(domain::T, range::S, coefficients::R) where {T<:SequenceSpace,S<:SequenceSpace,R<:AbstractMatrix} =
    Operator{T,S,R}(domain, range, coefficients)

## project

function project(A::Operator, domain, range)
    CoefType = eltype(A)
    C = Operator(domain, range, Matrix{CoefType}(undef, length(range), length(domain)))
    if A.domain == domain && A.range == range
        @. C.coefficients = A.coefficients
        return C
    elseif A.domain ⊆ domain && A.range == range
        C.coefficients .= zero(CoefType)
        @inbounds for α ∈ eachindex(A.domain)
            C[:,α] = A[:,α]
        end
        return c
    elseif domain ⊆ A.domain && A.range == range
        @inbounds for α ∈ eachindex(domain)
            C[:,α] = A[:,α]
        end
        return C
    elseif A.domain == domain && A.range ⊆ range
        C.coefficients .= zero(CoefType)
        @inbounds for α ∈ eachindex(A.range)
            C[α,:] = A[α,:]
        end
        return C
    elseif domain == A.domain && range ⊆ A.range
        @inbounds for α ∈ eachindex(range)
            C[α,:] = A[α,:]
        end
        return C
    elseif A.domain ⊆ domain && A.range ⊆ range
        C.coefficients .= zero(CoefType)
        @inbounds for β ∈ eachindex(A.domain), α ∈ eachindex(A.range)
            C[α,β] = A[α,β]
        end
        return C
    elseif domain ⊆ A.domain && range ⊆ A.range
        @inbounds for β ∈ eachindex(domain), α ∈ eachindex(range)
            C[α,β] = A[α,β]
        end
        return C
    else
        C.coefficients .= zero(CoefType)
        @inbounds for β ∈ eachindex(A.domain ∩ domain), α ∈ eachindex(A.range ∩ range)
            C[α,β] = A[α,β]
        end
        return C
    end
end

## utilities

Base.size(A::Operator) = (length(A.range), length(A.domain))
function Base.size(A::Operator, i::Int)
    i == 1 && return length(A.range)
    i == 2 && return length(A.domain)
    return 1
end

Base.axes(A::Operator) = (eachindex(A.range), eachindex(A.domain))
function Base.axes(A::Operator, i::Int)
    i == 1 && return eachindex(A.range)
    i == 2 && return eachindex(A.domain)
    return 1:1
end

Base.eltype(A::Operator) = eltype(A.coefficients)
Base.eltype(::Type{Operator{T,S,R}}) where {T<:SequenceSpace,S<:SequenceSpace,R<:AbstractMatrix} = eltype(R)

## getindex, view, setindex!

for f ∈ (:getindex, :view)
    @eval begin
        Base.@propagate_inbounds function Base.$f(A::Operator, α, β)
            @boundscheck(!isindexof(α, A.range) && !isindexof(β, A.domain) && throw(BoundsError((A.range, A.domain), (α, β))))
            return $f(A.coefficients, _findindex(α, A.range), _findindex(β, A.domain))
        end
    end
end

Base.@propagate_inbounds function Base.setindex!(A::Operator, x, α, β)
    @boundscheck(!isindexof(α, A.range) && !isindexof(β, A.domain) && throw(BoundsError((A.range, A.domain), (α, β))))
    return setindex!(A.coefficients, x, _findindex(α, A.range), _findindex(β, A.domain))
end

## ==, iszero, isapprox

Base.:(==)(A::Operator, B::Operator) = A.range == B.range && A.domain == B.domain && A.coefficients == B.coefficients

Base.iszero(A::Operator) = iszero(A.coefficients)

Base.isapprox(A::Operator, B::Operator; kwargs...) = A.range == B.range && A.domain == B.domain && isapprox(A.coefficients, B.coefficients; kwargs...)

## copy, similar

Base.copy(A::Operator) = Operator(A.domain, A.range, copy(A.coefficients))

Base.similar(A::Operator) = Operator(A.domain, A.range, similar(A.coefficients))

## zero, one

Base.zero(A::Operator) = Operator(A.domain, A.range, zero.(A.coefficients))

function Base.one(A::Operator)
    CoefType = eltype(A)
    B = zero(A)
    @inbounds for α ∈ eachindex(A.domain ∩ A.range)
        B[α,α] = one(CoefType)
    end
    return B
end

## float, real, complex, conj, imag, abs, abs2

for f ∈ (:float, :real, :complex, :conj, :imag)
    @eval Base.$f(A::Operator) = Operator(A.domain, A.range, $f(A.coefficients))
end

# Base.abs(A::Operator) = Operator(A.domain, A.range, abs.(A.coefficients))
# Base.abs2(A::Operator) = Operator(A.domain, A.range, abs2.(A.coefficients))

## promotion

Base.convert(::Type{Operator{T₁,S₁,R₁}}, A::Operator{T₂,S₂,R₂}) where {T₁<:SequenceSpace,S₁<:SequenceSpace,R₁<:AbstractMatrix,T₂<:SequenceSpace,S₂<:SequenceSpace,R₂<:AbstractMatrix} =
    Operator{T₁,S₁,R₁}(convert(T₁, A.domain), convert(S₁, A.range), convert(R₁, A.coefficients))

Base.promote_rule(::Type{Operator{T₁,S₁,R₁}}, ::Type{Operator{T₂,S₂,R₂}}) where {T₁<:SequenceSpace,S₁<:SequenceSpace,R₁<:AbstractMatrix,T₂<:SequenceSpace,S₂<:SequenceSpace,R₂<:AbstractMatrix} =
    Operator{promote_type(T₁,T₂),promote_type(S₁,S₂),promote_type(R₁,R₂)}

## opnorm

LinearAlgebra.opnorm(A::Operator{<:UnivariateSpace,<:UnivariateSpace}) =
    opnorm(A, 1, 1)
LinearAlgebra.opnorm(A::Operator{TensorSpace{T},<:UnivariateSpace}) where {N,T<:NTuple{N,UnivariateSpace}} =
    opnorm(A, ntuple(i -> 1, N), 1)
LinearAlgebra.opnorm(A::Operator{<:UnivariateSpace,TensorSpace{T}}) where {N,T<:NTuple{N,UnivariateSpace}} =
    opnorm(A, 1, ntuple(i -> 1, N))
LinearAlgebra.opnorm(A::Operator{TensorSpace{T},TensorSpace{S}}) where {N₁,T<:NTuple{N₁,UnivariateSpace},N₂,S<:NTuple{N₂,UnivariateSpace}} =
    opnorm(A, ntuple(i -> 1, N₁), ntuple(i -> 1, N₂))

LinearAlgebra.opnorm(A::Operator, ν, μ) =
    norm(Sequence(A.range, map(α -> opnorm(Functional(A.domain, view(A, α, :)), ν), eachindex(A.range))), μ)

function LinearAlgebra.opnorm(A::Operator{Taylor,Taylor}, ν::Real=1, μ::Real=1)
    @assert ν > 0 && μ > 0
    CoefType = float(promote_type(real(eltype(A)), typeof(ν), typeof(μ)))
    s = zero(CoefType)
    @inbounds for j ∈ 0:order(A.domain)
        s_ = convert(CoefType, abs(A[0,j]))
        @inbounds for i ∈ 1:order(A.range)
            s_ += abs(A[i,j]) * μ^i
        end
        s = max(s, s_ / ν^j)
    end
    return s
end

function LinearAlgebra.opnorm(A::Operator{<:Fourier,<:Fourier}, ν::Real=1, μ::Real=1)
    @assert ν > 0 && μ > 0
    CoefType = float(promote_type(real(eltype(A)), typeof(ν), typeof(μ)))
    s = zero(CoefType)
    @inbounds for j ∈ eachindex(A.domain)
        s_ = convert(CoefType, abs(A[0,j]))
        @inbounds for i ∈ 1:order(A.range)
            s_ += (abs(A[-i,j]) + abs(A[i,j])) * μ^i
        end
        s = max(s, s_ / ν^abs(j))
    end
    return s
end

function LinearAlgebra.opnorm(A::Operator{Chebyshev,Chebyshev}, ν::Real=1, μ::Real=1)
    @assert ν > 0 && μ > 0
    CoefType = float(promote_type(real(eltype(A)), typeof(ν), typeof(μ)))
    s = zero(CoefType)
    @inbounds for j ∈ 0:order(A.domain)
        s_ = convert(CoefType, abs(A[0,j]))
        @inbounds for i ∈ 1:order(A.range)
            s_ += abs(A[i,j]) * μ^i
        end
        s = max(s, s_ / ν^j)
    end
    return s
end

## action

(A::Operator)(b::Sequence) = *(A, b)

function Base.:*(A::Operator, b::Sequence)
    CoefType = promote_type(eltype(A), eltype(b))
    c = Sequence(A.range, Vector{CoefType}(undef, length(A.range)))
    if A.domain == b.space
        mul!(c.coefficients, A.coefficients, b.coefficients)
        return c
    elseif A.domain ⊆ b.space
        c.coefficients .= zero(CoefType)
        @inbounds for β ∈ eachindex(A.domain), α ∈ eachindex(A.range)
            c[α] += A[α,β]*b[β]
        end
        return c
    elseif b.space ⊆ A.domain
        c.coefficients .= zero(CoefType)
        @inbounds for β ∈ eachindex(b.space), α ∈ eachindex(A.range)
            c[α] += A[α,β]*b[β]
        end
        return c
    else
        c.coefficients .= zero(CoefType)
        @inbounds for β ∈ eachindex(A.domain ∩ b.space), α ∈ eachindex(A.range)
            c[α] += A[α,β]*b[β]
        end
        return c
    end
end

function Base.:\(A::Operator, b::Sequence)
    if b.space == A.range
        return Sequence(A.domain, A.coefficients \ b.coefficients)
    elseif b.space ⊆ A.range
        b_ = project(b, A.range)
        return Sequence(A.domain, A.coefficients \ b_.coefficients)
    elseif A.range ⊆ b.space
        A_ = project(A, A.domain, b.space)
        return Sequence(A.domain, A_.coefficients \ b.coefficients)
    else
        space = b.space ∪ A.range
        b_ = project(b, space)
        A_ = project(A, A.domain, space)
        return Sequence(A.domain, A_.coefficients \ b_.coefficients)
    end
end

## Multiplication operator

function Operator(domain::Taylor, range::Taylor, a::Sequence{Taylor})
    CoefType = eltype(a)
    A = Operator(domain, range, Matrix{CoefType}(undef, length(range), length(domain)))
    A.coefficients .= zero(CoefType)
    @inbounds for j ∈ eachindex(domain), i ∈ j:min(order(range),order(a.space)+j)
        A[i,j] = a[i-j]
    end
    return A
end

function Operator(domain::Fourier, range::Fourier, a::Sequence{<:Fourier})
    @assert domain.frequency == range.frequency == a.space.frequency
    CoefType = eltype(a)
    A = Operator(domain, range, Matrix{CoefType}(undef, length(range), length(domain)))
    A.coefficients .= zero(CoefType)
    @inbounds for j ∈ eachindex(domain), i ∈ max(-order(range),-order(a.space)+j):min(order(range),order(a.space)+j)
        A[i,j] = a[i-j]
    end
    return A
end

function Operator(domain::Chebyshev, range::Chebyshev, a::Sequence{Chebyshev})
    CoefType = float(eltype(a))
    A = Operator(domain, range, Matrix{CoefType}(undef, length(range), length(domain)))
    A.coefficients .= zero(CoefType)
    for j ∈ eachindex(domain), i ∈ max(-order(range),-order(a.space)+j):min(order(range),order(a.space)+j)
        if abs(i-j) == 0
            A[abs(i),j] += a[abs(i-j)]
        else
            A[abs(i),j] += a[abs(i-j)]/2
        end
    end
    return A
end

## Derivative operator

struct Derivative end

(::Derivative)(a::Sequence) = differentiate(a)

function Operator(domain::Taylor, range::Taylor, ::Derivative, ::Type{T}=Float64) where {T}
    A = Operator(domain, range, Matrix{T}(undef, length(range), length(domain)))
    A.coefficients .= zero(T)
    @inbounds for i ∈ 0:min(domain.order-1, range.order)
        A[i,i+1] = i+1
    end
    return A
end

function Operator(domain::Fourier, range::Fourier, ::Derivative, ::Type{T}=Complex{Float64}) where {T}
    @assert domain.frequency == range.frequency
    A = Operator(domain, range, Matrix{T}(undef, length(range), length(domain)))
    A.coefficients .= zero(T)
    iω = im*domain.frequency
    @inbounds for j ∈ 1:min(domain.order, range.order)
        iωj = iω*j
        A[-j,-j] = -iωj
        A[j,j] = iωj
    end
    return A
end

function Operator(domain::Chebyshev, range::Chebyshev, ::Derivative, ::Type{T}=Float64) where {T}
    A = Operator(domain, range, Matrix{T}(undef, length(range), length(domain)))
    A.coefficients .= zero(T)
    @inbounds for i ∈ 1:2:domain.order
        A[0,i] = i
    end
    for j ∈ 2:2:domain.order-1
        @inbounds for i ∈ 1:2:min(j-1, range.order)
            A[i,j] = 2j
        end
        @inbounds for i ∈ 2:2:min(j, range.order)
            A[i,j+1] = 2(j+1)
        end
    end
    if iseven(domain.order)
        @inbounds for i ∈ 1:2:min(domain.order-1, range.order)
            A[i,domain.order] = 2domain.order
        end
    end
    return A
end

## Integral operator

struct Integral end

(::Integral)(a::Sequence) = integrate(a)

function Operator(domain::Taylor, range::Taylor, ::Integral, ::Type{T}=Float64) where {T}
    A = Operator(domain, range, Matrix{T}(undef, length(range), length(domain)))
    A.coefficients .= zero(T)
    @inbounds for i ∈ 0:min(domain.order, range.order-1)
        A[i+1,i] = inv(i+1)
    end
    return A
end

function Operator(domain::Fourier, range::Fourier, ::Integral, ::Type{T}=Complex{Float64}) where {T}
    @assert domain.frequency == range.frequency
    A = Operator(domain, range, Matrix{T}(undef, length(range), length(domain)))
    A.coefficients .= zero(T)
    iω⁻¹ = im*inv(domain.frequency)
    @inbounds for j ∈ 1:min(domain.order, range.order)
        iω⁻¹j⁻¹ = iω⁻¹/j
        A[-j,-j] = iω⁻¹j⁻¹
        A[j,j] = -iω⁻¹j⁻¹
    end
    return A
end

function Operator(domain::Chebyshev, range::Chebyshev, ::Integral, ::Type{T}=Float64) where {T}
    A = Operator(domain, range, Matrix{T}(undef, length(range), length(domain)))
    A.coefficients .= zero(T)
    # first two columns
    @inbounds A[0,0] = one(T)
    if domain.order ≥ 1
        @inbounds A[0,1] = -0.25
        if range.order ≥ 2
            @inbounds A[2,1] = 0.25
        end
    end
    if range.order ≥ 1
        @inbounds A[1,0] = one(T)
    end
    # first row
    @inbounds for i ∈ 2:2:domain.order-1
        A[0,i] = inv(1-i^2)
        A[0,i+1] = inv(i*(i+2))
    end
    if iseven(domain.order)
        @inbounds A[0,domain.order] = inv(1-domain.order^2)
    end
    # remaining
    @inbounds for i ∈ 2:min(domain.order, range.order+1)
        A[i-1,i] = -inv(2(i-1))
    end
    @inbounds for i ∈ 2:min(domain.order, range.order-1)
        A[i+1,i] = inv(2(i+1))
    end
    #
    return A
end

## Shift operator

struct Shift{T}
    from :: T
    to :: T
end

(𝒮::Shift)(a::Sequence) = shift(a, 𝒮.from, 𝒮.to)

function Operator(domain::Fourier{T}, range::Fourier{S}, 𝒮::Shift) where {T,S}
    @assert domain.frequency == range.frequency
    iωΔτ = im*domain.frequency*one(S)*(𝒮.to-𝒮.from)
    CoefType = float(typeof(iωΔτ))
    A = Operator(domain, range, Matrix{CoefType}(undef, length(range), length(domain)))
    A.coefficients .= zero(CoefType)
    @inbounds A[0,0] = one(CoefType)
    @inbounds for j ∈ 1:min(domain.order, range.order)
        eiωΔτj = exp(iωΔτ*j)
        A[-j,-j] = conj(eiωΔτj)
        A[j,j] = eiωΔτj
    end
    return A
end

## rescale

struct Rescale{T}
    γ :: T
end

(ℛ::Rescale)(a::Sequence) = rescale(a, ℛ.γ)

function Operator(domain::Taylor, range::Taylor, ℛ::Rescale{T}) where {T}
    A = Operator(domain, range, Matrix{T}(undef, length(range), length(domain)))
    A.coefficients .= zero(T)
    @inbounds for i ∈ eachindex(domain ∩ range)
        A[i,i] = ℛ.γ ^ i
    end
    return A
end

function Operator(domain::TensorSpace{NTuple{N,Taylor}}, range::TensorSpace{NTuple{N,Taylor}}, ℛ::Rescale{NTuple{N,T}}) where {N,T}
    A = Operator(domain, range, Matrix{T}(undef, length(range), length(domain)))
    A.coefficients .= zero(T)
    @inbounds for α ∈ eachindex(domain ∩ range)
        A[α,α] = mapreduce(^, *, ℛ.γ, α)
    end
    return A
end
