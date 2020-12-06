## utilities

Base.size(A::Operator{CartesianSpace{T},CartesianSpace{S}}) where {N₁,T<:NTuple{N₁,SequenceSpace},N₂,S<:NTuple{N₂,SequenceSpace}} =
    (N₂, N₁)
Base.size(A::Operator{CartesianSpace{T},<:SequenceSpace}) where {N,T<:NTuple{N,SequenceSpace}} =
    (1, N)
Base.size(A::Operator{<:SequenceSpace,CartesianSpace{T}}) where {N,T<:NTuple{N,SequenceSpace}} =
    (N, 1)
function Base.size(A::Operator{CartesianSpace{T},CartesianSpace{S}}, i::Int) where {N₁,T<:NTuple{N₁,SequenceSpace},N₂,S<:NTuple{N₂,SequenceSpace}}
    i == 1 && return N₂
    i == 2 && return N₁
    return 1
end
function Base.size(A::Operator{CartesianSpace{T},<:SequenceSpace}, i::Int) where {N,T<:NTuple{N,SequenceSpace}}
    i == 2 && return N
    return 1
end
function Base.size(A::Operator{<:SequenceSpace,CartesianSpace{T}}, i::Int) where {N,T<:NTuple{N,SequenceSpace}}
    i == 1 && return N
    return 1
end

Base.length(A::Operator{CartesianSpace{T},CartesianSpace{S}}) where {N₁,T<:NTuple{N₁,SequenceSpace},N₂,S<:NTuple{N₂,SequenceSpace}} =
    N₁*N₂
Base.length(A::Operator{CartesianSpace{T},<:SequenceSpace}) where {N,T<:NTuple{N,SequenceSpace}} = N
Base.length(A::Operator{<:SequenceSpace,CartesianSpace{T}}) where {N,T<:NTuple{N,SequenceSpace}} = N

Base.iterate(A::Operator{<:CartesianSpace,<:CartesianSpace}) = (view(A, 1, 1), 2)
Base.iterate(A::Operator{<:CartesianSpace,<:SequenceSpace}) = (view(A, 1), 2)
Base.iterate(A::Operator{<:SequenceSpace,<:CartesianSpace}) = (view(A, 1), 2)

function Base.iterate(A::Operator{CartesianSpace{T},CartesianSpace{S}}, i::Int) where {N₁,T<:NTuple{N₁,SequenceSpace},N₂,S<:NTuple{N₂,SequenceSpace}}
    if 1 ≤ i ≤ N₁*N₂
        col_, row_ = divrem(i, N₂)
        col_ == 0 && return (view(A, row_, 1), i+1)
        row_ == 0 && return (view(A, N₂, col_), i+1)
        return (view(A, row_, col_), i+1)
    else
        return nothing
    end
end
Base.iterate(A::Operator{CartesianSpace{T},<:SequenceSpace}, i::Int) where {N,T<:NTuple{N,SequenceSpace}} =
    1 ≤ i ≤ N ? (view(A, i), i+1) : nothing
Base.iterate(A::Operator{<:SequenceSpace,CartesianSpace{T}}, i::Int) where {N,T<:NTuple{N,SequenceSpace}} =
    1 ≤ i ≤ N ? (view(A, i), i+1) : nothing

## getindex, view, setindex!

for f ∈ (:getindex, :view)
    @eval begin
        Base.@propagate_inbounds function Base.$f(A::Operator{<:CartesianSpace,<:CartesianSpace}, i::Int, j::Int)
            if i == j == 1
                return Operator(A.domain[1], A.range[1], $f(A.coefficients, 1:length(A.range[1]), 1:length(A.domain[1])))
            elseif i == 1
                len = mapreduce(k -> length(A.domain[k]), +, 1:j-1)
                indices = len+1:len+length(A.domain[j])
                return Operator(A.domain[j], A.range[1], $f(A.coefficients, 1:length(A.range[1]), indices))
            elseif j == 1
                len = mapreduce(k -> length(A.range[k]), +, 1:i-1)
                indices = len+1:len+length(A.range[i])
                return Operator(A.domain[1], A.range[i], $f(A.coefficients, indices, 1:length(A.domain[1])))
            else
                len₁ = mapreduce(k -> length(A.range[k]), +, 1:i-1)
                indices₁ = len₁+1:len₁+length(A.range[i])
                len₂ = mapreduce(k -> length(A.domain[k]), +, 1:j-1)
                indices₂ = len₂+1:len₂+length(A.domain[j])
                return Operator(A.domain[j], A.range[i], $f(A.coefficients, indices₁, indices₂))
            end
        end

        Base.@propagate_inbounds function Base.$f(A::Operator{<:CartesianSpace,<:SequenceSpace}, j::Int)
            if j == 1
                return Operator(A.domain[1], A.range, $f(A.coefficients, 1:length(A.range), 1:length(A.domain[1])))
            else
                len = mapreduce(k -> length(A.domain[k]), +, 1:j-1)
                indices = len+1:len+length(A.domain[j])
                return Operator(A.domain[j], A.range, $f(A.coefficients, 1:length(A.range), indices))
            end
        end

        Base.@propagate_inbounds function Base.$f(A::Operator{<:SequenceSpace,<:CartesianSpace}, i::Int)
            if i == 1
                return Operator(A.domain, A.range[1], $f(A.coefficients, 1:length(A.range[1]), 1:length(A.domain)))
            else
                len = mapreduce(k -> length(A.range[k]), +, 1:i-1)
                indices = len+1:len+length(A.range[i])
                return Operator(A.domain, A.range[i], $f(A.coefficients, indices, 1:length(A.domain)))
            end
        end
    end
end

Base.@propagate_inbounds function Base.setindex!(A::Operator{<:CartesianSpace,<:CartesianSpace}, x::Operator, i::Int, j::Int)
    @assert A.domain[j] == x.domain && A.range[i] == x.range
    if i == j == 1
        return setindex!(A.coefficients, x.coefficients, 1:length(A.range[1]), 1:length(A.domain[1]))
    elseif i == 1
        len = mapreduce(k -> length(A.domain[k]), +, 1:j-1)
        indices = len+1:len+length(A.domain[j])
        return setindex!(A.coefficients, x.coefficients, 1:length(A.range[1]), indices)
    elseif j == 1
        len = mapreduce(k -> length(A.range[k]), +, 1:i-1)
        indices = len+1:len+length(A.range[i])
        return setindex!(A.coefficients, x.coefficients, indices, 1:length(A.domain[1]))
    else
        len₁ = mapreduce(k -> length(A.range[k]), +, 1:i-1)
        indices₁ = len₁+1:len₁+length(A.range[i])
        len₂ = mapreduce(k -> length(A.domain[k]), +, 1:j-1)
        indices₂ = len₂+1:len₂+length(A.domain[j])
        return setindex!(A.coefficients, x.coefficients, indices₁, indices₂)
    end
end

Base.@propagate_inbounds function Base.setindex!(A::Operator{<:CartesianSpace,<:SequenceSpace}, x::Operator, j::Int)
    @assert A.domain[j] == x.domain
    if j == 1
        return setindex!(A.coefficients, x.coefficients, 1:length(A.range), 1:length(A.domain[1]))
    else
        len = mapreduce(k -> length(A.domain[k]), +, 1:j-1)
        indices = len+1:len+length(A.domain[j])
        return setindex!(A.coefficients, x.coefficients, 1:length(A.range), indices)
    end
end

Base.@propagate_inbounds function Base.setindex!(A::Operator{<:SequenceSpace,<:CartesianSpace}, x::Operator, i::Int)
    @assert A.range[i] == x.range
    if i == 1
        return setindex!(A.coefficients, x.coefficients, 1:length(A.range[1]), 1:length(A.domain))
    else
        len = mapreduce(k -> length(A.range[k]), +, 1:i-1)
        indices = len+1:len+length(A.range[i])
        return setindex!(A.coefficients, x.coefficients, indices, 1:length(A.domain))
    end
end

## opnorm - WARNING: not type stable for cartesian spaces with different space, e.g. Taylor × Chebyshev
# MWE of type instability:
# struct BigA{T,S,R}
#     x :: T
#     y :: S
#     mat :: R
# end
# struct A{T,S,R}
#     x :: T
#     y :: S
#     value :: R
# end
# f(a::A) = 2a.value
# Base.length(a::BigA) = length(a.mat)
# Base.iterate(a::BigA) = (A(a.x[1], a.y[1], a.mat[1]), 2)
# function Base.iterate(a::BigA, i::Int)
#     if 1 ≤ i ≤ length(a)
#         col_, row_ = divrem(i, size(a.mat, 1))
#         col_ == 0 && return (A(a.x[row_], a.y[1], a.mat[i]), i+1)
#         row_ == 0 && return (A(a.x[size(a.mat, 1)], a.y[col_], a.mat[i]), i+1)
#         return (A(a.x[row_], a.y[col_], a.mat[i]), i+1)
#     else
#         nothing
#     end
# end
# a = BigA((1, π), (1//3, false), [1. 2. ; 3. 4.])
# @code_warntype map(f, a)
# MWE of type instability - perhaps more clear:
# struct BigA{T,S,R}
#     x :: T
#     y :: S
#     mat :: R
# end
# struct A{T,S,R}
#     x :: T
#     y :: S
#     value :: R
# end
# f(a::A{T,S,R}) where {T,S,R} = (2a.value)::R
# Base.getindex(a::BigA, i::Int, j::Int) = A(a.x[j], a.y[i], getindex(a.mat, i, j))
# map(i -> f(a[1,i]), 1:length(a.x))
# map(t -> f(a[t[1],t[2]]), Iterators.product(1:length(a.y), 1:length(a.x)))

LinearAlgebra.opnorm(A::Operator{CartesianSpace{T},CartesianSpace{S}}) where {N₁,T<:NTuple{N₁,SequenceSpace},N₂,S<:NTuple{N₂,SequenceSpace}} =
    opnorm(reshape(map(opnorm, A), N₂, N₁), Inf)
LinearAlgebra.opnorm(A::Operator{<:CartesianSpace,<:SequenceSpace}) =
    norm(map(opnorm, A), 1)
LinearAlgebra.opnorm(A::Operator{<:SequenceSpace,<:CartesianSpace}) =
    norm(map(opnorm, A), Inf)

function LinearAlgebra.opnorm(A::Operator{CartesianSpace{T},CartesianSpace{S}}, ν, μ, p::Real=Inf) where {N₁,T<:NTuple{N₁,SequenceSpace},N₂,S<:NTuple{N₂,SequenceSpace}}
    @assert N₁ == length(ν) && N₂ == length(μ)
    return opnorm(reshape(map((Aᵢ, tᵢ) -> opnorm(Aᵢ, tᵢ[1], tᵢ[2]), A, Iterators.product(ν, μ)), N₂, N₁), p)
end
function LinearAlgebra.opnorm(A::Operator{CartesianSpace{T},<:SequenceSpace}, ν, μ, p::Real=Inf) where {N,T<:NTuple{N,SequenceSpace}}
    @assert N == length(ν)
    return opnorm(transpose(map((Aᵢ, νᵢ) -> opnorm(Aᵢ, νᵢ, μ), A, ν)), p)
end
function LinearAlgebra.opnorm(A::Operator{<:SequenceSpace,CartesianSpace{T}}, ν, μ, p::Real=Inf) where {N,T<:NTuple{N,SequenceSpace}}
    @assert N == length(μ)
    return norm(map((Aᵢ, μᵢ) -> opnorm(Aᵢ, ν, μᵢ), A, μ), p)
end

## action - WARNING: not type stable for Taylor × Chebyshev

function (A::Operator{CartesianSpace{T},CartesianSpace{S}})(b::Sequence{CartesianSpace{R}}) where {N₁,T<:NTuple{N₁,SequenceSpace},N₂,S<:NTuple{N₂,SequenceSpace},N₃,R<:NTuple{N₃,SequenceSpace}}
    @assert N₁ == N₃
    CoefType = promote_type(eltype(A), eltype(b))
    c = Sequence(A.range, Vector{CoefType}(undef, length(A.range)))
    c.coefficients .= zero(CoefType)
    @inbounds for j ∈ 1:N₁, i ∈ 1:N₂
        c[i] += view(A, i, j)(view(b, j))
    end
    return c
end

function (A::Operator{CartesianSpace{T},<:SequenceSpace})(b::Sequence{CartesianSpace{S}}) where {N₁,T<:NTuple{N₁,SequenceSpace},N₂,S<:NTuple{N₂,SequenceSpace}}
    @assert N₁ == N₂
    return mapreduce((Aⱼ, bⱼ) -> Aⱼ(bⱼ), +, A, b)
end

(A::Operator{<:SequenceSpace,<:CartesianSpace})(b::Sequence{<:SequenceSpace}) =
    Sequence(A.range, mapreduce(Aᵢ -> Aᵢ(b).coefficients, vcat, A))

## arithmetic

for f ∈ (:+, :-)
    @eval begin
        Base.$f(A::Operator{CartesianSpace{T₁},CartesianSpace{S₁},R₁}, B::Operator{CartesianSpace{T₂},CartesianSpace{S₂},R₂}) where {N₁,T₁<:NTuple{N₁,SequenceSpace},N₂,S₁<:NTuple{N₂,SequenceSpace},R₁,T₂<:NTuple{N₁,SequenceSpace},S₂<:NTuple{N₂,SequenceSpace},R₂} =
            Operator(A.domain ∪ B.domain, A.range ∪ B.range, mapreduce(j -> mapreduce(i -> $f(A[i,j], B[i,j]).coefficients, vcat, 1:N₂), hcat, 1:N₁))

        function Base.$f(A::Operator{CartesianSpace{T},CartesianSpace{S},R}, B::Matrix) where {N₁,T<:NTuple{N₁,SequenceSpace},N₂,S<:NTuple{N₂,SequenceSpace},R}
            @assert (N₂, N₁) == size(B)
            C = mapreduce(j -> mapreduce(i -> $f(A[i,j], B[i,j]), vcat, 1:N₂), hcat, 1:N₁)
            v = mapreduce(j -> mapreduce(i -> C[i,j].coefficients, vcat, 1:N₂), hcat, 1:N₁)
            domain = CartesianSpace(ntuple(j -> mapreduce(i -> C[i,j].domain, ∪, 1:N₂), N₁))
            range = CartesianSpace(ntuple(i -> mapreduce(j -> C[i,j].range, ∪, 1:N₁), N₂))
            return Operator(domain, range, v)
        end

        function Base.$f(B::Matrix, A::Operator{CartesianSpace{T},CartesianSpace{S},R}) where {N₁,T<:NTuple{N₁,SequenceSpace},N₂,S<:NTuple{N₂,SequenceSpace},R}
            @assert (N₂, N₁) == size(B)
            C = mapreduce(j -> mapreduce(i -> $f(B[i,j], A[i,j]), vcat, 1:N₂), hcat, 1:N₁)
            v = mapreduce(j -> mapreduce(i -> C[i,j].coefficients, vcat, 1:N₂), hcat, 1:N₁)
            domain = CartesianSpace(ntuple(j -> mapreduce(i -> C[i,j].domain, ∪, 1:N₂), N₁))
            range = CartesianSpace(ntuple(i -> mapreduce(j -> C[i,j].range, ∪, 1:N₁), N₂))
            return Operator(domain, range, v)
        end
    end
end

Base.:+(B::UniformScaling, A::Operator{<:CartesianSpace,<:CartesianSpace}) = +(A, B)

function Base.:+(A::Operator{CartesianSpace{T},CartesianSpace{S},R₁}, B::UniformScaling{R₂}) where {N,T<:NTuple{N,SequenceSpace},S<:NTuple{N,SequenceSpace},R₁,R₂}
    NewType = promote_type(R₁, R₂)
    C = Operator{CartesianSpace{T},CartesianSpace{S},NewType}(A.domain, A.range, undef)
    C.coefficients .= A.coefficients
    @inbounds for i ∈ 1:N
        C[i,i] += B.λ
    end
    return C
end

function Base.:-(A::Operator{CartesianSpace{T},CartesianSpace{S},R₁}, B::UniformScaling{R₂}) where {N,T<:NTuple{N,SequenceSpace},S<:NTuple{N,SequenceSpace},R₁,R₂}
    NewType = promote_type(R₁, R₂)
    C = Operator{CartesianSpace{T},CartesianSpace{S},NewType}(A.domain, A.range, undef)
    C.coefficients .= A.coefficients
    @inbounds for i ∈ 1:N
        C[i,i] -= B.λ
    end
    return C
end

function Base.:-(B::UniformScaling{R₂}, A::Operator{CartesianSpace{T},CartesianSpace{S},R₁}) where {N,T<:NTuple{N,SequenceSpace},S<:NTuple{N,SequenceSpace},R₁,R₂}
    NewType = promote_type(R₁, R₂)
    C = Operator{CartesianSpace{T},CartesianSpace{S},NewType}(A.domain, A.range, undef)
    @. C.coefficients = -A.coefficients
    @inbounds for i ∈ 1:N
        C[i,i] += B.λ
    end
    return C
end

## eigen

function LinearAlgebra.eigen(A::Operator{<:CartesianSpace,<:CartesianSpace})
    Λ, Ξ = eigen(A.coefficients)
    @inbounds Ξ_ = map(i -> Sequence(A.domain, Ξ[:,i]), 1:length(Λ))
    return Λ, Ξ_
end

#

Base.:+(A::Operator{CartesianSpace{T},CartesianSpace{S},R}, 𝒟::Derivative) where {N,T<:NTuple{N,SequenceSpace},S<:NTuple{N,SequenceSpace},R} =
    Operator(A.domain, A.range, mapreduce(j -> mapreduce(i -> i == j ? A[i,j].coefficients : (A[i,j] + 𝒟).coefficients, vcat, 1:N), hcat, 1:N))

Base.:+(𝒟::Derivative, A::Operator{CartesianSpace{T},CartesianSpace{S},R}) where {N,T<:NTuple{N,SequenceSpace},S<:NTuple{N,SequenceSpace},R} =
    Operator(A.domain, A.range, mapreduce(j -> mapreduce(i -> i == j ? A[i,j].coefficients : (𝒟 + A[i,j]).coefficients, vcat, 1:N), hcat, 1:N))

Base.:-(A::Operator{CartesianSpace{T},CartesianSpace{S},R}, 𝒟::Derivative) where {N,T<:NTuple{N,SequenceSpace},S<:NTuple{N,SequenceSpace},R} =
    Operator(A.domain, A.range, mapreduce(j -> mapreduce(i -> i == j ? A[i,j].coefficients : (A[i,j] - 𝒟).coefficients, vcat, 1:N), hcat, 1:N))

Base.:-(𝒟::Derivative, A::Operator{CartesianSpace{T},CartesianSpace{S},R}) where {N,T<:NTuple{N,SequenceSpace},S<:NTuple{N,SequenceSpace},R} =
    Operator(A.domain, A.range, mapreduce(j -> mapreduce(i -> i == j ? -A[i,j].coefficients : (𝒟 - A[i,j]).coefficients, vcat, 1:N), hcat, 1:N))
