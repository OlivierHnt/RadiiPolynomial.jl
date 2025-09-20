norm(::LinearOperator, ::Real=Inf) = throw(ArgumentError("`norm` is only defined for `Sequence`. Use `opnorm` instead"))
norm(::LinearOperator, ::BanachSpace) = throw(ArgumentError("`norm` is only defined for `Sequence`. Use `opnorm` instead"))
norm(::LinearOperator, ::BanachSpace, ::BanachSpace) = throw(ArgumentError("`norm` is only defined for `Sequence`. Use `opnorm` instead"))

"""
    norm(a::AbstractSequence, p::Real=Inf)

Compute the `p`-norm of `a`. Only `p` equal to `1`, `2` or `Inf` is supported.

This is equivalent to:
- `norm(a, Ell1(IdentityWeight()))` if `p == 1`
- `norm(a, Ell2(IdentityWeight()))` if `p == 2`
- `norm(a, EllInf(IdentityWeight()))` if `p == Inf`

See also: [`norm(::Sequence, ::BanachSpace)`](@ref).
"""
function norm(a::AbstractSequence, p::Real=Inf)
    if p == 1
        return norm(a, Ell1(IdentityWeight()))
    elseif p == 2
        return norm(a, Ell2(IdentityWeight()))
    elseif p == Inf
        return norm(a, EllInf(IdentityWeight()))
    else
        return throw(ArgumentError("p-norm is only supported for p equal to 1, 2 or Inf"))
    end
end

"""
    opnorm(A::LinearOperator, p::Real=Inf)

Compute the operator norm of `A` induced by the `p`-norm. Only `p` equal to `1`,
`2` or `Inf` is supported.

This is equivalent to:
- `opnorm(A, Ell1(IdentityWeight()))` if `p == 1`
- `opnorm(A, Ell2(IdentityWeight()))` if `p == 2`
- `opnorm(A, EllInf(IdentityWeight()))` if `p == Inf`

See also: [`opnorm(::LinearOperator, ::BanachSpace)`](@ref),
[`opnorm(::LinearOperator, ::BanachSpace, ::BanachSpace)`](@ref) and
[`opnorm(::LinearOperator{<:VectorSpace,ParameterSpace}, ::BanachSpace)`](@ref).
"""
function opnorm(A::LinearOperator, p::Real=Inf)
    if p == 1
        return opnorm(A, Ell1(IdentityWeight()))
    elseif p == 2
        return opnorm(A, Ell2(IdentityWeight()))
    elseif p == Inf
        return opnorm(A, EllInf(IdentityWeight()))
    else
        return throw(ArgumentError("p-norm is only supported for p equal to 1, 2 or Inf"))
    end
end

"""
    opnorm(A::LinearOperator, X::BanachSpace, Y::BanachSpace)

Compute the operator norm of `A` where `X` is the Banach space corresponding to
`domain(A)` and `Y` the Banach space corresponding to `codomain(A)`.

See also: [`opnorm(::LinearOperator, ::Real=Inf)`](@ref),
[`opnorm(::LinearOperator, ::BanachSpace)`](@ref) and
[`opnorm(::LinearOperator{<:VectorSpace,ParameterSpace}, ::BanachSpace)`](@ref).
"""
function opnorm(A::LinearOperator, X::BanachSpace, Y::BanachSpace)
    codomain_A = codomain(A)
    A_ = coefficients(A)
    @inbounds v₁ = norm(Sequence(codomain_A, view(A_, :, 1)), Y)
    T = typeof(v₁)
    sz = size(A_, 2)
    v = Vector{T}(undef, sz)
    @inbounds v[1] = v₁
    @inbounds for i ∈ 2:sz
        A_view = view(A_, :, i)
        if all(z -> _safe_isequal(z, zero(z)), A_view)
            v[i] = zero(T)
        else
            v[i] = norm(Sequence(codomain_A, A_view), Y)
        end
    end
    return opnorm(LinearOperator(domain(A), ParameterSpace(), transpose(v)), X)
end

"""
    opnorm(A::LinearOperator, X::BanachSpace)

Compute the operator norm of `A` where `X` is the Banach space corresponding to
both `domain(A)` and `codomain(A)`.

See also: [`opnorm(::LinearOperator, ::Real=Inf)`](@ref),
[`opnorm(::LinearOperator, ::BanachSpace, ::BanachSpace)`](@ref) and
[`opnorm(::LinearOperator{<:VectorSpace,ParameterSpace}, ::BanachSpace)`](@ref).
"""
opnorm(A::LinearOperator, X::BanachSpace) = opnorm(A, X, X)

#

opnorm(ℳ::Multiplication, X::BanachSpace) = norm(sequence(ℳ), X)

#

"""
    norm(a::Sequence, X::BanachSpace)

Compute the norm of `a` by interpreting `space(a)` as `X`.

See also: [`norm(::Sequence, ::Real=Inf)`](@ref).
"""
norm(::Sequence, ::BanachSpace)

function norm(a::InfiniteSequence, X::BanachSpace = banachspace(a))
    X == banachspace(a) || return throw(ArgumentError("banach spaces must be equal"))
    return sequence_norm(a) + sequence_error(a)
end

"""
    opnorm(A::LinearOperator{<:VectorSpace,ParameterSpace}, X::BanachSpace)

Compute the operator norm of `A` where `X` is the Banach space corresponding to
`domain(A)`.

See also: [`opnorm(::LinearOperator, ::Real=Inf)`](@ref),
[`opnorm(::LinearOperator, ::BanachSpace, ::BanachSpace)`](@ref) and
[`opnorm(::LinearOperator, ::BanachSpace)`](@ref).
"""
opnorm(::LinearOperator{<:VectorSpace,ParameterSpace}, ::BanachSpace)

for T ∈ (:Ell1, :Ell2, :EllInf)
    @eval begin
        norm(a::Sequence, X::$T{<:Weight}) = _apply(X, space(a), coefficients(a))
        norm(a::Sequence{TensorSpace{S}}, X::$T{<:Weight}) where {N,S<:NTuple{N,BaseSpace}} =
            norm(a, $T(ntuple(_ -> weight(X), Val(N))))
        function norm(a::Sequence{TensorSpace{S}}, X::$T{<:NTuple{N,Weight}}) where {N,S<:NTuple{N,BaseSpace}}
            space_a = space(a)
            A = _no_alloc_reshape(coefficients(a), dimensions(space_a))
            return _apply(X, space_a, A)
        end
        _apply(X::$T, space::TensorSpace, A) =
            @inbounds _apply($T(weight(X)[1]), space[1], _apply($T(Base.tail(weight(X))), Base.tail(space), A))
        _apply(X::$T, space::TensorSpace{<:Tuple{BaseSpace}}, A) =
            @inbounds _apply($T(weight(X)[1]), space[1], A)

        opnorm(A::LinearOperator{<:VectorSpace,ParameterSpace}, X::$T{<:Weight}) =
            _apply_dual(X, domain(A), vec(coefficients(A)))
        opnorm(A::LinearOperator{TensorSpace{S},ParameterSpace}, X::$T{<:Weight}) where {N,S<:NTuple{N,BaseSpace}} =
            opnorm(A, $T(ntuple(_ -> weight(X), Val(N))))
        function opnorm(A::LinearOperator{TensorSpace{S},ParameterSpace}, X::$T{<:NTuple{N,Weight}}) where {N,S<:NTuple{N,BaseSpace}}
            domain_A = domain(A)
            A_ = _no_alloc_reshape(coefficients(A), dimensions(domain_A))
            return _apply_dual(X, domain_A, A_)
        end
        _apply_dual(X::$T, space::TensorSpace, A) =
            @inbounds _apply_dual($T(weight(X)[1]), space[1], _apply_dual($T(Base.tail(weight(X))), Base.tail(space), A))
        _apply_dual(X::$T, space::TensorSpace{<:Tuple{BaseSpace}}, A) =
            @inbounds _apply_dual($T(weight(X)[1]), space[1], A)
    end
end

#

function opnorm(A::LinearOperator{<:CartesianSpace,<:CartesianSpace}, ::Ell1{IdentityWeight}, ::Ell1{IdentityWeight})
    r = z = abs(zero(eltype(A)))
    A_ = coefficients(A)
    for j ∈ axes(A_, 2)
        s = z
        @inbounds for i ∈ axes(A_, 1)
            s += abs(A_[i,j])
        end
        r = max(r, s)
    end
    return r
end

opnorm(A::LinearOperator, ::Ell2{IdentityWeight}, ::Ell2{IdentityWeight}) =
    sqrt(opnorm(A, Ell1(IdentityWeight())) * opnorm(A, EllInf(IdentityWeight())))

function opnorm(A::LinearOperator{<:CartesianSpace,<:CartesianSpace}, ::EllInf{IdentityWeight}, ::EllInf{IdentityWeight})
    r = z = abs(zero(eltype(A)))
    A_ = coefficients(A)
    for i ∈ axes(A_, 1)
        s = z
        @inbounds for j ∈ axes(A_, 2)
            s += abs(A_[i,j])
        end
        r = max(r, s)
    end
    return r
end

# ParameterSpace

_apply(::Ell1{IdentityWeight}, ::ParameterSpace, A::AbstractVector) = @inbounds abs(A[1])
_apply_dual(::Ell1{IdentityWeight}, ::ParameterSpace, A::AbstractVector) = @inbounds abs(A[1])

_apply(::Ell2{IdentityWeight}, ::ParameterSpace, A::AbstractVector) = @inbounds abs(A[1])
_apply_dual(::Ell2{IdentityWeight}, ::ParameterSpace, A::AbstractVector) = @inbounds abs(A[1])

_apply(::EllInf{IdentityWeight}, ::ParameterSpace, A::AbstractVector) = @inbounds abs(A[1])
_apply_dual(::EllInf{IdentityWeight}, ::ParameterSpace, A::AbstractVector) = @inbounds abs(A[1])

# SequenceSpace

# Taylor

_apply(::Ell1{IdentityWeight}, ::Taylor, A::AbstractVector) = sum(abs, A)
function _apply(::Ell1{IdentityWeight}, ::Taylor, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T)))
    @inbounds A₁ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₁))
    s .= abs.(A₁)
    @inbounds for i ∈ 2:size(A, N)
        s .+= abs.(selectdim(A, N, i))
    end
    return s
end
_apply_dual(::Ell1{IdentityWeight}, space::Taylor, A::AbstractArray) = _apply(EllInf(IdentityWeight()), space, A)

function _apply(X::Ell1{<:GeometricWeight}, space::Taylor, A::AbstractVector)
    ν = rate(weight(X))
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * one(ν)
    @inbounds for i ∈ ord-1:-1:0
        s = s * ν + abs(A[i+1])
    end
    return s
end
function _apply(X::Ell1{<:GeometricWeight}, space::Taylor, A::AbstractArray{T,N}) where {T,N}
    ν = rate(weight(X))
    CoefType = typeof(abs(zero(T))*ν)
    ord = order(space)
    @inbounds A₀ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ ord-1:-1:0
        s .= s .* ν .+ abs.(selectdim(A, N, i+1))
    end
    return s
end
function _apply_dual(X::Ell1{<:GeometricWeight}, space::Taylor, A::AbstractVector)
    ν = inv(rate(weight(X)))
    νⁱ = one(ν)
    @inbounds s = abs(A[1]) * νⁱ
    @inbounds for i ∈ 1:order(space)
        νⁱ *= ν
        s = max(s, abs(A[i+1]) * νⁱ)
    end
    return s
end
function _apply_dual(X::Ell1{<:GeometricWeight}, space::Taylor, A::AbstractArray{T,N}) where {T,N}
    ν = inv(rate(weight(X)))
    νⁱ = one(ν)
    CoefType = typeof(abs(zero(T))*νⁱ)
    @inbounds A₀ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:order(space)
        νⁱ *= ν
        s .= max.(s, abs.(selectdim(A, N, i+1)) .* νⁱ)
    end
    return s
end

function _apply(X::Ell1{<:AlgebraicWeight}, space::Taylor, A::AbstractVector)
    @inbounds s = abs(A[1]) * _getindex(weight(X), space, 0)
    @inbounds for i ∈ 1:order(space)
        s += abs(A[i+1]) * _getindex(weight(X), space, i)
    end
    return s
end
function _apply(X::Ell1{<:AlgebraicWeight}, space::Taylor, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))*_getindex(weight(X), space, 0))
    @inbounds A₀ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:order(space)
        s .+= abs.(selectdim(A, N, i+1)) .* _getindex(weight(X), space, i)
    end
    return s
end
function _apply_dual(X::Ell1{<:AlgebraicWeight}, space::Taylor, A::AbstractVector)
    @inbounds s = abs(A[1]) / _getindex(weight(X), space, 0)
    @inbounds for i ∈ 1:order(space)
        s = max(s, abs(A[i+1]) / _getindex(weight(X), space, i))
    end
    return s
end
function _apply_dual(X::Ell1{<:AlgebraicWeight}, space::Taylor, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))/_getindex(weight(X), space, 0))
    @inbounds A₀ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:order(space)
        s .= max.(s, abs.(selectdim(A, N, i+1)) ./ _getindex(weight(X), space, i))
    end
    return s
end

_apply(::Ell2{IdentityWeight}, ::Taylor, A::AbstractVector) = sqrt(sum(abs2, A))
function _apply(::Ell2{IdentityWeight}, ::Taylor, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(sqrt(abs2(zero(T))))
    @inbounds A₁ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₁))
    s .= abs2.(A₁)
    @inbounds for i ∈ 2:size(A, N)
        s .+= abs2.(selectdim(A, N, i))
    end
    s .= sqrt.(s)
    return s
end
_apply_dual(X::Ell2{IdentityWeight}, space::Taylor, A::AbstractArray) = _apply(X, space, A)

_apply(::EllInf{IdentityWeight}, ::Taylor, A::AbstractVector) = maximum(abs, A)
function _apply(::EllInf{IdentityWeight}, ::Taylor, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T)))
    @inbounds A₁ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₁))
    s .= abs.(A₁)
    @inbounds for i ∈ 2:size(A, N)
        s .= max.(s, abs.(selectdim(A, N, i)))
    end
    return s
end
_apply_dual(::EllInf{IdentityWeight}, space::Taylor, A::AbstractArray) = _apply(Ell1(IdentityWeight()), space, A)

function _apply(X::EllInf{<:GeometricWeight}, space::Taylor, A::AbstractVector)
    ν = rate(weight(X))
    νⁱ = one(ν)
    @inbounds s = abs(A[1]) * νⁱ
    @inbounds for i ∈ 1:order(space)
        νⁱ *= ν
        s = max(s, abs(A[i+1]) * νⁱ)
    end
    return s
end
function _apply(X::EllInf{<:GeometricWeight}, space::Taylor, A::AbstractArray{T,N}) where {T,N}
    ν = rate(weight(X))
    νⁱ = one(ν)
    CoefType = typeof(abs(zero(T))*νⁱ)
    @inbounds A₀ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:order(space)
        νⁱ *= ν
        s .= max.(s, abs.(selectdim(A, N, i+1)) .* νⁱ)
    end
    return s
end
function _apply_dual(X::EllInf{<:GeometricWeight}, space::Taylor, A::AbstractVector)
    ν = inv(rate(weight(X)))
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * one(ν)
    @inbounds for i ∈ ord-1:-1:0
        s = s * ν + abs(A[i+1])
    end
    return s
end
function _apply_dual(X::EllInf{<:GeometricWeight}, space::Taylor, A::AbstractArray{T,N}) where {T,N}
    ν = inv(rate(weight(X)))
    CoefType = typeof(abs(zero(T))*ν)
    ord = order(space)
    @inbounds A₀ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ ord-1:-1:0
        s .= s .* ν .+ abs.(selectdim(A, N, i+1))
    end
    return s
end

function _apply(X::EllInf{<:AlgebraicWeight}, space::Taylor, A::AbstractVector)
    @inbounds s = abs(A[1]) * _getindex(weight(X), space, 0)
    @inbounds for i ∈ 1:order(space)
        s = max(s, abs(A[i+1]) * _getindex(weight(X), space, i))
    end
    return s
end
function _apply(X::EllInf{<:AlgebraicWeight}, space::Taylor, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))*_getindex(weight(X), space, 0))
    @inbounds A₀ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:order(space)
        s .= max.(s, abs.(selectdim(A, N, i+1)) .* _getindex(weight(X), space, i))
    end
    return s
end
function _apply_dual(X::EllInf{<:AlgebraicWeight}, space::Taylor, A::AbstractVector)
    @inbounds s = abs(A[1]) / _getindex(weight(X), space, 0)
    @inbounds for i ∈ 1:order(space)
        s += abs(A[i+1]) / _getindex(weight(X), space, i)
    end
    return s
end
function _apply_dual(X::EllInf{<:AlgebraicWeight}, space::Taylor, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))/_getindex(weight(X), space, 0))
    @inbounds A₀ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:order(space)
        s .+= abs.(selectdim(A, N, i+1)) ./ _getindex(weight(X), space, i)
    end
    return s
end

# Fourier

_apply(::Ell1{IdentityWeight}, ::Fourier, A::AbstractVector) = sum(abs, A)
function _apply(::Ell1{IdentityWeight}, ::Fourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T)))
    @inbounds A₁ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₁))
    s .= abs.(A₁)
    @inbounds for i ∈ 2:size(A, N)
        s .+= abs.(selectdim(A, N, i))
    end
    return s
end
_apply_dual(::Ell1{IdentityWeight}, space::Fourier, A::AbstractArray) = _apply(EllInf(IdentityWeight()), space, A)

function _apply(X::Ell1{<:GeometricWeight}, space::Fourier, A::AbstractVector)
    ν = rate(weight(X))
    ord = order(space)
    if ord == 0
        @inbounds s = abs(A[1]) * one(ν)
    else
        @inbounds s = (abs(A[1]) + abs(A[2ord+1])) * one(ν)
        @inbounds for i ∈ ord-1:-1:1
            s = s * ν + abs(A[ord+1-i]) + abs(A[ord+1+i])
        end
        @inbounds s = s * ν + abs(A[ord+1])
    end
    return s
end
function _apply(X::Ell1{<:GeometricWeight}, space::Fourier, A::AbstractArray{T,N}) where {T,N}
    ν = rate(weight(X))
    CoefType = typeof(abs(zero(T))*ν)
    ord = order(space)
    @inbounds A₋ₙ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₋ₙ))
    if ord == 0
        s .= abs.(A₋ₙ)
    else
        @inbounds s .= abs.(selectdim(A, N, 2ord+1)) .+ abs.(A₋ₙ)
        @inbounds for i ∈ ord-1:-1:1
            s .= s .* ν .+ abs.(selectdim(A, N, ord+1-i)) .+ abs.(selectdim(A, N, ord+1+i))
        end
        @inbounds s .= s .* ν .+ abs.(selectdim(A, N, ord+1))
    end
    return s
end
function _apply_dual(X::Ell1{<:GeometricWeight}, space::Fourier, A::AbstractVector)
    ν = inv(rate(weight(X)))
    νⁱ = one(ν)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * νⁱ
    @inbounds for i ∈ 1:ord
        νⁱ *= ν
        s = max(s, abs(A[ord+1+i]) * νⁱ, abs(A[ord+1-i]) * νⁱ)
    end
    return s
end
function _apply_dual(X::Ell1{<:GeometricWeight}, space::Fourier, A::AbstractArray{T,N}) where {T,N}
    ν = inv(rate(weight(X)))
    νⁱ = one(ν)
    CoefType = typeof(abs(zero(T))*νⁱ)
    ord = order(space)
    @inbounds A₀ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:ord
        νⁱ *= ν
        s .= max.(s, abs.(selectdim(A, N, ord+1-i)) .* νⁱ, abs.(selectdim(A, N, ord+1+i)) .* νⁱ)
    end
    return s
end

function _apply(X::Ell1{<:AlgebraicWeight}, space::Fourier, A::AbstractVector)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * _getindex(weight(X), space, 0)
    @inbounds for i ∈ 1:ord
        s += (abs(A[ord+1-i]) + abs(A[ord+1+i])) * _getindex(weight(X), space, i)
    end
    return s
end
function _apply(X::Ell1{<:AlgebraicWeight}, space::Fourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))*_getindex(weight(X), space, 0))
    ord = order(space)
    @inbounds A₀ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    @inbounds s .= abs.(A₀)
    @inbounds for i ∈ 1:ord
        s .+= (abs.(selectdim(A, N, ord+1-i)) .+ abs.(selectdim(A, N, ord+1+i))) .* _getindex(weight(X), space, i)
    end
    return s
end
function _apply_dual(X::Ell1{<:AlgebraicWeight}, space::Fourier, A::AbstractVector{T}) where {T}
    ord = order(space)
    @inbounds s = abs(A[ord+1]) / _getindex(weight(X), space, 0)
    @inbounds for i ∈ 1:ord
        x = abs(one(T)) / _getindex(weight(X), space, i)
        s = max(s, abs(A[ord+1+i]) * x, abs(A[ord+1-i]) * x)
    end
    return s
end
function _apply_dual(X::Ell1{<:AlgebraicWeight}, space::Fourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))/_getindex(weight(X), space, 0))
    ord = order(space)
    @inbounds A₀ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:ord
        x = abs(one(T)) / _getindex(weight(X), space, i)
        s .= max.(s, abs.(selectdim(A, N, ord+1-i)) .* x, abs.(selectdim(A, N, ord+1+i)) .* x)
    end
    return s
end

_apply(::Ell2{IdentityWeight}, ::Fourier, A::AbstractVector) = sqrt(sum(abs2, A))
function _apply(::Ell2{IdentityWeight}, ::Fourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(sqrt(abs2(zero(T))))
    @inbounds A₁ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₁))
    s .= abs2.(A₁)
    @inbounds for i ∈ 2:size(A, N)
        s .+= abs2.(selectdim(A, N, i))
    end
    s .= sqrt.(s)
    return s
end
_apply_dual(X::Ell2{IdentityWeight}, space::Fourier, A::AbstractArray) = _apply(X, space, A)

function _apply(X::Ell2{<:BesselWeight}, space::Fourier, A::AbstractVector)
    ord = order(space)
    @inbounds x = abs2(A[ord+1]) * _getindex(weight(X), space, 0)
    @inbounds for i ∈ 1:ord
        x += (abs2(A[ord+1-i]) + abs2(A[ord+1+i])) * _getindex(weight(X), space, i)
    end
    return sqrt(x)
end
function norm(a::Sequence{TensorSpace{T}}, X::Ell2{<:BesselWeight}) where {N,T<:NTuple{N,Fourier}}
    space_a = space(a)
    x = zero(abs2(zero(eltype(a)))*_getindex(weight(X), space_a, ntuple(_ -> 0, Val(N))))
    @inbounds for α ∈ indices(space_a)
        x += abs2(a[α]) * _getindex(weight(X), space_a, α)
    end
    return sqrt(x)
end
function _apply_dual(X::Ell2{<:BesselWeight}, space::Fourier, A::AbstractVector)
    ord = order(space)
    @inbounds x = abs2(A[ord+1]) / _getindex(weight(X), space, 0)
    @inbounds for i ∈ 1:ord
        x += (abs2(A[ord+1-i]) + abs2(A[ord+1+i])) / _getindex(weight(X), space, i)
    end
    return sqrt(x)
end
function opnorm(A::LinearOperator{TensorSpace{T},ParameterSpace}, X::Ell2{<:BesselWeight}) where {N,T<:NTuple{N,Fourier}}
    domain_A = domain(A)
    x = zero(abs2(zero(eltype(A)))/_getindex(weight(X), domain_A, ntuple(_ -> 0, Val(N))))
    @inbounds for α ∈ indices(domain_A)
        x += abs2(A[1,α]) / _getindex(weight(X), domain_A, α)
    end
    return sqrt(x)
end

_apply(::EllInf{IdentityWeight}, ::Fourier, A::AbstractVector) = maximum(abs, A)
function _apply(::EllInf{IdentityWeight}, ::Fourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T)))
    @inbounds A₁ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₁))
    s .= abs.(A₁)
    @inbounds for i ∈ 2:size(A, N)
        s .= max.(s, abs.(selectdim(A, N, i)))
    end
    return s
end
_apply_dual(::EllInf{IdentityWeight}, space::Fourier, A::AbstractArray) = _apply(Ell1(IdentityWeight()), space, A)

function _apply(X::EllInf{<:GeometricWeight}, space::Fourier, A::AbstractVector)
    ν = rate(weight(X))
    νⁱ = one(ν)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * νⁱ
    @inbounds for i ∈ 1:ord
        νⁱ *= ν
        s = max(s, abs(A[ord+1+i]) * νⁱ, abs(A[ord+1-i]) * νⁱ)
    end
    return s
end
function _apply(X::EllInf{<:GeometricWeight}, space::Fourier, A::AbstractArray{T,N}) where {T,N}
    ν = rate(weight(X))
    νⁱ = one(ν)
    CoefType = typeof(abs(zero(T))*νⁱ)
    ord = order(space)
    @inbounds A₀ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:ord
        νⁱ *= ν
        s .= max.(s, abs.(selectdim(A, N, ord+1-i)) .* νⁱ, abs.(selectdim(A, N, ord+1+i)) .* νⁱ)
    end
    return s
end
function _apply_dual(X::EllInf{<:GeometricWeight}, space::Fourier, A::AbstractVector)
    ν = inv(rate(weight(X)))
    ord = order(space)
    if ord == 0
        @inbounds s = abs(A[1]) * one(ν)
    else
        @inbounds s = (abs(A[1]) + abs(A[2ord+1])) * one(ν)
        @inbounds for i ∈ ord-1:-1:1
            s = s * ν + abs(A[ord+1-i]) + abs(A[ord+1+i])
        end
        @inbounds s = s * ν + abs(A[ord+1])
    end
    return s
end
function _apply_dual(X::EllInf{<:GeometricWeight}, space::Fourier, A::AbstractArray{T,N}) where {T,N}
    ν = inv(rate(weight(X)))
    CoefType = typeof(abs(zero(T))*ν)
    ord = order(space)
    @inbounds A₋ₙ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₋ₙ))
    if ord == 0
        s .= abs.(A₋ₙ)
    else
        @inbounds s .= abs.(selectdim(A, N, 2ord+1)) .+ abs.(A₋ₙ)
        @inbounds for i ∈ ord-1:-1:1
            s .= s .* ν .+ abs.(selectdim(A, N, ord+1-i)) .+ abs.(selectdim(A, N, ord+1+i))
        end
        @inbounds s .= s .* ν .+ abs.(selectdim(A, N, ord+1))
    end
    return s
end

function _apply(X::EllInf{<:AlgebraicWeight}, space::Fourier, A::AbstractVector)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * _getindex(weight(X), space, 0)
    @inbounds for i ∈ 1:ord
        x = _getindex(weight(X), space, i)
        s = max(s, abs(A[ord+1+i]) * x, abs(A[ord+1-i]) * x)
    end
    return s
end
function _apply(X::EllInf{<:AlgebraicWeight}, space::Fourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))*_getindex(weight(X), space, 0))
    ord = order(space)
    @inbounds A₀ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:ord
        x = _getindex(weight(X), space, i)
        s .= max.(s, abs.(selectdim(A, N, ord+1-i)) .* x, abs.(selectdim(A, N, ord+1+i)) .* x)
    end
    return s
end
function _apply_dual(X::EllInf{<:AlgebraicWeight}, space::Fourier, A::AbstractVector)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) / _getindex(weight(X), space, 0)
    @inbounds for i ∈ 1:ord
        s += (abs(A[ord+1-i]) + abs(A[ord+1+i])) / _getindex(weight(X), space, i)
    end
    return s
end
function _apply_dual(X::EllInf{<:AlgebraicWeight}, space::Fourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))/_getindex(weight(X), space, 0))
    ord = order(space)
    @inbounds A₀ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    @inbounds s .= abs.(A₀)
    @inbounds for i ∈ 1:ord
        s .+= (abs.(selectdim(A, N, ord+1-i)) .+ abs.(selectdim(A, N, ord+1+i))) ./ _getindex(weight(X), space, i)
    end
    return s
end

# Chebyshev

_apply(::Ell1{IdentityWeight}, ::Chebyshev, A::AbstractVector) =
    @inbounds abs(A[1]) + exact(2) * sum(abs, view(A, 2:length(A)))
function _apply(::Ell1{IdentityWeight}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T)))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .+= abs.(selectdim(A, N, i+1))
        end
        @inbounds s .= exact(2) .* s .+ abs.(selectdim(A, N, 1))
    end
    return s
end
_apply_dual(::Ell1{IdentityWeight}, ::Chebyshev, A::AbstractVector) =
    @inbounds max(abs(A[1]), maximum(abs, view(A, 2:length(A))) / exact(2))
function _apply_dual(::Ell1{IdentityWeight}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T)))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .= max.(s, abs.(selectdim(A, N, i+1)))
        end
        @inbounds s .= max.(s ./ exact(2), abs.(selectdim(A, N, 1)))
    end
    return s
end

function _apply(X::Ell1{<:GeometricWeight}, space::Chebyshev, A::AbstractVector)
    ν = rate(weight(X))
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * one(ν)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s = s * ν + abs(A[i+1])
        end
        @inbounds s = (exact(2) * ν) * s + abs(A[1])
    end
    return s
end
function _apply(X::Ell1{<:GeometricWeight}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    ν = rate(weight(X))
    CoefType = typeof(abs(zero(T))*ν)
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .= s .* ν .+ abs.(selectdim(A, N, i+1))
        end
        @inbounds s .= (exact(2) * ν) .* s .+ abs.(selectdim(A, N, 1))
    end
    return s
end
function _apply_dual(X::Ell1{<:GeometricWeight}, space::Chebyshev, A::AbstractVector{T}) where {T}
    ν = inv(rate(weight(X)))
    νⁱ½ = one(ν) / exact(2)
    @inbounds s = abs(A[1]) * one(νⁱ½)
    @inbounds for i ∈ 1:order(space)
        νⁱ½ *= ν
        s = max(s, abs(A[i+1]) * νⁱ½)
    end
    return s
end
function _apply_dual(X::Ell1{<:GeometricWeight}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    ν = inv(rate(weight(X)))
    νⁱ½ = one(ν) / exact(2)
    CoefType = typeof(abs(zero(T))*νⁱ½)
    @inbounds A₀ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:order(space)
        νⁱ½ *= ν
        s .= max.(s, abs.(selectdim(A, N, i+1)) .* νⁱ½)
    end
    return s
end

function _apply(X::Ell1{<:AlgebraicWeight}, space::Chebyshev, A::AbstractVector)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * _getindex(weight(X), space, ord)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s += abs(A[i+1]) * _getindex(weight(X), space, i)
        end
        @inbounds s = exact(2) * s + abs(A[1])
    end
    return s
end
function _apply(X::Ell1{<:AlgebraicWeight}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))*_getindex(weight(X), space, 0))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ) .* _getindex(weight(X), space, ord)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .+= abs.(selectdim(A, N, i+1)) .* _getindex(weight(X), space, i)
        end
        @inbounds s .= exact(2) .* s .+ abs.(selectdim(A, N, 1))
    end
    return s
end
function _apply_dual(X::Ell1{<:AlgebraicWeight}, space::Chebyshev, A::AbstractVector)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) / _getindex(weight(X), space, ord)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s = max(s, abs(A[i+1]) / _getindex(weight(X), space, i))
        end
        @inbounds s = max(s / exact(2), abs(A[1]))
    end
    return s
end
function _apply_dual(X::Ell1{<:AlgebraicWeight}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))/_getindex(weight(X), space, 0))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ) ./ _getindex(weight(X), space, ord)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .= max.(s, abs.(selectdim(A, N, i+1)) ./ _getindex(weight(X), space, i))
        end
        @inbounds s .= max.(s ./ exact(2), abs.(selectdim(A, N, 1)))
    end
    return s
end

_apply(::Ell2{IdentityWeight}, ::Chebyshev, A::AbstractVector) =
    @inbounds sqrt(abs2(A[1]) + exact(2) * sum(abs2, view(A, 2:length(A))))
function _apply(::Ell2{IdentityWeight}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(sqrt(abs2(zero(T))))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs2.(Aᵢ)
    @inbounds for i ∈ ord-1:-1:1
        s .+= abs2.(selectdim(A, N, i+1))
    end
    @inbounds s .= sqrt.(exact(2) .* s .+ abs2.(selectdim(A, N, 1)))
    return s
end
_apply_dual(::Ell2{IdentityWeight}, ::Chebyshev, A::AbstractVector) =
    @inbounds sqrt(abs2(A[1]) + sum(abs2, view(A, 2:length(A))) / exact(2))
function _apply_dual(::Ell2{IdentityWeight}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(sqrt(abs2(zero(T))))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs2.(Aᵢ)
    @inbounds for i ∈ ord-1:-1:1
        s .+= abs2.(selectdim(A, N, i+1))
    end
    @inbounds s .= sqrt.(s ./ exact(2) .+ abs2.(selectdim(A, N, 1)))
    return s
end

_apply(::EllInf{IdentityWeight}, space::Chebyshev, A::AbstractVector) =
    @inbounds max(abs(A[1]), exact(2) * maximum(abs, view(A, 2:length(A))))
function _apply(::EllInf{IdentityWeight}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T)))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .= max.(s, abs.(selectdim(A, N, i+1)))
        end
        @inbounds s .= max.(exact(2) .* s, abs.(selectdim(A, N, 1)))
    end
    return s
end
_apply_dual(::EllInf{IdentityWeight}, space::Chebyshev, A::AbstractVector) =
    @inbounds abs(A[1]) + sum(abs, view(A, 2:length(A))) / exact(2)
function _apply_dual(::EllInf{IdentityWeight}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T)))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .+= abs.(selectdim(A, N, i+1))
        end
        @inbounds s .= s ./ exact(2) .+ abs.(selectdim(A, N, 1))
    end
    return s
end

function _apply(X::EllInf{<:GeometricWeight}, space::Chebyshev, A::AbstractVector)
    ν = rate(weight(X))
    νⁱ2 = exact(2) * one(ν)
    @inbounds s = abs(A[1]) * one(νⁱ2)
    @inbounds for i ∈ 1:order(space)
        νⁱ2 *= ν
        s = max(s, abs(A[i+1]) * νⁱ2)
    end
    return s
end
function _apply(X::EllInf{<:GeometricWeight}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    ν = rate(weight(X))
    νⁱ2 = exact(2) * one(ν)
    CoefType = typeof(abs(zero(T))*νⁱ2)
    @inbounds A₀ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:order(space)
        νⁱ2 *= ν
        s .= max.(s, abs.(selectdim(A, N, i+1)) .* νⁱ2)
    end
    return s
end
function _apply_dual(X::EllInf{<:GeometricWeight}, space::Chebyshev, A::AbstractVector)
    ν = inv(rate(weight(X)))
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * one(ν)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s = s * ν + abs(A[i+1])
        end
        @inbounds s = s * (ν / exact(2)) + abs(A[1])
    end
    return s
end
function _apply_dual(X::EllInf{<:GeometricWeight}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    ν = inv(rate(weight(X)))
    CoefType = typeof(abs(zero(T))*ν)
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .= s .* ν .+ abs.(selectdim(A, N, i+1))
        end
        @inbounds s .= s .* (ν / exact(2)) .+ abs.(selectdim(A, N, 1))
    end
    return s
end

function _apply(X::EllInf{<:AlgebraicWeight}, space::Chebyshev, A::AbstractVector)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * _getindex(weight(X), space, ord)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s = max(s, abs(A[i+1]) * _getindex(weight(X), space, i))
        end
        @inbounds s = max(exact(2) * s, abs(A[1]))
    end
    return s
end
function _apply(X::EllInf{<:AlgebraicWeight}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))*_getindex(weight(X), space, 0))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ) .* _getindex(weight(X), space, ord)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .= max.(s, abs.(selectdim(A, N, i+1)) .* _getindex(weight(X), space, i))
        end
        @inbounds s .= max.(exact(2) .* s, abs.(selectdim(A, N, 1)))
    end
    return s
end
function _apply_dual(X::EllInf{<:AlgebraicWeight}, space::Chebyshev, A::AbstractVector)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) / _getindex(weight(X), space, ord)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s += abs(A[i+1]) / _getindex(weight(X), space, i)
        end
        @inbounds s = s / exact(2) + abs(A[1])
    end
    return s
end
function _apply_dual(X::EllInf{<:AlgebraicWeight}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))/_getindex(weight(X), space, 0))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ) ./ _getindex(weight(X), space, ord)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .+= abs.(selectdim(A, N, i+1)) ./ _getindex(weight(X), space, i)
        end
        @inbounds s .= s ./ exact(2) .+ abs.(selectdim(A, N, 1))
    end
    return s
end

# CartesianSpace

_apply(::Ell1{IdentityWeight}, ::CartesianSpace, A::AbstractVector) = sum(abs, A)
_apply_dual(::Ell1{IdentityWeight}, space::CartesianSpace, A::AbstractVector) = _apply(EllInf(IdentityWeight()), space, A)

_apply(::Ell2{IdentityWeight}, ::CartesianSpace, A::AbstractVector) = sqrt(sum(abs2, A))
_apply_dual(::Ell2{IdentityWeight}, ::CartesianSpace, A::AbstractVector) = sqrt(sum(abs2, A))

_apply(::EllInf{IdentityWeight}, ::CartesianSpace, A::AbstractVector) = maximum(abs, A)
_apply_dual(::EllInf{IdentityWeight}, space::CartesianSpace, A::AbstractVector) = _apply(Ell1(IdentityWeight()), space, A)

for X ∈ (:Ell1, :EllInf)
    @eval begin
        function norm(a::Sequence{<:CartesianSpace}, X::NormedCartesianSpace{<:BanachSpace,$X{IdentityWeight}})
            0 < nspaces(space(a)) || return throw(ArgumentError("number of cartesian products must be strictly positive"))
            return _norm(a, X)
        end
        function norm(a::Sequence{<:CartesianSpace}, X::NormedCartesianSpace{<:NTuple{N,BanachSpace},$X{IdentityWeight}}) where {N}
            0 < nspaces(space(a)) == N || return throw(ArgumentError("number of cartesian products must be strictly positive and equal to the number of inner Banach spaces"))
            return _norm(a, X)
        end

        function opnorm(A::LinearOperator{<:CartesianSpace,ParameterSpace}, X::NormedCartesianSpace{<:BanachSpace,$X{IdentityWeight}})
            0 < nspaces(domain(A)) || return throw(ArgumentError("number of cartesian products must be strictly positive"))
            return _opnorm(A, X)
        end
        function opnorm(A::LinearOperator{<:CartesianSpace,ParameterSpace}, X::NormedCartesianSpace{<:NTuple{N,BanachSpace},$X{IdentityWeight}}) where {N}
            0 < nspaces(domain(A)) == N || return throw(ArgumentError("number of cartesian products must be strictly positive and equal to the number of inner Banach spaces"))
            return _opnorm(A, X)
        end
    end
end

function norm(a::Sequence{<:CartesianSpace}, X::NormedCartesianSpace{<:BanachSpace,Ell2{IdentityWeight}})
    0 < nspaces(space(a)) || return throw(ArgumentError("number of cartesian products must be strictly positive"))
    return sqrt(_norm2(a, X))
end
function norm(a::Sequence{<:CartesianSpace}, X::NormedCartesianSpace{<:NTuple{N,BanachSpace},Ell2{IdentityWeight}}) where {N}
    0 < nspaces(space(a)) == N || return throw(ArgumentError("number of cartesian products must be strictly positive and equal to the number of inner Banach spaces"))
    return sqrt(_norm2(a, X))
end

function opnorm(A::LinearOperator{<:CartesianSpace,ParameterSpace}, X::NormedCartesianSpace{<:BanachSpace,Ell2{IdentityWeight}})
    0 < nspaces(domain(A)) || return throw(ArgumentError("number of cartesian products must be strictly positive"))
    return sqrt(_opnorm2(A, X))
end
function opnorm(A::LinearOperator{<:CartesianSpace,ParameterSpace}, X::NormedCartesianSpace{<:NTuple{N,BanachSpace},Ell2{IdentityWeight}}) where {N}
    0 < nspaces(domain(A)) == N || return throw(ArgumentError("number of cartesian products must be strictly positive and equal to the number of inner Banach spaces"))
    return sqrt(_opnorm2(A, X))
end

# Ell1

function _norm(a::Sequence{<:CartesianPower}, X::NormedCartesianSpace{<:BanachSpace,Ell1{IdentityWeight}})
    @inbounds r = norm(component(a, 1), X.inner)
    @inbounds for i ∈ 2:nspaces(space(a))
        r += norm(component(a, i), X.inner)
    end
    return r
end
_norm(a::Sequence{CartesianProduct{T}}, X::NormedCartesianSpace{<:BanachSpace,Ell1{IdentityWeight}}) where {N,T<:NTuple{N,VectorSpace}} =
    @inbounds norm(component(a, 1), X.inner) + _norm(component(a, 2:N), X)
_norm(a::Sequence{<:CartesianProduct{<:Tuple{VectorSpace}}}, X::NormedCartesianSpace{<:BanachSpace,Ell1{IdentityWeight}}) =
    @inbounds norm(component(a, 1), X.inner)
_norm(a::Sequence{<:CartesianSpace}, X::NormedCartesianSpace{<:NTuple{N,BanachSpace},Ell1{IdentityWeight}}) where {N} =
    @inbounds norm(component(a, 1), X.inner[1]) + _norm(component(a, 2:N), NormedCartesianSpace(Base.tail(X.inner), X.outer))
_norm(a::Sequence{<:CartesianSpace}, X::NormedCartesianSpace{<:Tuple{BanachSpace},Ell1{IdentityWeight}}) =
    @inbounds norm(component(a, 1), X.inner[1])

function _opnorm(A::LinearOperator{<:CartesianPower,ParameterSpace}, X::NormedCartesianSpace{<:BanachSpace,Ell1{IdentityWeight}})
    @inbounds r = opnorm(component(A, 1), X.inner)
    @inbounds for i ∈ 2:nspaces(domain(A))
        r = max(r, opnorm(component(A, i), X.inner))
    end
    return r
end
_opnorm(A::LinearOperator{CartesianProduct{T},ParameterSpace}, X::NormedCartesianSpace{<:BanachSpace,Ell1{IdentityWeight}}) where {N,T<:NTuple{N,VectorSpace}} =
    @inbounds max(opnorm(component(A, 1), X.inner), _opnorm(component(A, 2:N), X))
_opnorm(A::LinearOperator{<:CartesianProduct{<:Tuple{VectorSpace}},ParameterSpace}, X::NormedCartesianSpace{<:BanachSpace,Ell1{IdentityWeight}}) =
    @inbounds opnorm(component(A, 1), X.inner)
_opnorm(A::LinearOperator{<:CartesianSpace,ParameterSpace}, X::NormedCartesianSpace{<:NTuple{N,BanachSpace},Ell1{IdentityWeight}}) where {N} =
    @inbounds max(opnorm(component(A, 1), X.inner[1]), _opnorm(component(A, 2:N), NormedCartesianSpace(Base.tail(X.inner), X.outer)))
_opnorm(A::LinearOperator{<:CartesianSpace,ParameterSpace}, X::NormedCartesianSpace{<:Tuple{BanachSpace},Ell1{IdentityWeight}}) =
    @inbounds opnorm(component(A, 1), X.inner[1])

# Ell2

function _norm2(a::Sequence{<:CartesianPower}, X::NormedCartesianSpace{<:BanachSpace,Ell2{IdentityWeight}})
    @inbounds v = norm(component(a, 1), X.inner)
    r = v*v
    @inbounds for i ∈ 2:nspaces(space(a))
        v = norm(component(a, i), X.inner)
        r += v*v
    end
    return r
end
function _norm2(a::Sequence{CartesianProduct{T}}, X::NormedCartesianSpace{<:BanachSpace,Ell2{IdentityWeight}}) where {N,T<:NTuple{N,VectorSpace}}
    @inbounds v = norm(component(a, 1), X.inner)
    return @inbounds v*v + _norm2(component(a, 2:N), X)
end
function _norm2(a::Sequence{<:CartesianProduct{<:Tuple{VectorSpace}}}, X::NormedCartesianSpace{<:BanachSpace,Ell2{IdentityWeight}})
    @inbounds v = norm(component(a, 1), X.inner)
    return v*v
end
function _norm2(a::Sequence{<:CartesianSpace}, X::NormedCartesianSpace{<:NTuple{N,BanachSpace},Ell2{IdentityWeight}}) where {N}
    @inbounds v = norm(component(a, 1), X.inner[1])
    return @inbounds v*v + _norm2(component(a, 2:N), NormedCartesianSpace(Base.tail(X.inner), X.outer))
end
function _norm2(a::Sequence{<:CartesianSpace}, X::NormedCartesianSpace{<:Tuple{BanachSpace},Ell2{IdentityWeight}})
    @inbounds v = norm(component(a, 1), X.inner[1])
    return v*v
end

function _opnorm2(A::LinearOperator{<:CartesianPower,ParameterSpace}, X::NormedCartesianSpace{<:BanachSpace,Ell2{IdentityWeight}})
    @inbounds v = opnorm(component(A, 1), X.inner)
    r = v*v
    @inbounds for i ∈ 2:nspaces(domain(A))
        v = opnorm(component(A, i), X.inner)
        r += v*v
    end
    return r
end
function _opnorm2(A::LinearOperator{CartesianProduct{T},ParameterSpace}, X::NormedCartesianSpace{<:BanachSpace,Ell2{IdentityWeight}}) where {N,T<:NTuple{N,VectorSpace}}
    @inbounds v = opnorm(component(A, 1), X.inner)
    return @inbounds v*v + _opnorm2(component(A, 2:N), X)
end
function _opnorm2(A::LinearOperator{<:CartesianProduct{<:Tuple{VectorSpace}},ParameterSpace}, X::NormedCartesianSpace{<:BanachSpace,Ell2{IdentityWeight}})
    @inbounds v = opnorm(component(A, 1), X.inner)
    return v*v
end
function _opnorm2(A::LinearOperator{<:CartesianSpace,ParameterSpace}, X::NormedCartesianSpace{<:NTuple{N,BanachSpace},Ell2{IdentityWeight}}) where {N}
    @inbounds v = opnorm(component(A, 1), X.inner[1])
    return @inbounds v*v + _opnorm2(component(A, 2:N), NormedCartesianSpace(Base.tail(X.inner), X.outer))
end
function _opnorm2(A::LinearOperator{<:CartesianSpace,ParameterSpace}, X::NormedCartesianSpace{<:Tuple{BanachSpace},Ell2{IdentityWeight}})
    @inbounds v = opnorm(component(A, 1), X.inner[1])
    return v*v
end

# EllInf

function _norm(a::Sequence{<:CartesianPower}, X::NormedCartesianSpace{<:BanachSpace,EllInf{IdentityWeight}})
    @inbounds r = norm(component(a, 1), X.inner)
    @inbounds for i ∈ 2:nspaces(space(a))
        r = max(r, norm(component(a, i), X.inner))
    end
    return r
end
_norm(a::Sequence{CartesianProduct{T}}, X::NormedCartesianSpace{<:BanachSpace,EllInf{IdentityWeight}}) where {N,T<:NTuple{N,VectorSpace}} =
    @inbounds max(norm(component(a, 1), X.inner), _norm(component(a, 2:N), X))
_norm(a::Sequence{<:CartesianProduct{<:Tuple{VectorSpace}}}, X::NormedCartesianSpace{<:BanachSpace,EllInf{IdentityWeight}}) =
    @inbounds norm(component(a, 1), X.inner)
_norm(a::Sequence{<:CartesianSpace}, X::NormedCartesianSpace{<:NTuple{N,BanachSpace},EllInf{IdentityWeight}}) where {N} =
    @inbounds max(norm(component(a, 1), X.inner[1]), _norm(component(a, 2:N), NormedCartesianSpace(Base.tail(X.inner), X.outer)))
_norm(a::Sequence{<:CartesianSpace}, X::NormedCartesianSpace{<:Tuple{BanachSpace},EllInf{IdentityWeight}}) =
    @inbounds norm(component(a, 1), X.inner[1])

function _opnorm(A::LinearOperator{<:CartesianPower,ParameterSpace}, X::NormedCartesianSpace{<:BanachSpace,EllInf{IdentityWeight}})
    @inbounds r = opnorm(component(A, 1), X.inner)
    @inbounds for i ∈ 2:nspaces(domain(A))
        r += opnorm(component(A, i), X.inner)
    end
    return r
end
_opnorm(A::LinearOperator{CartesianProduct{T},ParameterSpace}, X::NormedCartesianSpace{<:BanachSpace,EllInf{IdentityWeight}}) where {N,T<:NTuple{N,VectorSpace}} =
    @inbounds opnorm(component(A, 1), X.inner) + _opnorm(component(A, 2:N), X)
_opnorm(A::LinearOperator{<:CartesianProduct{<:Tuple{VectorSpace}},ParameterSpace}, X::NormedCartesianSpace{<:BanachSpace,EllInf{IdentityWeight}}) =
    @inbounds opnorm(component(A, 1), X.inner)
_opnorm(A::LinearOperator{<:CartesianSpace,ParameterSpace}, X::NormedCartesianSpace{<:NTuple{N,BanachSpace},EllInf{IdentityWeight}}) where {N} =
    @inbounds opnorm(component(A, 1), X.inner[1]) + _opnorm(component(A, 2:N), NormedCartesianSpace(Base.tail(X.inner), X.outer))
_opnorm(A::LinearOperator{<:CartesianSpace,ParameterSpace}, X::NormedCartesianSpace{<:Tuple{BanachSpace},EllInf{IdentityWeight}}) =
    @inbounds opnorm(component(A, 1), X.inner[1])










####

_apply(::Ell1{IdentityWeight}, ::CosFourier, A::AbstractVector) =
    @inbounds abs(A[1]) + exact(2) * sum(abs, view(A, 2:length(A)))
function _apply(::Ell1{IdentityWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T)))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .+= abs.(selectdim(A, N, i+1))
        end
        @inbounds s .= exact(2) .* s .+ abs.(selectdim(A, N, 1))
    end
    return s
end
_apply_dual(::Ell1{IdentityWeight}, ::CosFourier, A::AbstractVector) =
    @inbounds max(abs(A[1]), maximum(abs, view(A, 2:length(A))) / exact(2))
function _apply_dual(::Ell1{IdentityWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T)))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .= max.(s, abs.(selectdim(A, N, i+1)))
        end
        @inbounds s .= max.(s ./ exact(2), abs.(selectdim(A, N, 1)))
    end
    return s
end

function _apply(X::Ell1{<:GeometricWeight}, space::CosFourier, A::AbstractVector)
    ν = rate(weight(X))
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * one(ν)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s = s * ν + abs(A[i+1])
        end
        @inbounds s = (exact(2) * ν) * s + abs(A[1])
    end
    return s
end
function _apply(X::Ell1{<:GeometricWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    ν = rate(weight(X))
    CoefType = typeof(abs(zero(T))*ν)
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .= s .* ν .+ abs.(selectdim(A, N, i+1))
        end
        @inbounds s .= (exact(2) * ν) .* s .+ abs.(selectdim(A, N, 1))
    end
    return s
end
function _apply_dual(X::Ell1{<:GeometricWeight}, space::CosFourier, A::AbstractVector{T}) where {T}
    ν = inv(rate(weight(X)))
    νⁱ½ = one(ν) / exact(2)
    @inbounds s = abs(A[1]) * one(νⁱ½)
    @inbounds for i ∈ 1:order(space)
        νⁱ½ *= ν
        s = max(s, abs(A[i+1]) * νⁱ½)
    end
    return s
end
function _apply_dual(X::Ell1{<:GeometricWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    ν = inv(rate(weight(X)))
    νⁱ½ = one(ν) / exact(2)
    CoefType = typeof(abs(zero(T))*νⁱ½)
    @inbounds A₀ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:order(space)
        νⁱ½ *= ν
        s .= max.(s, abs.(selectdim(A, N, i+1)) .* νⁱ½)
    end
    return s
end

function _apply(X::Ell1{<:AlgebraicWeight}, space::CosFourier, A::AbstractVector)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * _getindex(weight(X), space, ord)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s += abs(A[i+1]) * _getindex(weight(X), space, i)
        end
        @inbounds s = exact(2) * s + abs(A[1])
    end
    return s
end
function _apply(X::Ell1{<:AlgebraicWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))*_getindex(weight(X), space, 0))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ) .* _getindex(weight(X), space, ord)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .+= abs.(selectdim(A, N, i+1)) .* _getindex(weight(X), space, i)
        end
        @inbounds s .=  exact(2) .* s .+ abs.(selectdim(A, N, 1))
    end
    return s
end
function _apply_dual(X::Ell1{<:AlgebraicWeight}, space::CosFourier, A::AbstractVector)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) / _getindex(weight(X), space, ord)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s = max(s, abs(A[i+1]) / _getindex(weight(X), space, i))
        end
        @inbounds s = max(s / exact(2), abs(A[1]))
    end
    return s
end
function _apply_dual(X::Ell1{<:AlgebraicWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))/_getindex(weight(X), space, 0))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ) ./ _getindex(weight(X), space, ord)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .= max.(s, abs.(selectdim(A, N, i+1)) ./ _getindex(weight(X), space, i))
        end
        @inbounds s .= max.(s ./ exact(2), abs.(selectdim(A, N, 1)))
    end
    return s
end

_apply(::Ell2{IdentityWeight}, ::CosFourier, A::AbstractVector) =
    @inbounds sqrt(abs2(A[1]) +  exact(2) * sum(abs2, view(A, 2:length(A))))
function _apply(::Ell2{IdentityWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(sqrt(abs2(zero(T))))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs2.(Aᵢ)
    for i ∈ ord-1:-1:1
        s .+= abs2.(selectdim(A, N, i+1))
    end
    @inbounds s .= sqrt.(exact(2) .* s .+ abs2.(selectdim(A, N, 1)))
    return s
end
_apply_dual(::Ell2{IdentityWeight}, ::CosFourier, A::AbstractVector) =
    @inbounds sqrt(abs2(A[1]) + sum(abs2, view(A, 2:length(A))) / exact(2))
function _apply_dual(::Ell2{IdentityWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(sqrt(abs2(zero(T))))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs2.(Aᵢ)
    for i ∈ ord-1:-1:1
        s .+= abs2.(selectdim(A, N, i+1))
    end
    @inbounds s .= sqrt.(s ./ exact(2) .+ abs2.(selectdim(A, N, 1)))
    return s
end

_apply(::EllInf{IdentityWeight}, ::CosFourier, A::AbstractVector) =
    @inbounds max(abs(A[1]),  exact(2) * maximum(abs, view(A, 2:length(A))))
function _apply(::EllInf{IdentityWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T)))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .= max.(s, abs.(selectdim(A, N, i+1)))
        end
        @inbounds s .= max.(exact(2) .* s, abs.(selectdim(A, N, 1)))
    end
    return s
end
_apply_dual(::EllInf{IdentityWeight}, ::CosFourier, A::AbstractVector) =
    @inbounds abs(A[1]) + sum(abs, view(A, 2:length(A))) / exact(2)
function _apply_dual(::EllInf{IdentityWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T)))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .+= abs.(selectdim(A, N, i+1))
        end
        @inbounds s .= s ./ exact(2) .+ abs.(selectdim(A, N, 1))
    end
    return s
end

function _apply(X::EllInf{<:GeometricWeight}, space::CosFourier, A::AbstractVector)
    ν = rate(weight(X))
    νⁱ2 = exact(2) * one(ν)
    @inbounds s = abs(A[1]) * one(νⁱ2)
    @inbounds for i ∈ 1:order(space)
        νⁱ2 *= ν
        s = max(s, abs(A[i+1]) * νⁱ2)
    end
    return s
end
function _apply(X::EllInf{<:GeometricWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    ν = rate(weight(X))
    νⁱ2 = exact(2) * one(ν)
    CoefType = typeof(abs(zero(T))*νⁱ2)
    @inbounds A₀ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:order(space)
        νⁱ2 *= ν
        s .= max.(s, abs.(selectdim(A, N, i+1)) .* νⁱ2)
    end
    return s
end
function _apply_dual(X::EllInf{<:GeometricWeight}, space::CosFourier, A::AbstractVector)
    ν = inv(rate(weight(X)))
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * one(ν)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s = s * ν + abs(A[i+1])
        end
        @inbounds s = s * (ν / exact(2)) + abs(A[1])
    end
    return s
end
function _apply_dual(X::EllInf{<:GeometricWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    ν = inv(rate(weight(X)))
    CoefType = typeof(abs(zero(T))*ν)
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .= s .* ν .+ abs.(selectdim(A, N, i+1))
        end
        @inbounds s .= s .* (ν / exact(2)) .+ abs.(selectdim(A, N, 1))
    end
    return s
end

function _apply(X::EllInf{<:AlgebraicWeight}, space::CosFourier, A::AbstractVector)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * _getindex(weight(X), space, ord)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s = max(s, abs(A[i+1]) * _getindex(weight(X), space, i))
        end
        @inbounds s = max(exact(2) * s, abs(A[1]))
    end
    return s
end
function _apply(X::EllInf{<:AlgebraicWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))*_getindex(weight(X), space, 0))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ) .* _getindex(weight(X), space, ord)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .= max.(s, abs.(selectdim(A, N, i+1)) .* _getindex(weight(X), space, i))
        end
        @inbounds s .= max.(exact(2) .* s, abs.(selectdim(A, N, 1)))
    end
    return s
end
function _apply_dual(X::EllInf{<:AlgebraicWeight}, space::CosFourier, A::AbstractVector)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) / _getindex(weight(X), space, ord)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s += abs(A[i+1]) / _getindex(weight(X), space, i)
        end
        @inbounds s = s / exact(2) + abs(A[1])
    end
    return s
end
function _apply_dual(X::EllInf{<:AlgebraicWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))/_getindex(weight(X), space, 0))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ) ./ _getindex(weight(X), space, ord)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .+= abs.(selectdim(A, N, i+1)) ./ _getindex(weight(X), space, i)
        end
        @inbounds s .= s ./ exact(2) .+ abs.(selectdim(A, N, 1))
    end
    return s
end





_apply(::Ell1{IdentityWeight}, ::SinFourier, A::AbstractVector) = exact(2) * sum(abs, A)
function _apply(::Ell1{IdentityWeight}, space::SinFourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T)))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    @inbounds for i ∈ ord-1:-1:1
        s .+= abs.(selectdim(A, N, i))
    end
    s .*= exact(2)
    return s
end
_apply_dual(::Ell1{IdentityWeight}, ::SinFourier, A::AbstractVector) = maximum(abs, A) / exact(2)
function _apply_dual(::Ell1{IdentityWeight}, space::SinFourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T)))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    @inbounds for i ∈ ord-1:-1:1
        s .= max.(s, abs.(selectdim(A, N, i)))
    end
    s ./= exact(2)
    return s
end

function _apply(X::Ell1{<:GeometricWeight}, space::SinFourier, A::AbstractVector)
    ν = rate(weight(X))
    ord = order(space)
    @inbounds s = abs(A[ord]) * one(ν)
    @inbounds for i ∈ ord-1:-1:1
        s = s * ν + abs(A[i])
    end
    s *= exact(2) * ν
    return s
end
function _apply(X::Ell1{<:GeometricWeight}, space::SinFourier, A::AbstractArray{T,N}) where {T,N}
    ν = rate(weight(X))
    CoefType = typeof(abs(zero(T))*ν)
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    @inbounds for i ∈ ord-1:-1:1
        s .= s .* ν .+ abs.(selectdim(A, N, i))
    end
    s .*= exact(2) * ν
    return s
end
function _apply_dual(X::Ell1{<:GeometricWeight}, space::SinFourier, A::AbstractVector{T}) where {T}
    ν = νⁱ = inv(rate(weight(X)))
    @inbounds s = abs(A[1]) * νⁱ
    @inbounds for i ∈ 2:order(space)
        νⁱ *= ν
        s = max(s, abs(A[i]) * νⁱ)
    end
    s /= exact(2)
    return s
end
function _apply_dual(X::Ell1{<:GeometricWeight}, space::SinFourier, A::AbstractArray{T,N}) where {T,N}
    ν = νⁱ = inv(rate(weight(X)))
    CoefType = typeof(abs(zero(T))*νⁱ)
    @inbounds A₁ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₁))
    s .= abs.(A₁) .* νⁱ
    @inbounds for i ∈ 2:order(space)
        νⁱ *= ν
        s .= max.(s, abs.(selectdim(A, N, i)) .* νⁱ)
    end
    s ./= exact(2)
    return s
end

_apply(::EllInf{IdentityWeight}, ::SinFourier, A::AbstractVector) = exact(2) * maximum(abs, A)
function _apply(::EllInf{IdentityWeight}, space::SinFourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T)))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    @inbounds for i ∈ ord-1:-1:1
        s .= max.(s, abs.(selectdim(A, N, i)))
    end
    s .*= exact(2)
    return s
end
_apply_dual(::EllInf{IdentityWeight}, ::SinFourier, A::AbstractVector) = sum(abs, A) / exact(2)
function _apply_dual(::EllInf{IdentityWeight}, space::SinFourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T)))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    @inbounds for i ∈ ord-1:-1:1
        s .+= abs.(selectdim(A, N, i))
    end
    s ./= exact(2)
    return s
end
