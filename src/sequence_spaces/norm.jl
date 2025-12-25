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
        norm(a::Sequence, X::$T) = _apply(X, space(a), coefficients(a))

        opnorm(A::LinearOperator{<:VectorSpace,ParameterSpace}, X::$T) =
            _apply_dual(X, domain(A), vec(coefficients(A)))
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

function _apply(X::Ell1, space::SequenceSpace, A::AbstractVector)
    s = abs(zero(eltype(A)))
    @inbounds for (i, k) ∈ enumerate(indices(space))
        w = _getindex(weight(X), space, k)
        iw = _intrinsic_weight(space, k)
        s += abs(A[i]) * w * iw
    end
    return s
end

function _apply_dual(X::Ell1, space::SequenceSpace, A::AbstractVector)
    s = abs(zero(eltype(A)))
    @inbounds for (i, k) ∈ enumerate(indices(space))
        w = _getindex(weight(X), space, k)
        iw = _intrinsic_weight(space, k)
        s = max(s, abs(A[i]) / (w * iw))
    end
    return s
end

function _apply(X::Ell2, space::SequenceSpace, A::AbstractVector)
    s = abs2(zero(eltype(A)))
    @inbounds for (i, k) ∈ enumerate(indices(space))
        w = _getindex(weight(X), space, k)
        iw = _intrinsic_weight(space, k)
        s += abs2(A[i]) * w * iw
    end
    return sqrt(s)
end

function _apply_dual(X::Ell2, space::SequenceSpace, A::AbstractVector)
    s = abs2(zero(eltype(A)))
    @inbounds for (i, k) ∈ enumerate(indices(space))
        w = _getindex(weight(X), space, k)
        iw = _intrinsic_weight(space, k)
        s += abs2(A[i]) / (w * iw)
    end
    return sqrt(s)
end

function _apply(X::EllInf, space::SequenceSpace, A::AbstractVector)
    s = abs(zero(eltype(A)))
    @inbounds for (i, k) ∈ enumerate(indices(space))
        w = _getindex(weight(X), space, k)
        iw = _intrinsic_weight(space, k)
        s = max(s, abs(A[i]) * w * iw)
    end
    return s
end

function _apply_dual(X::EllInf, space::SequenceSpace, A::AbstractVector)
    s = abs(zero(eltype(A)))
    @inbounds for (i, k) ∈ enumerate(indices(space))
        w = _getindex(weight(X), space, k)
        iw = _intrinsic_weight(space, k)
        s += abs(A[i]) / (w * iw)
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
