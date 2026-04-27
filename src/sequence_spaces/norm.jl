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
[`opnorm(::LinearOperator{<:VectorSpace,ScalarSpace}, ::BanachSpace)`](@ref).
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
[`opnorm(::LinearOperator{<:VectorSpace,ScalarSpace}, ::BanachSpace)`](@ref).
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
    return opnorm(LinearOperator(domain(A), ScalarSpace(), transpose(v)), X)
end

"""
    opnorm(A::LinearOperator, X::BanachSpace)

Compute the operator norm of `A` where `X` is the Banach space corresponding to
both `domain(A)` and `codomain(A)`.

See also: [`opnorm(::LinearOperator, ::Real=Inf)`](@ref),
[`opnorm(::LinearOperator, ::BanachSpace, ::BanachSpace)`](@ref) and
[`opnorm(::LinearOperator{<:VectorSpace,ScalarSpace}, ::BanachSpace)`](@ref).
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

"""
    opnorm(A::LinearOperator{<:VectorSpace,ScalarSpace}, X::BanachSpace)

Compute the operator norm of `A` where `X` is the Banach space corresponding to
`domain(A)`.

See also: [`opnorm(::LinearOperator, ::Real=Inf)`](@ref),
[`opnorm(::LinearOperator, ::BanachSpace, ::BanachSpace)`](@ref) and
[`opnorm(::LinearOperator, ::BanachSpace)`](@ref).
"""
opnorm(::LinearOperator{<:VectorSpace,ScalarSpace}, ::BanachSpace)

for T ∈ (:Ell1, :Ell2, :EllInf)
    @eval begin
        norm(a::Sequence, X::$T) = _norm(a, X)

        opnorm(A::LinearOperator{<:VectorSpace,ScalarSpace}, X::$T) =
            _norm_dual(Sequence(domain(A), vec(coefficients(A))), X)
    end
end

# ScalarSpace

_norm(a::Sequence{ScalarSpace}, ::Ell1{IdentityWeight}) = @inbounds abs(a[1])
_norm_dual(a::Sequence{ScalarSpace}, ::Ell1{IdentityWeight}) = @inbounds abs(a[1])

_norm(a::Sequence{ScalarSpace}, ::Ell2{IdentityWeight}) = @inbounds abs(a[1])
_norm_dual(a::Sequence{ScalarSpace}, ::Ell2{IdentityWeight}) = @inbounds abs(a[1])

_norm(a::Sequence{ScalarSpace}, ::EllInf{IdentityWeight}) = @inbounds abs(a[1])
_norm_dual(a::Sequence{ScalarSpace}, ::EllInf{IdentityWeight}) = @inbounds abs(a[1])

# SequenceSpace

function _norm(a::Sequence{<:SequenceSpace}, X::Ell1)
    space_a = space(a)
    A = coefficients(a)
    s = abs(zero(eltype(A)))
    @inbounds for (i, k) ∈ enumerate(indices(space_a))
        w = _getindex(weight(X), space_a, k)
        s += abs(A[i]) * w
    end
    return s
end

function _norm_dual(a::Sequence{<:SequenceSpace}, X::Ell1)
    space_a = space(a)
    A = coefficients(a)
    s = abs(zero(eltype(A)))
    @inbounds for (i, k) ∈ enumerate(indices(space_a))
        w = _getindex(weight(X), space_a, k)
        s = max(s, abs(A[i]) / w)
    end
    return s
end

function _norm(a::Sequence{<:SequenceSpace}, X::Ell2)
    space_a = space(a)
    A = coefficients(a)
    s = abs2(zero(eltype(A)))
    @inbounds for (i, k) ∈ enumerate(indices(space_a))
        w = _getindex(weight(X), space_a, k)
        s += abs2(A[i]) * w
    end
    return sqrt(s)
end

function _norm_dual(a::Sequence{<:SequenceSpace}, X::Ell2)
    space_a = space(a)
    A = coefficients(a)
    s = abs2(zero(eltype(A)))
    @inbounds for (i, k) ∈ enumerate(indices(space_a))
        w = _getindex(weight(X), space_a, k)
        s += abs2(A[i]) / w
    end
    return sqrt(s)
end

function _norm(a::Sequence{<:SequenceSpace}, X::EllInf)
    space_a = space(a)
    A = coefficients(a)
    s = abs(zero(eltype(A)))
    @inbounds for (i, k) ∈ enumerate(indices(space_a))
        w = _getindex(weight(X), space_a, k)
        s = max(s, abs(A[i]) * w)
    end
    return s
end

function _norm_dual(a::Sequence{<:SequenceSpace}, X::EllInf)
    space_a = space(a)
    A = coefficients(a)
    s = abs(zero(eltype(A)))
    @inbounds for (i, k) ∈ enumerate(indices(space_a))
        w = _getindex(weight(X), space_a, k)
        s += abs(A[i]) / w
    end
    return s
end

# CartesianSpace

for X ∈ (:Ell1, :EllInf)
    @eval begin
        norm(a::Sequence{<:CartesianSpace}, X::$X) = norm(a, NormedCartesianSpace(X, X))
        function norm(a::Sequence{<:CartesianSpace}, X::NormedCartesianSpace{<:BanachSpace,$X{IdentityWeight}})
            0 < nspaces(space(a)) || return throw(ArgumentError("number of cartesian products must be strictly positive"))
            return _norm(a, X)
        end
        function norm(a::Sequence{<:CartesianSpace}, X::NormedCartesianSpace{<:NTuple{N,BanachSpace},$X{IdentityWeight}}) where {N}
            0 < nspaces(space(a)) == N || return throw(ArgumentError("number of cartesian products must be strictly positive and equal to the number of inner Banach spaces"))
            return _norm(a, X)
        end

        opnorm(A::LinearOperator{<:CartesianSpace,ScalarSpace}, X::$X) = opnorm(A, NormedCartesianSpace(X, X))
        function opnorm(A::LinearOperator{<:CartesianSpace,ScalarSpace}, X::NormedCartesianSpace{<:BanachSpace,$X{IdentityWeight}})
            0 < nspaces(domain(A)) || return throw(ArgumentError("number of cartesian products must be strictly positive"))
            return _opnorm(A, X)
        end
        function opnorm(A::LinearOperator{<:CartesianSpace,ScalarSpace}, X::NormedCartesianSpace{<:NTuple{N,BanachSpace},$X{IdentityWeight}}) where {N}
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

function opnorm(A::LinearOperator{<:CartesianSpace,ScalarSpace}, X::NormedCartesianSpace{<:BanachSpace,Ell2{IdentityWeight}})
    0 < nspaces(domain(A)) || return throw(ArgumentError("number of cartesian products must be strictly positive"))
    return sqrt(_opnorm2(A, X))
end
function opnorm(A::LinearOperator{<:CartesianSpace,ScalarSpace}, X::NormedCartesianSpace{<:NTuple{N,BanachSpace},Ell2{IdentityWeight}}) where {N}
    0 < nspaces(domain(A)) == N || return throw(ArgumentError("number of cartesian products must be strictly positive and equal to the number of inner Banach spaces"))
    return sqrt(_opnorm2(A, X))
end

# Ell1

function _norm(a::Sequence{<:CartesianPower}, X::NormedCartesianSpace{<:BanachSpace,Ell1{IdentityWeight}})
    @inbounds r = norm(block(a, 1), X.inner)
    @inbounds for i ∈ 2:nspaces(space(a))
        r += norm(block(a, i), X.inner)
    end
    return r
end
_norm(a::Sequence{CartesianProduct{T}}, X::NormedCartesianSpace{<:BanachSpace,Ell1{IdentityWeight}}) where {N,T<:NTuple{N,VectorSpace}} =
    @inbounds norm(block(a, 1), X.inner) + _norm(block(a, 2:N), X)
_norm(a::Sequence{<:CartesianProduct{<:Tuple{VectorSpace}}}, X::NormedCartesianSpace{<:BanachSpace,Ell1{IdentityWeight}}) =
    @inbounds norm(block(a, 1), X.inner)
_norm(a::Sequence{<:CartesianSpace}, X::NormedCartesianSpace{<:NTuple{N,BanachSpace},Ell1{IdentityWeight}}) where {N} =
    @inbounds norm(block(a, 1), X.inner[1]) + _norm(block(a, 2:N), NormedCartesianSpace(Base.tail(X.inner), X.outer))
_norm(a::Sequence{<:CartesianSpace}, X::NormedCartesianSpace{<:Tuple{BanachSpace},Ell1{IdentityWeight}}) =
    @inbounds norm(block(a, 1), X.inner[1])

function _opnorm(A::LinearOperator{<:CartesianPower,ScalarSpace}, X::NormedCartesianSpace{<:BanachSpace,Ell1{IdentityWeight}})
    @inbounds r = opnorm(block(A, 1), X.inner)
    @inbounds for i ∈ 2:nspaces(domain(A))
        r = max(r, opnorm(block(A, i), X.inner))
    end
    return r
end
_opnorm(A::LinearOperator{CartesianProduct{T},ScalarSpace}, X::NormedCartesianSpace{<:BanachSpace,Ell1{IdentityWeight}}) where {N,T<:NTuple{N,VectorSpace}} =
    @inbounds max(opnorm(block(A, 1), X.inner), _opnorm(block(A, 2:N), X))
_opnorm(A::LinearOperator{<:CartesianProduct{<:Tuple{VectorSpace}},ScalarSpace}, X::NormedCartesianSpace{<:BanachSpace,Ell1{IdentityWeight}}) =
    @inbounds opnorm(block(A, 1), X.inner)
_opnorm(A::LinearOperator{<:CartesianSpace,ScalarSpace}, X::NormedCartesianSpace{<:NTuple{N,BanachSpace},Ell1{IdentityWeight}}) where {N} =
    @inbounds max(opnorm(block(A, 1), X.inner[1]), _opnorm(block(A, 2:N), NormedCartesianSpace(Base.tail(X.inner), X.outer)))
_opnorm(A::LinearOperator{<:CartesianSpace,ScalarSpace}, X::NormedCartesianSpace{<:Tuple{BanachSpace},Ell1{IdentityWeight}}) =
    @inbounds opnorm(block(A, 1), X.inner[1])

# Ell2

function _norm2(a::Sequence{<:CartesianPower}, X::NormedCartesianSpace{<:BanachSpace,Ell2{IdentityWeight}})
    @inbounds v = norm(block(a, 1), X.inner)
    r = v*v
    @inbounds for i ∈ 2:nspaces(space(a))
        v = norm(block(a, i), X.inner)
        r += v*v
    end
    return r
end
function _norm2(a::Sequence{CartesianProduct{T}}, X::NormedCartesianSpace{<:BanachSpace,Ell2{IdentityWeight}}) where {N,T<:NTuple{N,VectorSpace}}
    @inbounds v = norm(block(a, 1), X.inner)
    return @inbounds v*v + _norm(block(a, 2:N), X)
end
function _norm2(a::Sequence{<:CartesianProduct{<:Tuple{VectorSpace}}}, X::NormedCartesianSpace{<:BanachSpace,Ell2{IdentityWeight}})
    @inbounds v = norm(block(a, 1), X.inner)
    return v*v
end
function _norm2(a::Sequence{<:CartesianSpace}, X::NormedCartesianSpace{<:NTuple{N,BanachSpace},Ell2{IdentityWeight}}) where {N}
    @inbounds v = norm(block(a, 1), X.inner[1])
    return @inbounds v*v + _norm(block(a, 2:N), NormedCartesianSpace(Base.tail(X.inner), X.outer))
end
function _norm2(a::Sequence{<:CartesianSpace}, X::NormedCartesianSpace{<:Tuple{BanachSpace},Ell2{IdentityWeight}})
    @inbounds v = norm(block(a, 1), X.inner[1])
    return v*v
end

function _opnorm2(A::LinearOperator{<:CartesianPower,ScalarSpace}, X::NormedCartesianSpace{<:BanachSpace,Ell2{IdentityWeight}})
    @inbounds v = opnorm(block(A, 1), X.inner)
    r = v*v
    @inbounds for i ∈ 2:nspaces(domain(A))
        v = opnorm(block(A, i), X.inner)
        r += v*v
    end
    return r
end
function _opnorm2(A::LinearOperator{CartesianProduct{T},ScalarSpace}, X::NormedCartesianSpace{<:BanachSpace,Ell2{IdentityWeight}}) where {N,T<:NTuple{N,VectorSpace}}
    @inbounds v = opnorm(block(A, 1), X.inner)
    return @inbounds v*v + _opnorm(block(A, 2:N), X)
end
function _opnorm2(A::LinearOperator{<:CartesianProduct{<:Tuple{VectorSpace}},ScalarSpace}, X::NormedCartesianSpace{<:BanachSpace,Ell2{IdentityWeight}})
    @inbounds v = opnorm(block(A, 1), X.inner)
    return v*v
end
function _opnorm2(A::LinearOperator{<:CartesianSpace,ScalarSpace}, X::NormedCartesianSpace{<:NTuple{N,BanachSpace},Ell2{IdentityWeight}}) where {N}
    @inbounds v = opnorm(block(A, 1), X.inner[1])
    return @inbounds v*v + _opnorm(block(A, 2:N), NormedCartesianSpace(Base.tail(X.inner), X.outer))
end
function _opnorm2(A::LinearOperator{<:CartesianSpace,ScalarSpace}, X::NormedCartesianSpace{<:Tuple{BanachSpace},Ell2{IdentityWeight}})
    @inbounds v = opnorm(block(A, 1), X.inner[1])
    return v*v
end

# EllInf

function _norm(a::Sequence{<:CartesianPower}, X::NormedCartesianSpace{<:BanachSpace,EllInf{IdentityWeight}})
    @inbounds r = norm(block(a, 1), X.inner)
    @inbounds for i ∈ 2:nspaces(space(a))
        r = max(r, norm(block(a, i), X.inner))
    end
    return r
end
_norm(a::Sequence{CartesianProduct{T}}, X::NormedCartesianSpace{<:BanachSpace,EllInf{IdentityWeight}}) where {N,T<:NTuple{N,VectorSpace}} =
    @inbounds max(norm(block(a, 1), X.inner), _norm(block(a, 2:N), X))
_norm(a::Sequence{<:CartesianProduct{<:Tuple{VectorSpace}}}, X::NormedCartesianSpace{<:BanachSpace,EllInf{IdentityWeight}}) =
    @inbounds norm(block(a, 1), X.inner)
_norm(a::Sequence{<:CartesianSpace}, X::NormedCartesianSpace{<:NTuple{N,BanachSpace},EllInf{IdentityWeight}}) where {N} =
    @inbounds max(norm(block(a, 1), X.inner[1]), _norm(block(a, 2:N), NormedCartesianSpace(Base.tail(X.inner), X.outer)))
_norm(a::Sequence{<:CartesianSpace}, X::NormedCartesianSpace{<:Tuple{BanachSpace},EllInf{IdentityWeight}}) =
    @inbounds norm(block(a, 1), X.inner[1])

function _opnorm(A::LinearOperator{<:CartesianPower,ScalarSpace}, X::NormedCartesianSpace{<:BanachSpace,EllInf{IdentityWeight}})
    @inbounds r = opnorm(block(A, 1), X.inner)
    @inbounds for i ∈ 2:nspaces(domain(A))
        r += opnorm(block(A, i), X.inner)
    end
    return r
end
_opnorm(A::LinearOperator{CartesianProduct{T},ScalarSpace}, X::NormedCartesianSpace{<:BanachSpace,EllInf{IdentityWeight}}) where {N,T<:NTuple{N,VectorSpace}} =
    @inbounds opnorm(block(A, 1), X.inner) + _opnorm(block(A, 2:N), X)
_opnorm(A::LinearOperator{<:CartesianProduct{<:Tuple{VectorSpace}},ScalarSpace}, X::NormedCartesianSpace{<:BanachSpace,EllInf{IdentityWeight}}) =
    @inbounds opnorm(block(A, 1), X.inner)
_opnorm(A::LinearOperator{<:CartesianSpace,ScalarSpace}, X::NormedCartesianSpace{<:NTuple{N,BanachSpace},EllInf{IdentityWeight}}) where {N} =
    @inbounds opnorm(block(A, 1), X.inner[1]) + _opnorm(block(A, 2:N), NormedCartesianSpace(Base.tail(X.inner), X.outer))
_opnorm(A::LinearOperator{<:CartesianSpace,ScalarSpace}, X::NormedCartesianSpace{<:Tuple{BanachSpace},EllInf{IdentityWeight}}) =
    @inbounds opnorm(block(A, 1), X.inner[1])





#

function norm(a::InfiniteSequence, X::BanachSpace = banachspace(a))
    X == banachspace(a) || return a.full_norm
    _issubspace(banachspace(a), X) || return throw(DomainError((X, banachspace(a)), "X cannot be a smaller Banach space than the one associated to a"))
    return min(norm(sequence(a), X) + sequence_error(a), a.full_norm)
end

_issubspace(::BanachSpace, ::BanachSpace) = false
_issubspace(::Ell1{IdentityWeight}, ::Ell1{IdentityWeight}) = true
_issubspace(X::Ell1{<:GeometricWeight}, Y::Ell1{<:GeometricWeight}) = rate(X) ≥ rate(Y)
_issubspace(X::Ell1{<:GeometricWeight}, ::Ell1{IdentityWeight}) = rate(X) ≥ 1
_issubspace(X::Ell1{IdentityWeight}, ::Ell1{<:GeometricWeight}) = true
_issubspace(X::Ell1{<:NTuple{N,Weight}}, Y::Ell1{<:NTuple{N,Weight}}) where {N} = mapreduce((u, w) -> _issubspace(Ell1(u), Ell1(w)), &, weight(X), weight(Y))
