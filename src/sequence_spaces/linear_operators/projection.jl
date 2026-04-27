"""
    Projection{T<:VectorSpace,S<:Number} <: AbstractLinearOperator

Projection operator onto `space::T`. This is used to resize a sequence, a
linear operator, or materialize a finite projection of an abstract operator. The
type of the ouput entries is determined by promoting against `S`.

Field:
- `space :: T`
"""
struct Projection{T<:VectorSpace,S<:Number} <: AbstractLinearOperator
    space :: T
end

Projection(space::VectorSpace, ::Type{T}=Float64) where {T<:Number} = Projection{typeof(space),T}(space)

domain(P::Projection) = P.space # needed for general methods
function domain(P::Projection, s::VectorSpace)
    dom = P.space
    pdom, ps = _promote_space(dom, s)
    _iscompatible(pdom, ps) || return throw(ArgumentError("spaces must be compatible: projection space is $dom, codomain space is $s"))
    return dom
end

codomain(P::Projection) = P.space # needed for general methods
function codomain(P::Projection, s::VectorSpace)
    codom = P.space
    pcodom, ps = _promote_space(codom, s)
    _iscompatible(pcodom, ps) || return throw(ArgumentError("spaces must be compatible: projection space is $codom, domain space is $s"))
    return codom
end

coefficients(P::Projection) = project(UniformScalingOperator(one(eltype(P))), P.space, P.space) # needed for general methods

Base.eltype(::Projection{<:VectorSpace,S}) where {S<:Number} = S
Base.eltype(::Type{Projection{<:VectorSpace,S}}) where {S<:Number} = S
_coeftype(A::Projection, ::VectorSpace) = eltype(A)
_coeftype(A::Projection, ::VectorSpace, ::Type{T}) where {T} = promote_type(eltype(A), T)

IntervalArithmetic._infer_numtype(P::Projection) = numtype(eltype(P))
function IntervalArithmetic._interval_infsup(::Type{T}, P₁::Projection, P₂::Projection, d::IntervalArithmetic.Decoration) where {T<:IntervalArithmetic.NumTypes}
    @assert P₁.space == P₂.space
    return Projection(IntervalArithmetic._interval_infsup(T, P₁.space, P₂.space, d), Interval{T})
end

# materializing methods

#- action

Base.:*(P::Projection, a::AbstractSequence) = project(a, P.space, promote_type(eltype(P), eltype(a)))

#- arithmetic

Base.:+(P::Projection, A::LinearOperator) = project(P, P.space, P.space) + A
Base.:+(A::LinearOperator, P::Projection) = A + project(P, P.space, P.space)
Base.:-(P::Projection, A::LinearOperator) = project(P, P.space, P.space) - A
Base.:-(A::LinearOperator, P::Projection) = A - project(P, P.space, P.space)

Base.:*(P₁::Projection, P₂::Projection) = Projection(intersect(P₁.space, P₂.space))
Base.:∘(P₁::Projection, P₂::Projection) = Projection(intersect(P₁.space, P₂.space))

Base.:*(A::LinearOperator, P::Projection) = project(A, P.space, codomain(A), promote_type(eltype(P), eltype(A))) # needed to resolve method ambiguity
Base.:*(P::Projection, A::LinearOperator) = project(A, domain(A), P.space, promote_type(eltype(P), eltype(A))) # needed to resolve method ambiguity

Base.:*(A::AbstractLinearOperator, P::Projection) = project(A, P.space, codomain(A, P.space), _coeftype(A, P.space, eltype(P)))
Base.:*(P::Projection, A::AbstractLinearOperator) = _lproj(A, domain(A, P.space), P)
_lproj(A::AbstractLinearOperator, domain::VectorSpace, P::Projection) = project(A, domain, P.space, _coeftype(A, domain, eltype(P)))
_lproj(A::AbstractLinearOperator, ::EmptySpace, P::Projection) = ComposedOperator(P, A)

#- also trigger materilization

Base.:*(P::Add{<:Projection,<:Projection}, A::AbstractLinearOperator) = P.A * A + P.B * A
Base.:*(A::AbstractLinearOperator, P::Add{<:Projection,<:Projection}) = A * P.A + A * P.B
Base.:*(P::Add{<:Projection,<:Negate{<:Projection}}, A::AbstractLinearOperator) = P.A * A + P.B * A
Base.:*(A::AbstractLinearOperator, P::Add{<:Projection,<:Negate{<:Projection}}) = A * P.A + A * P.B
Base.:*(P::Add{<:Negate{<:Projection},<:Projection}, A::AbstractLinearOperator) = P.A * A + P.B * A
Base.:*(A::AbstractLinearOperator, P::Add{<:Negate{<:Projection},<:Projection}) = A * P.A + A * P.B
Base.:*(P::Add{<:Negate{<:Projection},<:Negate{<:Projection}}, A::AbstractLinearOperator) = P.A * A + P.B * A
Base.:*(A::AbstractLinearOperator, P::Add{<:Negate{<:Projection},<:Negate{<:Projection}}) = A * P.A + A * P.B

Base.:*(P::Negate{<:Projection}, A::AbstractLinearOperator) = -(P.A * A)
Base.:*(A::AbstractLinearOperator, P::Negate{<:Projection}) = -(A * P.A)



#

"""
    project(a::AbstractSequence, space_dest::VectorSpace, ::Type{T}=eltype(a))

Represent `a` as a [`Sequence`](@ref) in `space_dest`.

See also: [`project!`](@ref).
"""
project(a::AbstractSequence, space_dest::VectorSpace, ::Type{T}=eltype(a)) where {T} = project!(zeros(T, space_dest), a)

"""
    project!(c::Sequence, a::Sequence)

Represent `a` as a [`Sequence`](@ref) in `space(c)`. The result is stored in `c`
by overwriting it.

See also: [`project`](@ref).
"""
function project!(c::Sequence, a::Sequence)
    psc, psa = _promote_space(space(c), space(a))
    _iscompatible(psc, psa) || return throw(ArgumentError("spaces must be compatible: c has space $(space(c)), a has space $(space(a))"))
    coefficients(c) .= zero(eltype(c))
    _project!(Sequence(psc, coefficients(c)), Sequence(psa, coefficients(a)))
    return c
end

"""
    project(a::Sequence, ::ScalarSpace, codomain_dest::VectorSpace, ::Type{T}=eltype(a))

Represent `a` as a [`LinearOperator`](@ref) from `ScalarSpace` to `codomain_dest`.

See also: [`project!`](@ref).
"""
project(a::Sequence, domain_dest::ScalarSpace, codomain_dest::VectorSpace, ::Type{T}=eltype(a)) where {T} =
    project!(zeros(T, domain_dest, codomain_dest), a)

"""
    project!(C::LinearOperator{ScalarSpace,<:VectorSpace}, a::Sequence)

Represent `a` as a [`LinearOperator`](@ref) from `ScalarSpace` to `codomain(C)`.
The result is stored in `C` by overwriting it.

See also: [`project`](@ref).
"""
function project!(C::LinearOperator{ScalarSpace,<:VectorSpace}, a::Sequence)
    project!(Sequence(codomain(C), vec(coefficients(C))), a)
    return C
end

_project!(c::Sequence, a::Sequence) = _radd!(c, a)

#

"""
    project(A::AbstractLinearOperator, domain_dest::VectorSpace, codomain_dest::VectorSpace, ::Type{T}=eltype(A))

Represent `A` as a [`LinearOperator`](@ref) from `domain_dest` to `codomain_dest`.

See also: [`project!`](@ref).
"""
project(A::AbstractLinearOperator, domain_dest::VectorSpace, codomain_dest::VectorSpace, ::Type{T}=_coeftype(A, domain_dest)) where {T} =
    project!(zeros(T, domain_dest, codomain_dest), A)

"""
    project!(C::LinearOperator, A::AbstractLinearOperator)

Represent `A` as a [`LinearOperator`](@ref) from `domain(C)` to `codomain(C)`.
The result is stored in `C` by overwriting it.

See also: [`project`](@ref).
"""
function project!(C::LinearOperator, A::AbstractLinearOperator)
    pcodomC, pcodomA = _promote_space(codomain(C), codomain(A, domain(C)))
    _iscompatible(pcodomC, pcodomA) || return throw(ArgumentError("spaces must be compatible"))
    coefficients(C) .= zero(eltype(C))
    _project!(LinearOperator(domain(C), pcodomC, coefficients(C)), A)
    return C
end
function project!(C::LinearOperator, A::LinearOperator)
    pdomC, pdomA = _promote_space(domain(C), domain(A))
    pcodomC, pcodomA = _promote_space(codomain(C), codomain(A))
    _iscompatible(pdomC, pdomA) & _iscompatible(pcodomC, pcodomA) || return throw(ArgumentError("spaces must be compatible"))
    coefficients(C) .= zero(eltype(C))
    _project!(LinearOperator(pdomC, pcodomC, coefficients(C)), LinearOperator(pdomA, pcodomA, coefficients(A)))
    return C
end

project!(C::LinearOperator, J::UniformScaling) = project!(C, UniformScalingOperator(J))

# isomorphism concept between `ScalarSpace` and `SequenceSpace`
# the logic is a promotion system

_promote_space(s₁::VectorSpace, s₂::VectorSpace) = (s₁, s₂)
_promote_space(s::SequenceSpace, ::ScalarSpace) = (s, _zero_space(s))
_promote_space(::ScalarSpace, s::SequenceSpace) = (_zero_space(s), s)
function _promote_space(s₁::CartesianPower, s₂::CartesianPower)
    nspaces(s₁) == nspaces(s₂) || return (s₁, s₂)
    u, v = _promote_space(space(s₁), space(s₂))
    return CartesianPower(u, nspaces(s₁)), CartesianPower(v, nspaces(s₂))
end
function _promote_space(s₁::CartesianProduct{<:NTuple{N,VectorSpace}}, s₂::CartesianProduct{<:NTuple{N,VectorSpace}}) where {N}
    u, v = __promote_space((), (), spaces(s₁), spaces(s₂))
    return CartesianProduct(u), CartesianProduct(v)
end
function _promote_space(s₁::CartesianProduct{<:NTuple{N,VectorSpace}}, s₂::CartesianPower) where {N}
    N == nspaces(s₂) || return (s₁, s₂)
    u, v = __promote_space((), (), spaces(s₁), space(s₂))
    return CartesianProduct(u), CartesianProduct(v)
end
function _promote_space(s₁::CartesianPower, s₂::CartesianProduct{<:NTuple{N,VectorSpace}}) where {N}
    nspaces(s₁) == N || return (s₁, s₂)
    u, v = __promote_space((), (), space(s₁), spaces(s₂))
    return CartesianProduct(u), CartesianProduct(v)
end

function __promote_space(t1::Tuple, t2::Tuple, s1::NTuple{N,VectorSpace}, s2::NTuple{N,VectorSpace}) where {N}
    u, v = _promote_space(first(s1), first(s2))
    return __promote_space((t1..., u), (t2..., v), Base.tail(s1), Base.tail(s2))
end
function __promote_space(t1::Tuple, t2::Tuple, s1::Tuple{VectorSpace}, s2::Tuple{VectorSpace})
    u, v = _promote_space(first(s1), first(s2))
    return (t1..., u), (t2..., v)
end
function __promote_space(t1::Tuple, t2::Tuple, s1::NTuple{N,VectorSpace}, s2::VectorSpace) where {N}
    u, v = _promote_space(first(s1), s2)
    return __promote_space((t1..., u), (t2..., v), Base.tail(s1), s2)
end
function __promote_space(t1::Tuple, t2::Tuple, s1::Tuple{VectorSpace}, s2::VectorSpace)
    u, v = _promote_space(first(s1), s2)
    return (t1..., u), (t2..., v)
end
function __promote_space(t1::Tuple, t2::Tuple, s1::VectorSpace, s2::NTuple{N,VectorSpace}) where {N}
    u, v = _promote_space(s1, first(s2))
    return __promote_space((t1..., u), (t2..., v), s1, Base.tail(s2))
end
function __promote_space(t1::Tuple, t2::Tuple, s1::VectorSpace, s2::Tuple{VectorSpace})
    u, v = _promote_space(s1, first(s2))
    return (t1..., u), (t2..., v)
end

_zero_space(s::TensorSpace) = TensorSpace(map(_zero_space, s))
_zero_space(::Taylor) = Taylor(0)
_zero_space(s::Fourier) = Fourier(0, frequency(s))
_zero_space(::Chebyshev) = Chebyshev(0)
_zero_space(s::SymmetricSpace) = SymmetricSpace(_zero_space(desymmetrize(s)), _sym_with_cst_coef(symmetry(s)))

"""
    project(A::LinearOperator{ScalarSpace,<:VectorSpace}, space_dest::VectorSpace, ::Type{T}=eltype(A))

Represent `A` as a [`Sequence`](@ref) in `space_dest`.

See also: [`project!`](@ref).
"""
project(A::LinearOperator{ScalarSpace,<:VectorSpace}, space_dest::VectorSpace, ::Type{T}=eltype(A)) where {T} =
    project!(zeros(T, space_dest), A)

"""
    project!(c::Sequence, A::LinearOperator{ScalarSpace,<:VectorSpace})

Represent `A` as a [`Sequence`](@ref) in `space(c)`. The result is stored in `c`
by overwriting it.

See also: [`project`](@ref).
"""
function project!(c::Sequence, A::LinearOperator{ScalarSpace,<:VectorSpace})
    project!(LinearOperator(domain(A), space(c), reshape(coefficients(c), :, 1)), A)
    return c
end



#

function _project!(C::LinearOperator, A::AbstractLinearOperator)
    for j ∈ indices(domain(C)), i ∈ indices(codomain(C))
        C[i,j] = getcoefficient(A, (codomain(C), i), (domain(C), j), eltype(C))
    end
    return C
end
function _project!(C::LinearOperator, A::AbstractDiagonalOperator)
    for i ∈ indices(domain(C) ∩ codomain(C))
        C[i,i] = getcoefficient(A, (codomain(C), i), (domain(C), i), eltype(C))
    end
    return C
end

_project!(C::LinearOperator, A::LinearOperator) = _radd!(C, A)

_project!(A::LinearOperator, P::Projection) = radd!(A, coefficients(P))

_project!(A::LinearOperator, J::UniformScalingOperator) = radd!(A, J)
_project!(A::LinearOperator, J::UniformScaling) = _project!(A, UniformScalingOperator(J))

_project!(A::LinearOperator, B::Add) = add!(A, B.A, B.B)
_project!(A::LinearOperator, B::Negate) = rsub!(A, B.A)
_project!(A::LinearOperator, B::ComposedOperator) = mul!(A, B.outer, B.inner, exact(true), exact(false))



#

function Base.:*(P::Projection{<:CartesianSpace}, v::Vector)
    @assert nspaces(P.space) == length(v)
    u = [Projection(P.space[i], eltype(P)) * vᵢ for (i, vᵢ) ∈ enumerate(v)]
    return Sequence(P.space, vec(mapreduce(coefficients, vcat, u)))
end

Base.:*(P::Projection{<:CartesianSpace}, v::LinearAlgebra.Diagonal) = P * Matrix(v)
Base.:*(v::LinearAlgebra.Diagonal, P::Projection{<:CartesianSpace}) = Matrix(v) * P

function Base.:*(P::Projection{<:CartesianSpace}, v::Matrix)
    @assert nspaces(P.space) == size(v, 1)
    u = [Projection(P.space[i], eltype(P)) * v[i,j] for i ∈ axes(v, 1), j ∈ axes(v, 2)]
    dom = CartesianProduct([reduce(_union, [domain(u[i,j], P.space[i]) for i ∈ 1:size(v, 1)]) for j ∈ 1:size(v, 2)]...)
    any(sᵢ -> sᵢ isa EmptySpace, dom.spaces) && return u
    CoefType = reduce(promote_type, [reduce(promote_type, [eltype(u[i,j]) for i ∈ 1:size(v, 1)]) for j ∈ 1:size(v, 2)])
    u_ = zeros(CoefType, dom, P.space)
    @inbounds for j ∈ 1:size(v, 2), i ∈ 1:size(v, 1)
        project!(block(u_, i, j), u[i,j])
    end
    return u_
end

function Base.:*(v::Matrix, P::Projection{<:CartesianSpace})
    @assert nspaces(P.space) == size(v, 2)
    u = [v[i,j] * Projection(P.space[j], eltype(P)) for i ∈ axes(v, 1), j ∈ axes(v, 2)]
    codom = CartesianProduct([reduce(union, [codomain(u[i,j]) for j ∈ 1:size(v, 2)]) for i ∈ 1:size(v, 1)]...)
    CoefType = reduce(promote_type, [reduce(promote_type, [eltype(u[i,j]) for j ∈ 1:size(v, 2)]) for i ∈ 1:size(v, 1)])
    u_ = zeros(CoefType, P.space, codom)
    @inbounds for j ∈ 1:size(v, 2), i ∈ 1:size(v, 1)
        project!(block(u_, i, j), u[i,j])
    end
    return u_
end





#

function project(a::InfiniteSequence, space_dest::SequenceSpace, ::Type{T}=eltype(a)) where {T}
    c = zeros(T, space_dest)
    project!(c, a)

    discarded = copy(sequence(a))
    @inbounds view(discarded, indices(space_dest ∩ space(a))) .= zero(T)

    X = banachspace(a)
    return _unsafe_infinite_sequence(c, norm(c, X), sequence_error(a) + norm(discarded, X), a.full_norm, X)
end

function project!(c::Sequence, a::InfiniteSequence)
    project!(c, sequence(a))
    X = banachspace(a)
    ord_a = order(a)
    space_c = space(c)
    CoefType = eltype(c)
    @inbounds for k ∈ indices(space_c)
        if any(abs.(k) .> ord_a)
            w_k = _getindex(weight(X), space_c, k)
            c[k] = _to_interval(CoefType, sup(sequence_error(a) / w_k))
        end
    end
    return c
end





# # tail

# tail(a::Sequence, order) = tail!(copy(a), order)

# _tail_order_getindex(order::Number, i) = order

# _tail_order_getindex(order, i) = order[i]

# function tail!(a::Sequence{<:CartesianSpace}, order)
#     for i ∈ 1:nspaces(space(a))
#         tail!(block(a, i), _tail_order_getindex(order, i))
#     end
#     return a
# end

# function tail!(a::Sequence{ScalarSpace}, order)
#     if order ≥ 0
#         a[1] = zero(eltype(a))
#     end
#     return a
# end

# function tail!(a::Sequence{<:TensorSpace}, order)
#     for α ∈ indices(space(a))
#         if all(abs.(α) .≤ order)
#             a[α] = zero(eltype(a))
#         end
#     end
#     return a
# end

# function tail!(a::Sequence{<:BaseSpace}, order)
#     for α ∈ indices(space(a))
#         if abs(α) ≤ order
#             a[α] = zero(eltype(a))
#         end
#     end
#     return a
# end
