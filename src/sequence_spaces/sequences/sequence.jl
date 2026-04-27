"""
    AbstractSequence

Abstract type for all sequences.
"""
abstract type AbstractSequence end

Base.broadcastable(a::AbstractSequence) = Ref(a)

# order, frequency

order(a::AbstractSequence) = order(space(a))
order(a::AbstractSequence, i::Int) = order(space(a), i)

frequency(a::AbstractSequence) = frequency(space(a))
frequency(a::AbstractSequence, i::Int) = frequency(space(a), i)

# utilities

Base.firstindex(a::AbstractSequence) = _firstindex(space(a))

Base.lastindex(a::AbstractSequence) = _lastindex(space(a))

Base.length(a::AbstractSequence) = length(coefficients(a))

Base.size(a::AbstractSequence) = size(coefficients(a)) # necessary for broadcasting

Base.iterate(a::AbstractSequence) = iterate(coefficients(a))
Base.iterate(a::AbstractSequence, i) = iterate(coefficients(a), i)

# getindex, view

Base.@propagate_inbounds function Base.getindex(a::AbstractSequence, α)
    space_a = space(a)
    @boundscheck(_checkbounds_indices(α, space_a) || throw(BoundsError(indices(space_a), α)))
    return getindex(coefficients(a), _findposition(α, space_a))
end
Base.@propagate_inbounds function Base.getindex(a::AbstractSequence, u::Union{AbstractVector,TensorIndices})
    v = Vector{eltype(a)}(undef, length(u))
    for (i, uᵢ) ∈ enumerate(u)
        v[i] = a[uᵢ]
    end
    return v
end

Base.@propagate_inbounds function Base.view(a::AbstractSequence, α)
    space_a = space(a)
    @boundscheck(_checkbounds_indices(α, space_a) || throw(BoundsError(indices(space_a), α)))
    return view(coefficients(a), _findposition(α, space_a))
end

#

"""
    Sequence{T<:VectorSpace,S<:AbstractVector} <: AbstractSequence

Compactly supported sequence in the given space.

Fields:
- `space :: T`
- `coefficients :: S`

Constructors:
- `Sequence(::VectorSpace, ::AbstractVector)`
- `Sequence(coefficients::AbstractVector)`: equivalent to `Sequence(ScalarSpace()^length(coefficients), coefficients)`

# Examples

```jldoctest
julia> Sequence(Taylor(2), [1, 2, 1]) # 1 + 2x + x^2
Sequence in Taylor(2) with coefficients Vector{Int64}:
 1
 2
 1

julia> Sequence(Taylor(1) ⊗ Fourier(1, 1.0), [0.5, 0.5, 0.0, 0.0, 0.5, 0.5]) # (1 + x) cos(y)
Sequence in Taylor(1) ⊗ Fourier(1, 1.0) with coefficients Vector{Float64}:
 0.5
 0.5
 0.0
 0.0
 0.5
 0.5

julia> Sequence([1, 2, 3])
Sequence in 𝕂³ with coefficients Vector{Int64}:
 1
 2
 3
```
"""
struct Sequence{T<:VectorSpace,S<:AbstractVector} <: AbstractSequence
    space :: T
    coefficients :: S
    function Sequence{T,S}(space::T, coefficients::S) where {T<:VectorSpace,S<:AbstractVector}
        l = length(coefficients)
        Base.OneTo(l) == eachindex(coefficients) || return throw(ArgumentError("offset vectors are not supported"))
        d = dimension(space)
        d == l || return throw(DimensionMismatch("dimensions must match: space has dimension $d, coefficients has length $l"))
        return new{T,S}(space, coefficients)
    end
end

Sequence(space::T, coefficients::S) where {T<:VectorSpace,S<:AbstractVector} =
    Sequence{T,S}(space, coefficients)

Sequence(coefficient::Number) = Sequence(ScalarSpace(), [coefficient])
Sequence(coefficients::AbstractVector) =
    Sequence(ScalarSpace()^length(coefficients), coefficients)

space(a::Sequence) = a.space

coefficients(a::Sequence) = a.coefficients

# to allow a[...] .= f.(...)
Base.@propagate_inbounds Base.Broadcast.dotview(a::Sequence, α) = view(a, α)

# utilities

Base.eltype(a::Sequence) = eltype(coefficients(a))
Base.eltype(::Type{<:Sequence{<:VectorSpace,T}}) where {T<:AbstractVector} = eltype(T)

Base.:(==)(a::Sequence, b::Sequence) =
    space(a) == space(b) && coefficients(a) == coefficients(b)

Base.iszero(a::Sequence) = iszero(coefficients(a))

Base.isapprox(a::Sequence, b::Sequence; kwargs...) =
    space(a) == space(b) && isapprox(coefficients(a), coefficients(b); kwargs...)

Base.copy(a::Sequence) = Sequence(space(a), copy(coefficients(a)))

Base.similar(a::Sequence) = Sequence(space(a), similar(coefficients(a)))
Base.similar(a::Sequence, ::Type{T}) where {T} = Sequence(space(a), similar(coefficients(a), T))

Base.zeros(s::VectorSpace) = Sequence(s, zeros(dimension(s)))
Base.zeros(::Type{T}, s::VectorSpace) where {T} = Sequence(s, zeros(T, dimension(s)))

Base.ones(s::VectorSpace) = Sequence(s, ones(dimension(s)))
Base.ones(::Type{T}, s::VectorSpace) where {T} = Sequence(s, ones(T, dimension(s)))

Base.fill(value, s::VectorSpace) = Sequence(s, fill(value, dimension(s)))

function Base.fill!(a::Sequence, value)
    fill!(coefficients(a), value)
    return a
end

IntervalArithmetic._infer_numtype(a::Sequence) = numtype(eltype(a))
IntervalArithmetic._interval_infsup(::Type{T}, a::Sequence, b::Sequence, d::IntervalArithmetic.Decoration) where {T<:IntervalArithmetic.NumTypes} =
    Sequence(IntervalArithmetic._interval_infsup(T, space(a), space(b), d), IntervalArithmetic._interval_infsup(T, coefficients(a), coefficients(b), d))

Base.reverse(a::Sequence; dims = :) = Sequence(space(a), reverse(coefficients(a); dims = dims))

Base.reverse!(a::Sequence; dims = :) = Sequence(space(a), reverse!(coefficients(a); dims = dims))

Base.zero(a::Sequence) = zeros(eltype(a), space(a))
Base.zero(::Type{Sequence{T,S}}) where {T<:VectorSpace,S<:AbstractVector} = zeros(eltype(S), _zero_space(T))
_zero_space(::Type{TensorSpace{T}}) where {T<:Tuple} = TensorSpace(map(_zero_space, fieldtypes(T)))
_zero_space(::Type{Taylor}) = Taylor(0)
_zero_space(::Type{Fourier{T}}) where {T<:Real} = Fourier(0, one(T))
_zero_space(::Type{Chebyshev}) = Chebyshev(0)

Base.one(a::Sequence{ScalarSpace}) = Sequence(space(a), [one(eltype(a))])
function Base.one(a::Sequence{<:SequenceSpace})
    new_space = _compatible_space_with_constant_index(space(a))
    CoefType = eltype(a)
    c = zeros(CoefType, new_space)
    @inbounds c[_findindex_constant(new_space)] = one(CoefType)
    return c
end
Base.one(::Type{Sequence{T,S}}) where {T<:VectorSpace,S<:AbstractVector} = ones(eltype(S), _zero_space(T))

Base.float(a::Sequence) = Sequence(_float_space(space(a)), float.(coefficients(a)))
Base.big(a::Sequence) = Sequence(_big_space(space(a)), big.(coefficients(a)))
IntervalArithmetic.mid(a::Sequence) = Sequence(_mid_space(space(a)), mid.(coefficients(a)))
for (f, g) ∈ ((:float, :_float_space), (:big, :_big_space), (:mid, :_mid_space))
    @eval begin
        $g(s::ScalarSpace) = s

        $g(s::TensorSpace) = TensorSpace(map($g, spaces(s)))

        $g(s::Taylor) = s
        $g(s::Fourier) = Fourier(order(s), $f(frequency(s)))
        $g(s::Chebyshev) = s

        $g(s::CartesianPower) = CartesianPower($g(space(s)), nspaces(s))
        $g(s::CartesianProduct) = CartesianProduct(map($g, spaces(s)))
    end
end

for f ∈ (:complex, :real, :imag, :conj)
    @eval Base.$f(a::Sequence) = Sequence(space(a), $f.(coefficients(a)))
end
Base.conj!(a::Sequence) = Sequence(space(a), conj!(coefficients(a)))
Base.complex(a::Sequence, b::Sequence) = Sequence(codomain(+, space(a), space(b)), complex.(coefficients(a), coefficients(b)))
Base.complex(::Type{Sequence{T,S}}) where {T<:VectorSpace,S<:AbstractVector} = Sequence{T,Vector{complex(eltype(S))}}

Base.permutedims(a::Sequence{<:TensorSpace}, σ::AbstractVector{<:Integer}) =
    Sequence(space(a)[σ], vec(permutedims(_no_alloc_reshape(coefficients(a), dimensions(space(a))), σ)))

# getindex, view

Base.@propagate_inbounds function Base.getindex(a::Sequence, α::VectorSpace)
    space_a = space(a)
    @boundscheck(_checkbounds_indices(α, space_a) || throw(BoundsError(space_a, α)))
    return Sequence(α, getindex(coefficients(a), _findposition(α, space_a))) # project(a, α)
end

Base.@propagate_inbounds function Base.view(a::Sequence, α::VectorSpace)
    space_a = space(a)
    @boundscheck(_checkbounds_indices(α, space_a) || throw(BoundsError(space_a, α)))
    return Sequence(α, view(coefficients(a), _findposition(α, space_a)))
end

# setindex!

Base.@propagate_inbounds function Base.setindex!(a::Sequence, x, α)
    space_a = space(a)
    @boundscheck(_checkbounds_indices(α, space_a) || throw(BoundsError(indices(space_a), α)))
    setindex!(coefficients(a), x, _findposition(α, space_a))
    return a
end
Base.@propagate_inbounds function Base.setindex!(a::Sequence, x, u::AbstractVector)
    for (i, uᵢ) ∈ enumerate(u)
        a[uᵢ] = x[i]
    end
    return a
end
Base.@propagate_inbounds function Base.setindex!(a::Sequence{<:TensorSpace}, x, u::TensorIndices)
    for (i, uᵢ) ∈ enumerate(u)
        a[uᵢ] = x[i]
    end
    return a
end

#

Base.@propagate_inbounds function Base.selectdim(a::Sequence{<:TensorSpace}, dim::Int, i)
    A = _no_alloc_reshape(coefficients(a), dimensions(space(a)))
    return selectdim(A, dim, _findposition(i, spaces(space(a))[dim]))
end

# assume compatibility

getcoefficient(a::Sequence, (s, α)::Tuple{VectorSpace,Any}) =
    _getcoefficient(a, α)

function getcoefficient(a::Sequence{<:SymmetricSpace}, (s, α)::Tuple{SymmetricSpace,Any})
    symmetry(space(a)) == symmetry(s) || return throw(DomainError((symmetry(space(a)), symmetry(s)), "symmetries must be equal"))
    return _getcoefficient(a, α)
end

function _getcoefficient(a::Sequence, α)
    _checkbounds_indices(α, space(a)) || return zero(eltype(a))
    return @inbounds a[α]
end

getcoefficient(a::Sequence{<:SymmetricSpace}, (s, α)::Tuple{NoSymSpace,Any}) = _desym_getcoefficient(a, α)

function _desym_getcoefficient(a::Sequence{<:SymmetricSpace}, α)
    CoefType = complex(eltype(a))
    _checkbounds_indices(α, desymmetrize(space(a))) || return zero(CoefType)
    k0, factor = _unsafe_get_representative_and_action(space(a), α)
    _checkbounds_indices(k0, space(a)) || return zero(CoefType)
    return @inbounds convert(CoefType, factor * a[k0])
end

_unsafe_get_representative_and_action(::NoSymSpace, k) = k, exact(true)
_unsafe_get_representative_and_action(s::SymmetricSpace, k) = @inbounds s.rep_idx_action[_findposition(k, desymmetrize(s))]

#

"""
    eachblock(a::Sequence{<:CartesianSpace})

Create a generator whose iterates yield each [`Sequence`](@ref) composing the block sequence.

# Examples

```jldoctest
julia> a = Sequence(Taylor(1)^2, [1, 2, 3, 4])
Sequence in Taylor(1)² with coefficients Vector{Int64}:
 1
 2
 3
 4

julia> m = eachblock(a)
Base.Generator{Base.OneTo{Int64}, RadiiPolynomial.var"#eachblock##0#eachblock##1"{Sequence{CartesianPower{Taylor}, Vector{Int64}}}}(RadiiPolynomial.var"#eachblock##0#eachblock##1"{Sequence{CartesianPower{Taylor}, Vector{Int64}}}(Sequence(Taylor(1)², [1, 2, 3, 4])), Base.OneTo(2))

julia> [v for v = m]
2-element Vector{Sequence{Taylor, SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true}}}:
 Sequence(Taylor(1), [1, 2])
 Sequence(Taylor(1), [3, 4])
```
"""
eachblock(a::Sequence{<:CartesianSpace}) =
    (@inbounds(block(a, i)) for i ∈ Base.OneTo(nspaces(space(a))))

"""
    block(a::Sequence{<:CartesianSpace})

Return the collection of blocks composing the sequence.

# Examples

```jldoctest
julia> a = Sequence(Taylor(1)^2, [1, 2, 3, 4])
Sequence in Taylor(1)² with coefficients Vector{Int64}:
 1
 2
 3
 4

julia> block(a)
2-element Vector{Sequence{Taylor, SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true}}}:
 Sequence(Taylor(1), [1, 2])
 Sequence(Taylor(1), [3, 4])
```
"""
block(a::Sequence{<:CartesianSpace}) = collect(eachblock(a))

"""
    block(a::Sequence{<:CartesianSpace}, i)

Return the ``i``-th [`Sequence`](@ref) composing the block sequence.

# Examples

```jldoctest
julia> a = Sequence(Taylor(1)^2, [1, 2, 3, 4])
Sequence in Taylor(1)² with coefficients Vector{Int64}:
 1
 2
 3
 4

julia> block(a, 1)
Sequence in Taylor(1) with coefficients SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true}:
 1
 2

julia> block(a, 2)
Sequence in Taylor(1) with coefficients SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true}:
 3
 4
```
"""
Base.@propagate_inbounds block(a::Sequence{<:CartesianSpace}, i) =
    Sequence(space(a)[i], view(coefficients(a), _component_findposition(i, space(a))))

# promotion

# Base.convert(::Type{Sequence{ScalarSpace,S}}, a::Number) where {S} = Sequence(ScalarSpace(), eltype(S)[a])


# Base.convert(::Type{Sequence{S,R}}, a::Number) where {S<:SequenceSpace,R} = Sequence(_zero_space(S), eltype(R)[a])

# Base.promote_rule(::Type{Sequence{S,R}}, ::Type{T}) where {S<:SequenceSpace,R,T<:Number} =
#     Sequence{S, Vector{promote_type(eltype(R), T)}}

# Base.convert(::Type{Sequence{S₁,R₁}}, a::Sequence{S₂,R₂}) where {S₁<:SequenceSpace,S₂<:SequenceSpace,R₁,R₂} =
#     Sequence{S₁,R₁}(convert(S₁, space(a)), convert(R₁, coefficients(a)))

# Base.promote_rule(::Type{Sequence{S₁,R₁}}, ::Type{Sequence{S₂,R₂}}) where {S₁<:SequenceSpace,S₂<:SequenceSpace,R₁,R₂} =
#     Sequence{promote_type(S₁, S₂), promote_type(R₁, R₂)}


# Base.convert(::Type{Sequence{T₁,S₁}}, a::Sequence{T₂,S₂}) where {T₁,S₁,T₂,S₂} =
#     Sequence{T₁,S₁}(convert(T₁, space(a)), convert(S₁, coefficients(a)))

# Base.promote_rule(::Type{Sequence{T₁,S₁}}, ::Type{Sequence{T₂,S₂}}) where {T₁,S₁,T₂,S₂} =
#     Sequence{promote_type(T₁, T₂), promote_type(S₁, S₂)}

# show

function Base.show(io::IO, ::MIME"text/plain", a::Sequence)
    println(io, "Sequence in ", _prettystring(space(a), true), " with coefficients ", typeof(coefficients(a)), ":")
    return Base.print_array(io, coefficients(a))
end

function Base.show(io::IO, a::Sequence)
    get(io, :compact, false) && return show(io, coefficients(a))
    return print(io, "Sequence(", space(a), ", ", coefficients(a), ")")
end










# conjugacy symmetry

# conjugacy_symmetry!(a::Sequence{<:VectorSpace,<:AbstractVector{<:Real}}) = a

conjugacy_symmetry!(a::Sequence) = _conjugacy_symmetry!(a)

_conjugacy_symmetry!(::Sequence) = throw(DomainError) # TODO: lift restriction

function _conjugacy_symmetry!(a::Sequence{ScalarSpace})
    @inbounds a[1] = real(a[1])
    return a
end

function _conjugacy_symmetry!(a::Sequence{<:Fourier})
    A = coefficients(a)
    A .= (A .+ conj.(reverse(A))) ./ 2
    return a
end

function _conjugacy_symmetry!(a::Sequence{<:TensorSpace{<:Tuple{Vararg{Fourier}}}})
    A = _no_alloc_reshape(coefficients(a), dimensions(space(a)))
    A .= (A .+ conj.(reverse(A))) ./ 2
    return a
end

function _conjugacy_symmetry!(a::Sequence{<:CartesianSpace})
    for aᵢ ∈ eachblock(a)
        _conjugacy_symmetry!(aᵢ)
    end
    return a
end

function _conjugacy_symmetry!(a::Sequence{CartesianProduct{T}}) where {N,T<:NTuple{N,VectorSpace}}
    @inbounds _conjugacy_symmetry!(block(a, 1))
    @inbounds _conjugacy_symmetry!(block(a, 2:N))
    return a
end
function _conjugacy_symmetry!(a::Sequence{CartesianProduct{T}}) where {T<:Tuple{VectorSpace}}
    @inbounds _conjugacy_symmetry!(block(a, 1))
    return a
end





# decay rate

_linear_regression_log_abs(x) = log(abs(x))
_linear_regression_log_abs(x::Interval) = log(mag(x))
_linear_regression_log_abs(x::Complex{<:Interval}) = log(mag(x))

function __linear_regression(f, A, j)
    sum_x = t = n = 0
    sum_log_abs_A = zero(_linear_regression_log_abs(one(eltype(A))))
    mean = sum_log_abs_A/1
    u = 0*sum_log_abs_A
    for (i, _Aᵢ_) ∈ enumerate(A)
        Aᵢ = mid(_Aᵢ_)
        if !iszero(Aᵢ)
            log_abs_Aᵢ = _linear_regression_log_abs(Aᵢ)
            if i ≤ j
                f_i = f(i)
                sum_x += f_i
                u += f_i*log_abs_Aᵢ
                t += f_i*f_i
                sum_log_abs_A += log_abs_Aᵢ
                n += 1
            else
                mean += log_abs_Aᵢ
            end
        end
    end
    mean /= (length(A) - j)
    x̄ = sum_x/n
    val = t - sum_x*x̄
    β = (u - x̄*sum_log_abs_A)/ifelse(iszero(val), one(val), val)
    α = sum_log_abs_A/n - β * x̄
    err = zero(sum_log_abs_A)
    for (i, _Aᵢ_) ∈ enumerate(A)
        Aᵢ = mid(_Aᵢ_)
        if !iszero(Aᵢ)
            log_abs_Aᵢ = _linear_regression_log_abs(Aᵢ)
            if i ≤ j
                err += (log_abs_Aᵢ - α - β * f(i))^2
            else
                err += (log_abs_Aᵢ - mean)^2
            end
        end
    end
    return β, err
end

#

function _linear_regression(s::TensorSpace{<:NTuple{N,BaseSpace}}, f, A) where {N}
    log_abs_A = _linear_regression_log_abs.(filter(!iszero, A))
    len = length(log_abs_A)
    len == 0 && return ntuple(_ -> f(one(eltype(log_abs_A))), Val(N)), zero(eltype(log_abs_A)), missing
    x = ones(Float64, len, N+1)
    n = 0
    @inbounds for (i, α) ∈ enumerate(indices(s))
        if !iszero(mid(A[i]))
            view(x, i-n, 2:N+1) .= f.(abs.(α) .+ 1)
        else
            n += 1
        end
    end
    r = LinearAlgebra.svd(x) \ log_abs_A
    err = norm(mul!(log_abs_A, x, r, 1, -1), 2)
    return ntuple(i -> r[i+1], Val(N)), err, missing
end

# Taylor

function _linear_regression(::Taylor, f, A)
    j = length(A)
    β, err = β_, err_ = __linear_regression(f, A, j)
    while err_ ≤ err && j ≥ 3
        j -= 1
        β, err = β_, err_
        β_, err_ = __linear_regression(f, A, j)
    end
    return β, err, j
end

# Fourier

function _linear_regression(s::Fourier, f, A)
    j = order(s)+1
    @inbounds A1 = view(A, j:2*j-1)
    β1, err1 = __linear_regression(f, A1, j)
    @inbounds A2 = view(A, j:-1:1)
    β2, err2 = __linear_regression(f, A2, j)
    β = β_ = (β1 + β2)/2
    err = err_ = max(err1, err2)
    while err_ ≤ err && j ≥ 3
        j -= 1
        β, err = β_, err_
        β1_, err1_ = __linear_regression(f, A1, j)
        β2_, err2_ = __linear_regression(f, A2, j)
        β_ = (β1_ + β2_)/2
        err_ = max(err1_, err2_)
    end
    return β, err, j
end

# Chebyshev

function _linear_regression(::Chebyshev, f, A)
    j = length(A)
    β, err = β_, err_ = __linear_regression(f, A, j)
    while err_ ≤ err && j ≥ 3
        j -= 1
        β, err = β_, err_
        β_, err_ = __linear_regression(f, A, j)
    end
    return β, err, j
end

"""
    geometricweight(a::Sequence{<:SequenceSpace})

Compute an approximation of the geometric decay rate of `a` by performing the
ordinary least squares method on the logarithm of the absolute value of the
coefficients of `a`.

See also: [`GeometricWeight`](@ref), [`IdentityWeight`](@ref),
[`AlgebraicWeight`](@ref), [`algebraicweight`](@ref) and [`BesselWeight`](@ref).

# Examples

```jldoctest
julia> rate(geometricweight(Sequence(Taylor(10), [inv(2.0^i) for i in 0:10]))) ≈ 2
true

julia> rate.(geometricweight(Sequence(Taylor(10) ⊗ Fourier(3, 1.0), vec([inv(2.0^i * 3.0^abs(j)) for i in 0:10, j in -3:3])))) .≈ (2, 3)
(true, true)
```
"""
function geometricweight(a::Sequence{<:SequenceSpace})
    s = space(a)
    A = coefficients(a)
    rate, _, _ = _geometric_rate(s, A)
    return GeometricWeight.(rate)
end

function _geometric_rate(s::BaseSpace, A)
    β, err, j = _linear_regression(s, identity, A)
    rate = exp(ifelse(isfinite(β) & (β < 0), -β, zero(β)))
    return rate, err, j
end

function _geometric_rate(s::TensorSpace{<:NTuple{N,BaseSpace}}, A) where {N}
    β, err, j = _linear_regression(s, identity, A)
    trv_inds = [i for i ∈ 1:N if iszero(order(s[i]))]
    rate = ntuple(Val(N)) do i
        @inbounds βᵢ = β[i]
        v = ifelse(isfinite(βᵢ) & (βᵢ < 0) & (i ∉ trv_inds), -βᵢ, zero(βᵢ))
        return exp(v)
    end
    return rate, err, j
end

"""
    algebraicweight(a::Sequence{<:SequenceSpace})

Compute an approximation of the algebraic decay rate of `a` by performing the
ordinary least squares method on the logarithm of the absolute value of the
coefficients of `a`.

See also: [`AlgebraicWeight`](@ref), [`IdentityWeight`](@ref),
[`GeometricWeight`](@ref), [`geometricweight`](@ref) and [`BesselWeight`](@ref).

# Examples

```jldoctest
julia> rate(algebraicweight(Sequence(Taylor(10), [inv((1.0 + i)^2) for i in 0:10]))) ≈ 2
true

julia> rate.(algebraicweight(Sequence(Taylor(10) ⊗ Fourier(3, 1.0), vec([inv((1.0 + i)^2 * (1.0 + abs(j))^3) for i in 0:10, j in -3:3])))) .≈ (2, 3)
(true, true)
```
"""
function algebraicweight(a::Sequence{<:SequenceSpace})
    s = space(a)
    A = coefficients(a)
    rate, _, _ = _algebraic_rate(s, A)
    return AlgebraicWeight.(rate)
end

function _algebraic_rate(s::BaseSpace, A)
    β, err, j = _linear_regression(s, log, A)
    rate = ifelse(isfinite(β) & (β < 0), -β, zero(β))
    return rate, err, j
end

function _algebraic_rate(s::TensorSpace{<:NTuple{N,BaseSpace}}, A) where {N}
    β, err, j = _linear_regression(s, log, A)
    trv_inds = [i for i ∈ 1:N if iszero(order(s[i]))]
    rate = ntuple(Val(N)) do i
        @inbounds βᵢ = β[i]
        v = ifelse(isfinite(βᵢ) & (βᵢ < 0) & (i ∉ trv_inds), -βᵢ, zero(βᵢ))
        return v
    end
    return rate, err, j
end

# retrieve the optimal weight

weight(a::Sequence{<:SequenceSpace}) = _weight(a::Sequence{<:SequenceSpace})[1]

function _weight(a::Sequence{<:SequenceSpace})
    s = space(a)
    A = coefficients(a)
    geo_rate, geo_err, j = _geometric_rate(s, A)
    alg_rate, alg_err, j = _algebraic_rate(s, A)
    geo_err ≤ alg_err && return GeometricWeight.(geo_rate), j
    return AlgebraicWeight.(alg_rate), j
end

#

polish!(a::Sequence{ScalarSpace}) = a

polish!(a::Sequence{<:TensorSpace}) = a

function polish!(a::Sequence{<:BaseSpace})
    w, ord = _weight(a)
    s = space(a)
    norm_a = norm(a, 1)
    for i ∈ indices(s)
        if abs(i) > ord
            val = norm_a / _getindex(w, s, i)
            if abs(a[i]) > val
                a[i] = 0
            end
        end
    end
    return a
end

function polish!(a::Sequence{<:CartesianSpace})
    for i ∈ 1:nspaces(space(a))
        polish!(block(a, i))
    end
    return a
end
