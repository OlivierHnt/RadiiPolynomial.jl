if isdefined(LinearAlgebra, :PivotingStrategy)
    _qr_nopivot!(A) = qr!(A, NoPivot())
else # Julia v1.6
    _qr_nopivot!(A) = qr!(A, Val(false))
end

#

_linear_regression_log_abs(x) = log(abs(x))
_linear_regression_log_abs(x::Interval) = log(mag(x))
_linear_regression_log_abs(x::Complex{<:Interval}) = log(mag(x))

"""
    Weight

Abstract type for all weights.
"""
abstract type Weight end

_getindex(weight::NTuple{N,Weight}, s::TensorSpace{<:NTuple{N,BaseSpace}}, α::NTuple{N,Int}) where {N} =
    @inbounds _getindex(weight[1], s[1], α[1]) * _getindex(Base.tail(weight), Base.tail(s), Base.tail(α))
_getindex(weight::Tuple{Weight}, s::TensorSpace{<:Tuple{BaseSpace}}, α::Tuple{Int}) =
    @inbounds _getindex(weight[1], s[1], α[1])

"""
    IdentityWeight <: Weight

Identity weight.
"""
struct IdentityWeight <: Weight end

"""
    GeometricWeight{<:Real} <: Weight

Geometric weight associated to some `rate::Real` satisfying `isfinite(rate)` and `rate > 0`.

# Examples
```jldoctest
julia> w = GeometricWeight(1.0)
GeometricWeight{Float64}(1.0)

julia> rate(w)
1.0
```
"""
struct GeometricWeight{T<:Real} <: Weight
    rate :: T
    function GeometricWeight{T}(rate::T) where {T<:Real}
        isfinite(rate) & (rate > 0) || return throw(DomainError(rate, "rate must be finite and strictly positive"))
        return new{T}(rate)
    end
end

GeometricWeight(rate::T) where {T<:Real} = GeometricWeight{T}(rate)

rate(weight::GeometricWeight) = weight.rate

function geometricweight(a::Sequence{<:BaseSpace})
    rate = _geometric_rate(space(a), coefficients(a))
    return GeometricWeight(rate)
end
function geometricweight(a::Sequence{<:TensorSpace})
    rate = _geometric_rate(space(a), coefficients(a))
    return GeometricWeight.(rate)
end

function _geometric_rate(s::TensorSpace{<:NTuple{N,BaseSpace}}, A) where {N}
    A_ = _linear_regression_log_abs.(filter(!iszero, A))
    x = ones(Float64, length(A_), N+1)
    n = 0
    @inbounds for (i, α) ∈ enumerate(indices(s))
        if !iszero(A[i])
            view(x, i-n, 2:N+1) .= abs.(α) .+ 1
        else
            n += 1
        end
    end
    r = ldiv!(_qr_nopivot!(x), A_)
    return ntuple(Val(N)) do i
        @inbounds rᵢ₊₁ = r[i+1]
        v = ifelse(isfinite(rᵢ₊₁), rᵢ₊₁, zero(rᵢ₊₁))
        return exp(-v)
    end
end

# Taylor

_getindex(weight::GeometricWeight, ::Taylor, i::Int) = weight.rate ^ i
_getindex(weight::GeometricWeight{<:Interval}, ::Taylor, i::Int) = pow(weight.rate, i)

function _geometric_rate(::Taylor, A)
    sum_x = t = n = 0
    sum_log_abs_A = zero(_linear_regression_log_abs(one(eltype(A))))
    u = 0*sum_log_abs_A
    for (i, Aᵢ) ∈ enumerate(A)
        if !iszero(Aᵢ)
            log_abs_Aᵢ = _linear_regression_log_abs(Aᵢ)
            sum_x += i
            u += i*log_abs_Aᵢ
            t += i*i
            sum_log_abs_A += log_abs_Aᵢ
            n += 1
        end
    end
    x̄ = sum_x/n
    r = (u - x̄*sum_log_abs_A)/(t - sum_x*x̄)
    v = ifelse(isfinite(r), r, zero(r))
    return exp(-v)
end

# Fourier

_getindex(weight::GeometricWeight, ::Fourier, i::Int) = weight.rate ^ abs(i)
_getindex(weight::GeometricWeight{<:Interval}, ::Fourier, i::Int) = pow(weight.rate, abs(i))

function _geometric_rate(s::Fourier, A)
    ord = order(s)
    sum_x = t = n = 0
    sum_log_abs_A = zero(_linear_regression_log_abs(one(eltype(A))))
    u = 0*sum_log_abs_A
    for (i, Aᵢ) ∈ enumerate(A)
        if !iszero(Aᵢ)
            abs_i = abs(i-ord-1)+1
            log_abs_Aᵢ = _linear_regression_log_abs(Aᵢ)
            sum_x += abs_i
            u += abs_i*log_abs_Aᵢ
            t += abs_i*abs_i
            sum_log_abs_A += log_abs_Aᵢ
            n += 1
        end
    end
    x̄ = sum_x/n
    r = (u - x̄*sum_log_abs_A)/(t - sum_x*x̄)
    v = ifelse(isfinite(r), r, zero(r))
    return exp(-v)
end

# Chebyshev

_getindex(weight::GeometricWeight, ::Chebyshev, i::Int) = weight.rate ^ i
_getindex(weight::GeometricWeight{<:Interval}, ::Chebyshev, i::Int) = pow(weight.rate, i)

function _geometric_rate(::Chebyshev, A)
    sum_x = t = n = 0
    sum_log_abs_A = zero(_linear_regression_log_abs(one(eltype(A))))
    u = 0*sum_log_abs_A
    for (i, Aᵢ) ∈ enumerate(A)
        if !iszero(Aᵢ)
            log_abs_Aᵢ = _linear_regression_log_abs(Aᵢ)
            sum_x += i
            u += i*log_abs_Aᵢ
            t += i*i
            sum_log_abs_A += log_abs_Aᵢ
            n += 1
        end
    end
    x̄ = sum_x/n
    r = (u - x̄*sum_log_abs_A)/(t - sum_x*x̄)
    v = ifelse(isfinite(r), r, zero(r))
    return exp(-v)
end

"""
    AlgebraicWeight{<:Real} <: Weight

Algebraic weight associated to some `rate::Real` satisfying `isfinite(rate)` and `rate ≥ 0`.

# Examples
```jldoctest
julia> w = AlgebraicWeight(1.0)
AlgebraicWeight{Float64}(1.0)

julia> rate(w)
1.0
```
"""
struct AlgebraicWeight{T<:Real} <: Weight
    rate :: T
    function AlgebraicWeight{T}(rate::T) where {T<:Real}
        isfinite(rate) & (rate ≥ 0) || return throw(DomainError(rate, "rate must be finite and positive"))
        return new{T}(rate)
    end
end

AlgebraicWeight(rate::T) where {T<:Real} = AlgebraicWeight{T}(rate)

rate(weight::AlgebraicWeight) = weight.rate

function algebraicweight(a::Sequence{<:BaseSpace})
    rate = _algebraic_rate(space(a), coefficients(a))
    return AlgebraicWeight(rate)
end
function algebraicweight(a::Sequence{<:TensorSpace})
    rate = _algebraic_rate(space(a), coefficients(a))
    return AlgebraicWeight.(rate)
end

function _algebraic_rate(s::TensorSpace{<:NTuple{N,BaseSpace}}, A) where {N}
    A_ = _linear_regression_log_abs.(filter(!iszero, A))
    x = ones(Float64, length(A_), N+1)
    n = 0
    @inbounds for (i, α) ∈ enumerate(indices(s))
        if !iszero(A[i])
            view(x, i-n, 2:N+1) .= log.(abs.(α) .+ 1)
        else
            n += 1
        end
    end
    r = ldiv!(_qr_nopivot!(x), A_)
    return ntuple(Val(N)) do i
        @inbounds rᵢ₊₁ = r[i+1]
        return ifelse(isfinite(rᵢ₊₁) & (rᵢ₊₁ < 0), -rᵢ₊₁, zero(rᵢ₊₁))
    end
end

# Taylor

_getindex(weight::AlgebraicWeight, ::Taylor, i::Int) = (one(weight.rate) + i) ^ weight.rate
_getindex(weight::AlgebraicWeight{<:Interval}, ::Taylor, i::Int) = pow(one(weight.rate) + i, weight.rate)

function _algebraic_rate(::Taylor, A)
    sum_x = t = 0.0
    n = 0
    sum_log_abs_A = zero(_linear_regression_log_abs(one(eltype(A))))
    u = 0.0*sum_log_abs_A
    for (i, Aᵢ) ∈ enumerate(A)
        if !iszero(Aᵢ)
            log_i = log(i)
            log_abs_Aᵢ = _linear_regression_log_abs(Aᵢ)
            sum_x += log_i
            u += log_i*log_abs_Aᵢ
            t += log_i*log_i
            sum_log_abs_A += log_abs_Aᵢ
            n += 1
        end
    end
    x̄ = sum_x/n
    r = (u - x̄*sum_log_abs_A)/(t - sum_x*x̄)
    return ifelse(isfinite(r) & (r < 0), -r, zero(r))
end

# Fourier

_getindex(weight::AlgebraicWeight, ::Fourier, i::Int) = (one(weight.rate) + abs(i)) ^ weight.rate
_getindex(weight::AlgebraicWeight{<:Interval}, ::Fourier, i::Int) = pow(one(weight.rate) + abs(i), weight.rate)

function _algebraic_rate(s::Fourier, A)
    ord = order(s)
    sum_x = t = 0.0
    n = 0
    sum_log_abs_A = zero(_linear_regression_log_abs(one(eltype(A))))
    u = 0.0*sum_log_abs_A
    for (i, Aᵢ) ∈ enumerate(A)
        if !iszero(Aᵢ)
            log_abs_i = log(abs(i-ord-1)+1)
            log_abs_Aᵢ = _linear_regression_log_abs(Aᵢ)
            sum_x += log_abs_i
            u += log_abs_i*log_abs_Aᵢ
            t += log_abs_i*log_abs_i
            sum_log_abs_A += log_abs_Aᵢ
            n += 1
        end
    end
    x̄ = sum_x/n
    r = (u - x̄*sum_log_abs_A)/(t - sum_x*x̄)
    return ifelse(isfinite(r) & (r < 0), -r, zero(r))
end

# Chebyshev

_getindex(weight::AlgebraicWeight, ::Chebyshev, i::Int) = (one(weight.rate) + i) ^ weight.rate
_getindex(weight::AlgebraicWeight{<:Interval}, ::Chebyshev, i::Int) = pow(one(weight.rate) + i, weight.rate)

function _algebraic_rate(::Chebyshev, A)
    sum_x = t = 0.0
    n = 0
    sum_log_abs_A = zero(_linear_regression_log_abs(one(eltype(A))))
    u = 0.0*sum_log_abs_A
    for (i, Aᵢ) ∈ enumerate(A)
        if !iszero(Aᵢ)
            log_i = log(i)
            log_abs_Aᵢ = _linear_regression_log_abs(Aᵢ)
            sum_x += log_i
            u += log_i*log_abs_Aᵢ
            t += log_i*log_i
            sum_log_abs_A += log_abs_Aᵢ
            n += 1
        end
    end
    x̄ = sum_x/n
    r = (u - x̄*sum_log_abs_A)/(t - sum_x*x̄)
    return ifelse(isfinite(r) & (r < 0), -r, zero(r))
end

"""
    BesselWeight{<:Real} <: Weight

Bessel weight associated to some `rate::Real` satisfying `isfinite(rate)` and `rate ≥ 0`.

# Examples
```jldoctest
julia> w = BesselWeight(1.0)
BesselWeight{Float64}(1.0)

julia> rate(w)
1.0
```
"""
struct BesselWeight{T<:Real} <: Weight
    rate :: T
    function BesselWeight{T}(rate::T) where {T<:Real}
        isfinite(rate) & (rate ≥ 0) || return throw(DomainError(rate, "rate must be finite and positive"))
        return new{T}(rate)
    end
end

BesselWeight(rate::T) where {T<:Real} = BesselWeight{T}(rate)

rate(weight::BesselWeight) = weight.rate

_getindex(weight::BesselWeight, ::TensorSpace{<:NTuple{N,Fourier}}, α::NTuple{N,Int}) where {N} =
    (one(weight.rate) + mapreduce(abs2, +, α)) ^ weight.rate
_getindex(weight::BesselWeight{<:Interval}, ::TensorSpace{<:NTuple{N,Fourier}}, α::NTuple{N,Int}) where {N} =
    pow(one(weight.rate) + mapreduce(abs2, +, α), weight.rate)

_getindex(weight::BesselWeight, ::Fourier, i::Int) = (one(weight.rate) + i*i) ^ weight.rate
_getindex(weight::BesselWeight{<:Interval}, ::Fourier, i::Int) = pow(one(weight.rate) + i*i, weight.rate)

#

"""
    BanachSpace

Abstract type for all Banach spaces.
"""
abstract type BanachSpace end



struct Ell1{T<:Union{Weight,Tuple{Vararg{Weight}}}} <: BanachSpace
    weight :: T
    Ell1{T}(weight::T) where {T<:Union{Weight,Tuple{Vararg{Weight}}}} = new{T}(weight)
    Ell1{Tuple{}}(::Tuple{}) = throw(ArgumentError("Ell1 is only defined for at least one Weight"))
end

Ell1(weight::T) where {T<:Weight} = Ell1{T}(weight)
Ell1(weight::T) where {T<:Tuple{Vararg{Weight}}} = Ell1{T}(weight)
Ell1() = Ell1{IdentityWeight}(IdentityWeight())
Ell1(weight::Weight...) = Ell1(weight)

const ℓ¹ = Ell1

"""
    Ell1{<:Union{Weight,Tuple{Vararg{Weight}}}} <: BanachSpace

Weighted ``\\ell^1`` space.

Unicode alias `ℓ¹` can be typed by `\\ell<tab>\\^1<tab>` in the Julia REPL, and in many editors.

See also: [`Ell2`](@ref) and [`EllInf`](@ref).

# Examples
```jldoctest
julia> Ell1()
Ell1{IdentityWeight}(IdentityWeight())

julia> Ell1(GeometricWeight(1.0))
Ell1{GeometricWeight{Float64}}(GeometricWeight{Float64}(1.0))

julia> Ell1(GeometricWeight(1.0), AlgebraicWeight(2.0))
Ell1{Tuple{GeometricWeight{Float64}, AlgebraicWeight{Float64}}}((GeometricWeight{Float64}(1.0), AlgebraicWeight{Float64}(2.0)))
```
"""
Ell1, ℓ¹



struct Ell2{T<:Union{Weight,Tuple{Vararg{Weight}}}} <: BanachSpace
    weight :: T
    Ell2{T}(weight::T) where {T<:Union{Weight,Tuple{Vararg{Weight}}}} = new{T}(weight)
    Ell2{Tuple{}}(::Tuple{}) = throw(ArgumentError("Ell2 is only defined for at least one Weight"))
end

Ell2(weight::T) where {T<:Weight} = Ell2{T}(weight)
Ell2(weight::T) where {T<:Tuple{Vararg{Weight}}} = Ell2{T}(weight)
Ell2() = Ell2{IdentityWeight}(IdentityWeight())
Ell2(weight::Weight...) = Ell2(weight)

const ℓ² = Ell2

"""
    Ell2{<:Union{Weight,Tuple{Vararg{Weight}}}} <: BanachSpace

Weighted ``\\ell^2`` space.

Unicode alias `ℓ²` can be typed by `\\ell<tab>\\^2<tab>` in the Julia REPL, and in many editors.

See also: [`Ell1`](@ref) and [`EllInf`](@ref).

# Examples
```jldoctest
julia> Ell2()
Ell2{IdentityWeight}(IdentityWeight())

julia> Ell2(BesselWeight(1.0))
Ell2{BesselWeight{Float64}}(BesselWeight{Float64}(1.0))

julia> Ell2(BesselWeight(1.0), GeometricWeight(2.0))
Ell2{Tuple{BesselWeight{Float64}, GeometricWeight{Float64}}}((BesselWeight{Float64}(1.0), GeometricWeight{Float64}(2.0)))
```
"""
Ell2, ℓ²



struct EllInf{T<:Union{Weight,Tuple{Vararg{Weight}}}} <: BanachSpace
    weight :: T
    EllInf{T}(weight::T) where {T<:Union{Weight,Tuple{Vararg{Weight}}}} = new{T}(weight)
    EllInf{Tuple{}}(::Tuple{}) = throw(ArgumentError("EllInf is only defined for at least one Weight"))
end

EllInf(weight::T) where {T<:Weight} = EllInf{T}(weight)
EllInf(weight::T) where {T<:Tuple{Vararg{Weight}}} = EllInf{T}(weight)
EllInf() = EllInf{IdentityWeight}(IdentityWeight())
EllInf(weight::Weight...) = EllInf(weight)

const ℓ∞ = EllInf

"""
    EllInf{<:Union{Weight,Tuple{Vararg{Weight}}}} <: BanachSpace

Weighted ``\\ell^\\infty`` space.

Unicode alias `ℓ∞` can be typed by `\\ell<tab>\\infty<tab>` in the Julia REPL, and in many editors.

See also: [`Ell1`](@ref) and [`Ell2`](@ref).

# Examples
```jldoctest
julia> EllInf()
EllInf{IdentityWeight}(IdentityWeight())

julia> EllInf(GeometricWeight(1.0))
EllInf{GeometricWeight{Float64}}(GeometricWeight{Float64}(1.0))

julia> EllInf(GeometricWeight(1.0), AlgebraicWeight(2.0))
EllInf{Tuple{GeometricWeight{Float64}, AlgebraicWeight{Float64}}}((GeometricWeight{Float64}(1.0), AlgebraicWeight{Float64}(2.0)))
```
"""
EllInf, ℓ∞



# normed cartesian space

"""
    NormedCartesianSpace{<:Union{BanachSpace,Tuple{Vararg{BanachSpace}}},<:BanachSpace} <: BanachSpace

Cartesian Banach space.

See also: [`Ell1`](@ref), [`Ell2`](@ref) and [`EllInf`](@ref).

# Examples
```jldoctest
julia> NormedCartesianSpace(Ell1(), EllInf())
NormedCartesianSpace{Ell1{IdentityWeight}, EllInf{IdentityWeight}}(Ell1{IdentityWeight}(IdentityWeight()), EllInf{IdentityWeight}(IdentityWeight()))

julia> NormedCartesianSpace((Ell1(), Ell2()), EllInf())
NormedCartesianSpace{Tuple{Ell1{IdentityWeight}, Ell2{IdentityWeight}}, EllInf{IdentityWeight}}((Ell1{IdentityWeight}(IdentityWeight()), Ell2{IdentityWeight}(IdentityWeight())), EllInf{IdentityWeight}(IdentityWeight()))
```
"""
struct NormedCartesianSpace{T<:Union{BanachSpace,Tuple{Vararg{BanachSpace}}},S<:BanachSpace} <: BanachSpace
    inner :: T
    outer :: S
end

# fallback methods

LinearAlgebra.opnorm(A::LinearOperator, X::BanachSpace) = opnorm(A, X, X)

function LinearAlgebra.opnorm(A::LinearOperator, X_domain::BanachSpace, X_codomain::BanachSpace)
    codomain_A = codomain(A)
    A_ = coefficients(A)
    @inbounds v₁ = norm(Sequence(codomain_A, view(A_, :, 1)), X_codomain)
    sz = size(A_, 2)
    v = Vector{typeof(v₁)}(undef, sz)
    @inbounds v[1] = v₁
    @inbounds for i ∈ 2:sz
        v[i] = norm(Sequence(codomain_A, view(A_, :, i)), X_codomain)
    end
    return opnorm(LinearOperator(domain(A), ParameterSpace(), transpose(v)), X_domain)
end

function LinearAlgebra.norm(a::Sequence, p::Real=Inf)
    if p == 1
        return norm(a, Ell1(IdentityWeight()))
    elseif p == 2
        return norm(a, Ell2(IdentityWeight()))
    elseif p == Inf
        return norm(a, EllInf(IdentityWeight()))
    else
        return throw(ArgumentError("p-norm is only supported for p = 1, 2, Inf"))
    end
end

function LinearAlgebra.opnorm(A::LinearOperator, p::Real=Inf)
    if p == 1
        return opnorm(A, Ell1(IdentityWeight()))
    elseif p == 2
        return opnorm(A, Ell2(IdentityWeight()))
    elseif p == Inf
        return opnorm(A, EllInf(IdentityWeight()))
    else
        return throw(ArgumentError("p-norm is only supported for p = 1, 2, Inf"))
    end
end

#

for T ∈ (:Ell1, :Ell2, :EllInf)
    @eval begin
        LinearAlgebra.norm(a::Sequence, X::$T{<:Weight}) = _apply(X, space(a), coefficients(a))
        function LinearAlgebra.norm(a::Sequence{TensorSpace{S}}, X::$T{<:NTuple{N,Weight}}) where {N,S<:NTuple{N,BaseSpace}}
            space_a = space(a)
            A = _no_alloc_reshape(coefficients(a), dimensions(space_a))
            return _apply(X, space_a, A)
        end
        _apply(X::$T, space::TensorSpace, A) =
            @inbounds _apply($T(X.weight[1]), space[1], _apply($T(Base.tail(X.weight)), Base.tail(space), A))
        _apply(X::$T, space::TensorSpace{<:Tuple{BaseSpace}}, A) =
            @inbounds _apply($T(X.weight[1]), space[1], A)

        LinearAlgebra.opnorm(A::LinearOperator{<:VectorSpace,ParameterSpace}, X::$T{<:Weight}) =
            _apply_dual(X, domain(A), vec(coefficients(A)))
        function LinearAlgebra.opnorm(A::LinearOperator{TensorSpace{S},ParameterSpace}, X::$T{<:NTuple{N,Weight}}) where {N,S<:NTuple{N,BaseSpace}}
            domain_A = domain(A)
            A_ = _no_alloc_reshape(coefficients(A), dimensions(domain_A))
            return _apply_dual(X, domain_A, A_)
        end
        _apply_dual(X::$T, space::TensorSpace, A) =
            @inbounds _apply_dual($T(X.weight[1]), space[1], _apply_dual($T(Base.tail(X.weight)), Base.tail(space), A))
        _apply_dual(X::$T, space::TensorSpace{<:Tuple{BaseSpace}}, A) =
            @inbounds _apply_dual($T(X.weight[1]), space[1], A)
    end
end

#

function LinearAlgebra.opnorm(A::LinearOperator, ::Ell1{IdentityWeight}, ::Ell1{IdentityWeight})
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

function LinearAlgebra.opnorm(A::LinearOperator, ::Ell2{IdentityWeight}, ::Ell2{IdentityWeight})
    X_1 = Ell1(IdentityWeight())
    X_Inf = EllInf(IdentityWeight())
    return sqrt(opnorm(A, X_1, X_1) * opnorm(A, X_Inf, X_Inf))
end

function LinearAlgebra.opnorm(A::LinearOperator, ::EllInf{IdentityWeight}, ::EllInf{IdentityWeight})
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

_apply(::Ell1{IdentityWeight}, ::TensorSpace, A::AbstractVector) = sum(abs, A)
_apply_dual(::Ell1{IdentityWeight}, space::TensorSpace, A::AbstractVector) = _apply(EllInf(IdentityWeight()), space, A)

_apply(::Ell2{IdentityWeight}, ::TensorSpace, A::AbstractVector) = sqrt(sum(abs2, A))
_apply_dual(::Ell2{IdentityWeight}, ::TensorSpace, A::AbstractVector) = sqrt(sum(abs2, A))

_apply(::EllInf{IdentityWeight}, ::TensorSpace, A::AbstractVector) = maximum(abs, A)
_apply_dual(::EllInf{IdentityWeight}, space::TensorSpace, A::AbstractVector) = _apply(Ell1(IdentityWeight()), space, A)

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
    ν = rate(X.weight)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * one(ν)
    @inbounds for i ∈ ord-1:-1:0
        s = s * ν + abs(A[i+1])
    end
    return s
end
function _apply(X::Ell1{<:GeometricWeight}, space::Taylor, A::AbstractArray{T,N}) where {T,N}
    ν = rate(X.weight)
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
_apply_dual(X::Ell1{<:GeometricWeight}, space::Taylor, A::AbstractArray{T}) where {T} =
    _apply(EllInf(GeometricWeight(abs(one(T))/rate(X.weight))), space, A)

function _apply(X::Ell1{<:AlgebraicWeight}, space::Taylor, A::AbstractVector)
    @inbounds s = abs(A[1]) * _getindex(X.weight, space, 0)
    @inbounds for i ∈ 1:order(space)
        s += abs(A[i+1]) * _getindex(X.weight, space, i)
    end
    return s
end
function _apply(X::Ell1{<:AlgebraicWeight}, space::Taylor, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))*_getindex(X.weight, space, 0))
    @inbounds A₀ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:order(space)
        s .+= abs.(selectdim(A, N, i+1)) .* _getindex(X.weight, space, i)
    end
    return s
end
function _apply_dual(X::Ell1{<:AlgebraicWeight}, space::Taylor, A::AbstractVector)
    @inbounds s = abs(A[1]) / _getindex(X.weight, space, 0)
    @inbounds for i ∈ 1:order(space)
        s = max(s, abs(A[i+1]) / _getindex(X.weight, space, i))
    end
    return s
end
function _apply_dual(X::Ell1{<:AlgebraicWeight}, space::Taylor, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))/_getindex(X.weight, space, 0))
    @inbounds A₀ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:order(space)
        s .= max.(s, abs.(selectdim(A, N, i+1)) ./ _getindex(X.weight, space, i))
    end
    return s
end

_apply(::Ell2{IdentityWeight}, ::Taylor, A::AbstractVector) = sqrt(sum(abs2, A))
_apply_dual(::Ell2{IdentityWeight}, ::Taylor, A::AbstractVector) = sqrt(sum(abs2, A))

_apply(::EllInf{IdentityWeight}, ::Taylor, A::AbstractVector) = maximum(abs, A)
function _apply(::EllInf{IdentityWeight}, ::Taylor, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T)))
    @inbounds A₁ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₁))
    s .= abs.(A₁)
    for i ∈ 2:size(A, N)
        s .= max.(s, abs.(selectdim(A, N, i)))
    end
    return s
end
_apply_dual(::EllInf{IdentityWeight}, space::Taylor, A::AbstractArray) = _apply(Ell1(IdentityWeight()), space, A)

function _apply(X::EllInf{<:GeometricWeight}, space::Taylor, A::AbstractVector{T}) where {T}
    ν = rate(X.weight)
    νⁱ = one(ν)
    @inbounds s = abs(A[1]) * νⁱ
    @inbounds for i ∈ 1:order(space)
        νⁱ *= ν
        s = max(s, abs(A[i+1]) * νⁱ)
    end
    return s
end
function _apply(X::EllInf{<:GeometricWeight}, space::Taylor, A::AbstractArray{T,N}) where {T,N}
    ν = rate(X.weight)
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
_apply_dual(X::EllInf{<:GeometricWeight}, space::Taylor, A::AbstractArray{T}) where {T} =
    _apply(Ell1(GeometricWeight(abs(one(T))/rate(X.weight))), space, A)

function _apply(X::EllInf{<:AlgebraicWeight}, space::Taylor, A::AbstractVector)
    @inbounds s = abs(A[1]) * _getindex(X.weight, space, 0)
    @inbounds for i ∈ 1:order(space)
        s = max(s, abs(A[i+1]) * _getindex(X.weight, space, i))
    end
    return s
end
function _apply(X::EllInf{<:AlgebraicWeight}, space::Taylor, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))*_getindex(X.weight, space, 0))
    @inbounds A₀ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:order(space)
        s .= max.(s, abs.(selectdim(A, N, i+1)) .* _getindex(X.weight, space, i))
    end
    return s
end
function _apply_dual(X::EllInf{<:AlgebraicWeight}, space::Taylor, A::AbstractVector)
    @inbounds s = abs(A[1]) / _getindex(X.weight, space, 0)
    @inbounds for i ∈ 1:order(space)
        s += abs(A[i+1]) / _getindex(X.weight, space, i)
    end
    return s
end
function _apply_dual(X::EllInf{<:AlgebraicWeight}, space::Taylor, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))/_getindex(X.weight, space, 0))
    @inbounds A₀ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:order(space)
        s .+= abs.(selectdim(A, N, i+1)) ./ _getindex(X.weight, space, i)
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
    ν = rate(X.weight)
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
    ν = rate(X.weight)
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
_apply_dual(X::Ell1{<:GeometricWeight}, space::Fourier, A::AbstractArray{T}) where {T} =
    _apply(EllInf(GeometricWeight(abs(one(T))/rate(X.weight))), space, A)

function _apply(X::Ell1{<:AlgebraicWeight}, space::Fourier, A::AbstractVector)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * _getindex(X.weight, space, 0)
    @inbounds for i ∈ 1:ord
        s += (abs(A[ord+1-i]) + abs(A[ord+1+i])) * _getindex(X.weight, space, i)
    end
    return s
end
function _apply(X::Ell1{<:AlgebraicWeight}, space::Fourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))*_getindex(X.weight, space, 0))
    ord = order(space)
    @inbounds A₀ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    @inbounds s .= abs.(A₀)
    @inbounds for i ∈ 1:ord
        s .+= (abs.(selectdim(A, N, ord+1-i)) .+ abs.(selectdim(A, N, ord+1+i))) .* _getindex(X.weight, space, i)
    end
    return s
end
function _apply_dual(X::Ell1{<:AlgebraicWeight}, space::Fourier, A::AbstractVector{T}) where {T}
    ord = order(space)
    @inbounds s = abs(A[ord+1]) / _getindex(X.weight, space, 0)
    @inbounds for i ∈ 1:ord
        x = abs(one(T)) / _getindex(X.weight, space, i)
        s = max(s, abs(A[ord+1+i]) * x, abs(A[ord+1-i]) * x)
    end
    return s
end
function _apply_dual(X::Ell1{<:AlgebraicWeight}, space::Fourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))/_getindex(X.weight, space, 0))
    ord = order(space)
    @inbounds A₀ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:ord
        x = abs(one(T)) / _getindex(X.weight, space, i)
        s .= max.(s, abs.(selectdim(A, N, ord+1-i)) .* x, abs.(selectdim(A, N, ord+1+i)) .* x)
    end
    return s
end

_apply(::Ell2{IdentityWeight}, ::Fourier, A::AbstractVector) = sqrt(sum(abs2, A))
_apply_dual(::Ell2{IdentityWeight}, ::Fourier, A::AbstractVector) = sqrt(sum(abs2, A))

function _apply(X::Ell2{<:BesselWeight}, space::Fourier, A::AbstractVector)
    ord = order(space)
    @inbounds x = abs2(A[ord+1]) * _getindex(X.weight, space, 0)
    @inbounds for i ∈ 1:ord
        x += (abs2(A[ord+1-i]) + abs2(A[ord+1+i])) * _getindex(X.weight, space, i)
    end
    return sqrt(x)
end
function _apply(X::Ell2{<:BesselWeight}, space::TensorSpace{<:NTuple{N,Fourier}}, A::AbstractVector{T}) where {N,T}
    x = zero(abs2(zero(T))*_getindex(X.weight, space, ntuple(i -> 0, Val(N))))
    @inbounds for α ∈ indices(space)
        x += abs2(A[_findposition(α, space)]) * _getindex(X.weight, space, α)
    end
    return sqrt(x)
end
function _apply_dual(X::Ell2{<:BesselWeight}, space::Fourier, A::AbstractVector)
    ord = order(space)
    @inbounds x = abs2(A[ord+1]) / _getindex(X.weight, space, 0)
    @inbounds for i ∈ 1:ord
        x += (abs2(A[ord+1-i]) + abs2(A[ord+1+i])) / _getindex(X.weight, space, i)
    end
    return sqrt(x)
end
function _apply_dual(X::Ell2{<:BesselWeight}, space::TensorSpace{<:NTuple{N,Fourier}}, A::AbstractVector{T}) where {N,T}
    x = zero(abs2(zero(T))/_getindex(X.weight, space, ntuple(i -> 0, Val(N))))
    @inbounds for α ∈ indices(space)
        x += abs2(A[_findposition(α, space)]) / _getindex(X.weight, space, α)
    end
    return sqrt(x)
end

_apply(::EllInf{IdentityWeight}, ::Fourier, A::AbstractVector) = maximum(abs, A)
function _apply(::EllInf{IdentityWeight}, ::Fourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T)))
    @inbounds A₁ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₁))
    s .= abs.(A₁)
    for i ∈ 2:size(A, N)
        s .= max.(s, abs.(selectdim(A, N, i)))
    end
    return s
end
_apply_dual(::EllInf{IdentityWeight}, space::Fourier, A::AbstractArray) = _apply(Ell1(IdentityWeight()), space, A)

function _apply(X::EllInf{<:GeometricWeight}, space::Fourier, A::AbstractVector{T}) where {T}
    ν = rate(X.weight)
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
    ν = rate(X.weight)
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
_apply_dual(X::EllInf{<:GeometricWeight}, space::Fourier, A::AbstractArray{T}) where {T} =
    _apply(Ell1(GeometricWeight(abs(one(T))/rate(X.weight))), space, A)

function _apply(X::EllInf{<:AlgebraicWeight}, space::Fourier, A::AbstractVector{T}) where {T}
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * _getindex(X.weight, space, 0)
    @inbounds for i ∈ 1:ord
        x = _getindex(X.weight, space, i)
        s = max(s, abs(A[ord+1+i]) * x, abs(A[ord+1-i]) * x)
    end
    return s
end
function _apply(X::EllInf{<:AlgebraicWeight}, space::Fourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))*_getindex(X.weight, space, 0))
    ord = order(space)
    @inbounds A₀ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:ord
        x = _getindex(X.weight, space, i)
        s .= max.(s, abs.(selectdim(A, N, ord+1-i)) .* x, abs.(selectdim(A, N, ord+1+i)) .* x)
    end
    return s
end
function _apply_dual(X::EllInf{<:AlgebraicWeight}, space::Fourier, A::AbstractVector)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) / _getindex(X.weight, space, 0)
    @inbounds for i ∈ 1:ord
        s += (abs(A[ord+1-i]) + abs(A[ord+1+i])) / _getindex(X.weight, space, i)
    end
    return s
end
function _apply_dual(X::EllInf{<:AlgebraicWeight}, space::Fourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))/_getindex(X.weight, space, 0))
    ord = order(space)
    @inbounds A₀ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    @inbounds s .= abs.(A₀)
    @inbounds for i ∈ 1:ord
        s .+= (abs.(selectdim(A, N, ord+1-i)) .+ abs.(selectdim(A, N, ord+1+i))) ./ _getindex(X.weight, space, i)
    end
    return s
end

# Chebyshev

_apply(::Ell1{IdentityWeight}, ::Chebyshev, A::AbstractVector) =
    @inbounds abs(A[1]) + 2sum(abs, view(A, 2:length(A)))
function _apply(::Ell1{IdentityWeight}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(2abs(zero(T)))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    @inbounds for i ∈ ord-1:-1:1
        s .+= abs.(selectdim(A, N, i+1))
    end
    @inbounds s .= 2 .* s .+ abs.(selectdim(A, N, 1))
    return s
end
_apply_dual(::Ell1{IdentityWeight}, ::Chebyshev, A::AbstractVector) =
    @inbounds max(abs(A[1]), maximum(abs, view(A, 2:length(A)))/2)
function _apply_dual(::Ell1{IdentityWeight}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))/2)
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    @inbounds for i ∈ ord-1:-1:1
        s .= max.(s, abs.(selectdim(A, N, i+1)))
    end
    @inbounds s .= max.(s ./ 2, abs.(selectdim(A, N, 1)))
    return s
end

function _apply(X::Ell1{<:GeometricWeight}, space::Chebyshev, A::AbstractVector)
    ν = rate(X.weight)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * one(ν)
    @inbounds for i ∈ ord-1:-1:1
        s = s * ν + abs(A[i+1])
    end
    return @inbounds 2s * ν + abs(A[1])
end
function _apply(X::Ell1{<:GeometricWeight}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    ν = rate(X.weight)
    CoefType = typeof(2abs(zero(T))*ν)
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    @inbounds for i ∈ ord-1:-1:1
        s .= s .* ν .+ abs.(selectdim(A, N, i+1))
    end
    @inbounds s .= 2 .* s .* ν .+ abs.(selectdim(A, N, 1))
    return s
end
function _apply_dual(X::Ell1{<:GeometricWeight}, space::Chebyshev, A::AbstractVector{T}) where {T}
    ν = rate(X.weight)
    ν⁻¹ = abs(one(T))/ν
    ν⁻ⁱ = one(ν⁻¹)/2
    @inbounds s = abs(A[1]) * one(ν⁻ⁱ)
    @inbounds for i ∈ 1:order(space)
        ν⁻ⁱ *= ν⁻¹
        s = max(s, abs(A[i+1]) * ν⁻ⁱ)
    end
    return s
end
function _apply_dual(X::Ell1{<:GeometricWeight}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    ν = rate(X.weight)
    ν⁻¹ = abs(one(T))/ν
    ν⁻ⁱ = one(ν⁻¹)/2
    CoefType = typeof(ν⁻ⁱ)
    @inbounds A₀ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:order(space)
        ν⁻ⁱ *= ν⁻¹
        s .= max.(s, abs.(selectdim(A, N, i+1)) .* ν⁻ⁱ)
    end
    return s
end

function _apply(X::Ell1{<:AlgebraicWeight}, space::Chebyshev, A::AbstractVector)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * _getindex(X.weight, space, ord)
    @inbounds for i ∈ ord-1:-1:1
        s += abs(A[i+1]) * _getindex(X.weight, space, i)
    end
    return @inbounds 2s + abs(A[1])
end
function _apply(X::Ell1{<:AlgebraicWeight}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(2*(abs(zero(T))*_getindex(X.weight, space, 0)))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ) .* _getindex(X.weight, space, ord)
    @inbounds for i ∈ ord-1:-1:1
        s .+= abs.(selectdim(A, N, i+1)) .* _getindex(X.weight, space, i)
    end
    @inbounds s .= 2 .* s .+ abs.(selectdim(A, N, 1))
    return s
end
function _apply_dual(X::Ell1{<:AlgebraicWeight}, space::Chebyshev, A::AbstractVector{T}) where {T}
    ord = order(space)
    @inbounds s = abs(A[ord+1]) / _getindex(X.weight, space, ord)
    @inbounds for i ∈ ord-1:-1:1
        s = max(s, abs(A[i+1]) / _getindex(X.weight, space, i))
    end
    return @inbounds max(s/2, abs(A[1]))
end
function _apply_dual(X::Ell1{<:AlgebraicWeight}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof((abs(zero(T))/_getindex(X.weight, space, 0))/2)
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ) ./ _getindex(X.weight, space, ord)
    @inbounds for i ∈ ord-1:-1:1
        s .= max.(s, abs.(selectdim(A, N, i+1)) ./ _getindex(X.weight, space, i))
    end
    @inbounds s .= max.(s ./ 2, abs.(selectdim(A, N, 1)))
    return s
end

_apply(::Ell2{IdentityWeight}, ::Chebyshev, A::AbstractVector) = @inbounds sqrt(abs2(A[1]) + 4sum(abs2, view(A, 2:length(A))))
_apply_dual(::Ell2{IdentityWeight}, ::Chebyshev, A::AbstractVector) = @inbounds sqrt(abs2(A[1]) + sum(abs2, view(A, 2:length(A)))/4)

_apply(::EllInf{IdentityWeight}, space::Chebyshev, A::AbstractVector) =
    @inbounds max(abs(A[1]), 2maximum(abs, view(A, 2:length(A))))
function _apply(::EllInf{IdentityWeight}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(2abs(zero(T)))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    @inbounds for i ∈ ord-1:-1:1
        s .= max.(s, abs.(selectdim(A, N, i+1)))
    end
    @inbounds s .= max.(2 .* s, abs.(selectdim(A, N, 1)))
    return s
end
_apply_dual(::EllInf{IdentityWeight}, space::Chebyshev, A::AbstractVector) =
    @inbounds abs(A[1]) + sum(abs, view(A, 2:length(A)))/2
function _apply_dual(::EllInf{IdentityWeight}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))/2)
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    @inbounds for i ∈ ord-1:-1:1
        s .+= abs.(selectdim(A, N, i+1))
    end
    @inbounds s .= s ./ 2 .+ abs.(selectdim(A, N, 1))
    return s
end

function _apply(X::EllInf{<:AlgebraicWeight}, space::Chebyshev, A::AbstractVector{T}) where {T}
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * _getindex(X.weight, space, ord)
    @inbounds for i ∈ ord-1:-1:1
        s = max(s, abs(A[i+1]) * _getindex(X.weight, space, i))
    end
    return @inbounds max(2s, abs(A[1]))
end
function _apply(X::EllInf{<:AlgebraicWeight}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(2*(abs(zero(T))*_getindex(X.weight, space, 0)))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ) .* _getindex(X.weight, space, ord)
    @inbounds for i ∈ ord-1:-1:1
        s .= max.(s, abs.(selectdim(A, N, i+1)) .* _getindex(X.weight, space, i))
    end
    @inbounds s .= max.(2 .* s, abs.(selectdim(A, N, 1)))
    return s
end
function _apply_dual(X::EllInf{<:AlgebraicWeight}, space::Chebyshev, A::AbstractVector)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) / _getindex(X.weight, space, ord)
    @inbounds for i ∈ ord-1:-1:1
        s += abs(A[i+1]) / _getindex(X.weight, space, i)
    end
    return @inbounds s/2 + abs(A[1])
end
function _apply_dual(X::EllInf{<:AlgebraicWeight}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof((abs(zero(T))/_getindex(X.weight, space, 0))/2)
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ) ./ _getindex(X.weight, space, ord)
    @inbounds for i ∈ ord-1:-1:1
        s .+= abs.(selectdim(A, N, i+1)) ./ _getindex(X.weight, space, i)
    end
    @inbounds s .= s ./ 2 .+ abs.(selectdim(A, N, 1))
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
        function LinearAlgebra.norm(a::Sequence{<:CartesianSpace}, X::NormedCartesianSpace{<:BanachSpace,$X{IdentityWeight}})
            0 < nspaces(space(a)) || return throw(ArgumentError("number of cartesian products must be strictly positive"))
            return _norm(a, X)
        end
        function LinearAlgebra.norm(a::Sequence{<:CartesianSpace}, X::NormedCartesianSpace{<:NTuple{N,BanachSpace},$X{IdentityWeight}}) where {N}
            0 < nspaces(space(a)) == N || return throw(ArgumentError("number of cartesian products must be strictly positive and equal to the number of inner Banach spaces"))
            return _norm(a, X)
        end

        function LinearAlgebra.opnorm(A::LinearOperator{<:CartesianSpace,ParameterSpace}, X::NormedCartesianSpace{<:BanachSpace,$X{IdentityWeight}})
            0 < nspaces(domain(A)) || return throw(ArgumentError("number of cartesian products must be strictly positive"))
            return _opnorm(A, X)
        end
        function LinearAlgebra.opnorm(A::LinearOperator{<:CartesianSpace,ParameterSpace}, X::NormedCartesianSpace{<:NTuple{N,BanachSpace},$X{IdentityWeight}}) where {N}
            0 < nspaces(domain(A)) == N || return throw(ArgumentError("number of cartesian products must be strictly positive and equal to the number of inner Banach spaces"))
            return _opnorm(A, X)
        end
    end
end

function LinearAlgebra.norm(a::Sequence{<:CartesianSpace}, X::NormedCartesianSpace{<:BanachSpace,Ell2{IdentityWeight}})
    0 < nspaces(space(a)) || return throw(ArgumentError("number of cartesian products must be strictly positive"))
    return sqrt(_norm2(a, X))
end
function LinearAlgebra.norm(a::Sequence{<:CartesianSpace}, X::NormedCartesianSpace{<:NTuple{N,BanachSpace},Ell2{IdentityWeight}}) where {N}
    0 < nspaces(space(a)) == N || return throw(ArgumentError("number of cartesian products must be strictly positive and equal to the number of inner Banach spaces"))
    return sqrt(_norm2(a, X))
end

function LinearAlgebra.opnorm(A::LinearOperator{<:CartesianSpace,ParameterSpace}, X::NormedCartesianSpace{<:BanachSpace,Ell2{IdentityWeight}})
    0 < nspaces(domain(A)) || return throw(ArgumentError("number of cartesian products must be strictly positive"))
    return sqrt(_opnorm2(A, X))
end
function LinearAlgebra.opnorm(A::LinearOperator{<:CartesianSpace,ParameterSpace}, X::NormedCartesianSpace{<:NTuple{N,BanachSpace},Ell2{IdentityWeight}}) where {N}
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
