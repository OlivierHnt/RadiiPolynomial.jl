"""
    FixedPointProblemFiniteDimension{T₁,T₂,T₃,T₄,T₅,T₆}

Fields:
- `x₀ :: T₁`
- `G :: T₂`
- `DG :: T₃`
- `D²G :: T₄`
- `p :: T₅`
- `r₀ :: T₆`
"""
struct FixedPointProblemFiniteDimension{T₁,T₂,T₃,T₄,T₅,T₆}
    x₀ :: T₁
    G :: T₂
    DG :: T₃
    D²G :: T₄
    p :: T₅
    r₀ :: T₆
end

FixedPointProblemFiniteDimension(; x₀, G, DG, D²G, p, r₀) =
    FixedPointProblemFiniteDimension(x₀, G, DG, D²G, p, r₀)

function prove(pb::FixedPointProblemFiniteDimension, n::Int)
    n == 1 && return roots_radii_polynomial(Y(pb), Z₁(pb), pb.r₀)
    n == 2 && return roots_radii_polynomial(Y(pb), Z₁(pb), Z₂(pb), pb.r₀)
    return throw(DomainError(n, "only the Radii Polynomial Theorem of order 1 and 2 is supported"))
end

"""
    Y(pb::FixedPointProblemFiniteDimension)

Compute the norm ``\\| G(x_0) - x_0 \\|_p``.
"""
Y(pb::FixedPointProblemFiniteDimension) = norm(pb.G - pb.x₀, pb.p)

"""
    Z₁(pb::FixedPointProblemFiniteDimension)

Compute the operator norm ``\\| D G(y) \\|_p`` where ``y = x_0`` or ``y \\in \\overline{B_{r_0}(x_0)}``.
"""
Z₁(pb::FixedPointProblemFiniteDimension) = opnorm(pb.DG, pb.p)

"""
    Z₂(pb::FixedPointProblemFiniteDimension)

Compute the operator norm ``\\| D^2 G(y) \\|_p`` where ``y \\in \\overline{B_{r_0}(x_0)}``.
"""
Z₂(pb::FixedPointProblemFiniteDimension) = opnorm(pb.D²G, pb.p)

#

"""
    ZeroFindingProblemFiniteDimension{T₁,T₂,T₃,T₄,T₅,T₆,T₇}

Fields:
- `x₀ :: T₁`
- `F :: T₂`
- `DF :: T₃`
- `A :: T₄`
- `D²F :: T₅`
- `p :: T₆`
- `r₀ :: T₇`
"""
struct ZeroFindingProblemFiniteDimension{T₁,T₂,T₃,T₄,T₅,T₆,T₇}
    x₀ :: T₁
    F :: T₂
    DF :: T₃
    A :: T₄
    D²F :: T₅
    p :: T₆
    r₀ :: T₇
end

ZeroFindingProblemFiniteDimension(; x₀, F, DF, A, D²F, p, r₀) =
    ZeroFindingProblemFiniteDimension(x₀, F, DF, A, D²F, p, r₀)

function prove(pb::ZeroFindingProblemFiniteDimension, n::Int)
    n == 1 && return roots_radii_polynomial(Y(pb), Z₁(pb), pb.r₀)
    n == 2 && return roots_radii_polynomial(Y(pb), Z₁(pb), Z₂(pb), pb.r₀)
    return throw(DomainError(n, "only the Radii Polynomial Theorem of order 1 and 2 is supported"))
end

"""
    Y(pb::ZeroFindingProblemFiniteDimension)

Compute the norm ``\\| A F(x_0) \\|_p``.
"""
Y(pb::ZeroFindingProblemFiniteDimension) = norm(pb.A * pb.F, pb.p)

"""
    Z₁(pb::ZeroFindingProblemFiniteDimension)

Compute the operator norm ``\\| A D F(y) - I \\|_p`` where ``y = x_0`` or ``y \\in \\overline{B_{r_0}(x_0)}``.
"""
Z₁(pb::ZeroFindingProblemFiniteDimension) = opnorm(pb.A * pb.DF - I, pb.p)

"""
    Z₂(pb::ZeroFindingProblemFiniteDimension)

Compute the operator norm ``\\| A D^2 F(y) \\|_p`` where ``y \\in \\overline{B_{r_0}(x_0)}``.
"""
Z₂(pb::ZeroFindingProblemFiniteDimension) = opnorm(pb.A * pb.D²F, pb.p)
