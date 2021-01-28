"""
    ZeroFindingProblemCategory1{T₁,T₂,T₃,T₄,T₅,T₆,T₇,T₈,T₉,T₁₀}

Fields:
- `∞f :: T₁`
- `Df :: T₂`
- `D²_abs_f :: T₃`
- `ᴺF :: T₄`
- `ᴺDF :: T₅`
- `ᴺAᴺ :: T₆`
- `ᴺL∞_bound :: T₇`
- `∞L⁻¹∞_bound :: T₈`
- `ν :: T₉`
- `r₀ :: T₁₀`
"""
struct ZeroFindingProblemCategory1{T₁,T₂,T₃,T₄,T₅,T₆,T₇,T₈,T₉,T₁₀}
    ∞f :: T₁
    Df :: T₂
    D²_abs_f :: T₃
    ᴺF :: T₄
    ᴺDF :: T₅
    ᴺAᴺ :: T₆
    ᴺL∞_bound :: T₇
    ∞L⁻¹∞_bound :: T₈
    ν :: T₉
    r₀ :: T₁₀
end

prove(pb::ZeroFindingProblemCategory1) =
    roots_radii_polynomial(Y(pb), Z₁(pb), Z₂(pb), pb.r₀)

Y(pb::ZeroFindingProblemCategory1{T}) where {T<:Sequence} =
    norm_weighted_ℓ¹(pb.ᴺAᴺ * pb.ᴺF, pb.ν) + pb.∞L⁻¹∞_bound * norm_weighted_ℓ¹(pb.∞f, pb.ν)

Y(pb::ZeroFindingProblemCategory1{T}) where {T<:AbstractVector{<:Sequence}} =
    norm(opnorm_weighted_ℓ¹.(pb.ᴺAᴺ * pb.ᴺF, pb.ν), Inf) + pb.∞L⁻¹∞_bound * norm(norm_weighted_ℓ¹.(pb.∞f, pb.ν), Inf)

Z₁(pb::ZeroFindingProblemCategory1{T}) where {T<:Sequence} =
    opnorm_weighted_ℓ¹(pb.ᴺAᴺ * pb.ᴺDF -̄ I, pb.ν, pb.ν) + pb.ᴺL∞_bound * opnorm_weighted_ℓ¹(pb.ᴺAᴺ, pb.ν, pb.ν) + pb.∞L⁻¹∞_bound * norm_weighted_ℓ¹(pb.Df, pb.ν)

Z₁(pb::ZeroFindingProblemCategory1{T}) where {T<:AbstractVector{<:Sequence}} =
    opnorm(opnorm_weighted_ℓ¹.(pb.ᴺAᴺ * pb.ᴺDF -̄ I, pb.ν, pb.ν), Inf) + pb.ᴺL∞_bound * opnorm(norm_weighted_ℓ¹.(pb.ᴺAᴺ, pb.ν, pb.ν), Inf) + pb.∞L⁻¹∞_bound * opnorm(norm_weighted_ℓ¹.(pb.Df, pb.ν), Inf)

Z₂(pb::ZeroFindingProblemCategory1{T}) where {T<:Sequence} =
    (opnorm_weighted_ℓ¹(pb.ᴺAᴺ, pb.ν, pb.ν) + pb.∞L⁻¹∞_bound) * pb.D²_abs_f

Z₂(pb::ZeroFindingProblemCategory1{T}) where {T<:AbstractVector{<:Sequence}} =
    (opnorm(opnorm_weighted_ℓ¹.(pb.ᴺAᴺ, pb.ν, pb.ν), Inf) + pb.∞L⁻¹∞_bound) * opnorm(pb.D²_abs_f, Inf)

#

"""
    TailProblem{T₁,T₂,T₃,T₄,T₅,T₆}

Fields:
- `∞f :: T₁`
- `Df :: T₂`
- `D²_abs_f :: T₃`
- `∞L∞_bound :: T₄`
- `ν :: T₅`
- `r₀ :: T₆`
"""
struct TailProblem{T₁,T₂,T₃,T₄,T₅,T₆}
    ∞f :: T₁
    Df :: T₂
    D²_abs_f :: T₃
    ∞L∞_bound :: T₄
    ν :: T₅
    r₀ :: T₆
end

prove(pb::TailProblem) =
    roots_radii_polynomial(Y(pb), Z₁(pb), Z₂(pb), pb.r₀)

Y(pb::TailProblem{T}) where {T<:Sequence} =
    pb.∞L∞_bound * norm_weighted_ℓ¹(pb.∞f, pb.ν)

Y(pb::TailProblem{T}) where {T<:AbstractVector{<:Sequence}} =
    pb.∞L∞_bound * norm(norm_weighted_ℓ¹.(pb.∞f, pb.ν), Inf)

Z₁(pb::TailProblem{T}) where {T<:Sequence} =
    pb.∞L∞_bound * norm_weighted_ℓ¹(pb.Df, pb.ν)

Z₁(pb::TailProblem{T}) where {T<:AbstractVector{<:Sequence}} =
    pb.∞L∞_bound * opnorm(norm_weighted_ℓ¹.(pb.Df, pb.ν), Inf)

Z₂(pb::TailProblem{T}) where {T<:Sequence} =
    pb.∞L∞_bound * pb.D²_abs_f

Z₂(pb::TailProblem{T}) where {T<:AbstractVector{<:Sequence}} =
    pb.∞L∞_bound * opnorm(pb.D²_abs_f, Inf)
