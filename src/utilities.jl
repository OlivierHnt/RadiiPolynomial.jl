_supremum(x::Real) = x
_supremum(x::Interval) = sup(x)
_infimum(x::Real) = x
_infimum(x::Interval) = inf(x)

# Allocation free reshaping (cf. Issue #36313)
_no_alloc_reshape(a, dims) = invoke(Base._reshape, Tuple{AbstractArray,typeof(dims)}, a, dims)

# Implement fast interval matrix multiplication
# S. M. Rump, [Verification methods: Rigorous results using floating-point arithmetic](https://doi.org/10.1017/S096249291000005X), *Acta Numerica*, **19** (2010), 287-449.
@inline __mul!(C, A, B, α, β) = mul!(C, A, B, α, β)
for (T, S) ∈ ((:Interval, :Interval), (:Interval, :Any), (:Any, :Interval))
    @eval function __mul!(C, A::AbstractMatrix{T}, B::AbstractVecOrMat{S}, α, β) where {T<:$T,S<:$S}
        if iszero(α)
            if iszero(β)
                C .= zero(eltype(C))
            elseif !isone(β)
                C .*= β
            end
        else
            ABinf, ABsup = __mul(A, B)
            if isone(α)
                if iszero(β)
                    C .= Interval.(ABinf, ABsup)
                elseif isone(β)
                    C .+= Interval.(ABinf, ABsup)
                else
                    C .= Interval.(ABinf, ABsup) .+ C .* β
                end
            else
                if iszero(β)
                    C .= Interval.(ABinf, ABsup) .* α
                elseif isone(β)
                    C .+= Interval.(ABinf, ABsup) .* α
                else
                    C .= Interval.(ABinf, ABsup) .* α .+ C .* β
                end
            end
        end
        return C
    end
end
for (T, S) ∈ ((:(Complex{<:Interval}), :(Complex{<:Interval})),
        (:(Complex{<:Interval}), :Complex), (:Complex, :(Complex{<:Interval})))
    @eval function __mul!(C, A::AbstractMatrix{T}, B::AbstractVecOrMat{S}, α, β) where {T<:$T,S<:$S}
        if iszero(α)
            if iszero(β)
                C .= zero(eltype(C))
            elseif !isone(β)
                C .*= β
            end
        else
            A_real, A_imag = reim(A)
            B_real, B_imag = reim(B)
            ABinf_1, ABsup_1 = __mul(A_real, B_real)
            ABinf_2, ABsup_2 = __mul(A_imag, B_imag)
            ABinf_3, ABsup_3 = __mul(A_real, B_imag)
            ABinf_4, ABsup_4 = __mul(A_imag, B_real)
            if isone(α)
                if iszero(β)
                    C .= complex.(Interval.(ABinf_1, ABsup_1) .- Interval.(ABinf_2, ABsup_2),
                        Interval.(ABinf_3, ABsup_3) .+ Interval.(ABinf_4, ABsup_4))
                elseif isone(β)
                    C .+= complex.(Interval.(ABinf_1, ABsup_1) .- Interval.(ABinf_2, ABsup_2),
                        Interval.(ABinf_3, ABsup_3) .+ Interval.(ABinf_4, ABsup_4))
                else
                    C .= complex.(Interval.(ABinf_1, ABsup_1) .- Interval.(ABinf_2, ABsup_2),
                        Interval.(ABinf_3, ABsup_3) .+ Interval.(ABinf_4, ABsup_4)) .+ C .* β
                end
            else
                if iszero(β)
                    C .= complex.(Interval.(ABinf_1, ABsup_1) .- Interval.(ABinf_2, ABsup_2),
                        Interval.(ABinf_3, ABsup_3) .+ Interval.(ABinf_4, ABsup_4)) .* α
                elseif isone(β)
                    C .+= complex.(Interval.(ABinf_1, ABsup_1) .- Interval.(ABinf_2, ABsup_2),
                        Interval.(ABinf_3, ABsup_3) .+ Interval.(ABinf_4, ABsup_4)) .* α
                else
                    C .= complex.(Interval.(ABinf_1, ABsup_1) .- Interval.(ABinf_2, ABsup_2),
                        Interval.(ABinf_3, ABsup_3) .+ Interval.(ABinf_4, ABsup_4)) .* α .+ C .* β
                end
            end
        end
        return C
    end
end
for (T, S) ∈ ((:(Complex{<:Interval}), :Interval), (:(Complex{<:Interval}), :Any), (:Complex, :Interval))
    @eval begin
        function __mul!(C, A::AbstractMatrix{T}, B::AbstractVecOrMat{S}, α, β) where {T<:$T,S<:$S}
            if iszero(α)
                if iszero(β)
                    C .= zero(eltype(C))
                elseif !isone(β)
                    C .*= β
                end
            else
                A_real, A_imag = reim(A)
                ABinf_real, ABsup_real = __mul(A_real, B)
                ABinf_imag, ABsup_imag = __mul(A_imag, B)
                if isone(α)
                    if iszero(β)
                        C .= complex.(Interval.(ABinf_real, ABsup_real), Interval.(ABinf_imag, ABsup_imag))
                    elseif isone(β)
                        C .+= complex.(Interval.(ABinf_real, ABsup_real), Interval.(ABinf_imag, ABsup_imag))
                    else
                        C .= complex.(Interval.(ABinf_real, ABsup_real), Interval.(ABinf_imag, ABsup_imag)) .+ C .* β
                    end
                else
                    if iszero(β)
                        C .= complex.(Interval.(ABinf_real, ABsup_real), Interval.(ABinf_imag, ABsup_imag)) .* α
                    elseif isone(β)
                        C .+= complex.(Interval.(ABinf_real, ABsup_real), Interval.(ABinf_imag, ABsup_imag)) .* α
                    else
                        C .= complex.(Interval.(ABinf_real, ABsup_real), Interval.(ABinf_imag, ABsup_imag)) .* α .+ C .* β
                    end
                end
            end
            return C
        end
        function __mul!(C, A::AbstractMatrix{S}, B::AbstractVecOrMat{T}, α, β) where {S<:$S,T<:$T}
            if iszero(α)
                if iszero(β)
                    C .= zero(eltype(C))
                elseif !isone(β)
                    C .*= β
                end
            else
                B_real, B_imag = reim(B)
                ABinf_real, ABsup_real = __mul(A, B_real)
                ABinf_imag, ABsup_imag = __mul(A, B_imag)
                if isone(α)
                    if iszero(β)
                        C .= complex.(Interval.(ABinf_real, ABsup_real), Interval.(ABinf_imag, ABsup_imag))
                    elseif isone(β)
                        C .+= complex.(Interval.(ABinf_real, ABsup_real), Interval.(ABinf_imag, ABsup_imag))
                    else
                        C .= complex.(Interval.(ABinf_real, ABsup_real), Interval.(ABinf_imag, ABsup_imag)) .+ C .* β
                    end
                else
                    if iszero(β)
                        C .= complex.(Interval.(ABinf_real, ABsup_real), Interval.(ABinf_imag, ABsup_imag)) .* α
                    elseif isone(β)
                        C .+= complex.(Interval.(ABinf_real, ABsup_real), Interval.(ABinf_imag, ABsup_imag)) .* α
                    else
                        C .= complex.(Interval.(ABinf_real, ABsup_real), Interval.(ABinf_imag, ABsup_imag)) .* α .+ C .* β
                    end
                end
            end
            return C
        end
    end
end
function __mul(A::AbstractMatrix{Interval{T}}, B::AbstractVecOrMat{Interval{S}}) where {T<:Real,S<:Real}
    NewType = promote_type(T, S)
    mA, mB, rC, Csup = setrounding(NewType, RoundUp) do
        mA = (inf.(A) .+ sup.(A)) ./ 2
        rA = mA .- inf.(A)
        mB = (inf.(B) .+ sup.(B)) ./ 2
        rB = mB .- inf.(B)
        rC = rA * (abs.(mB) .+ rB)
        rA .= abs.(mA)
        mul!(rC, rA, rB, true, true)
        Csup = mul!(copy(rC), mA, mB, true, true)
        return mA, mB, rC, Csup
    end
    Cinf = setrounding(NewType, RoundDown) do
        return mul!(rC, mA, mB, true, -1)
    end
    return Cinf, Csup
end
function __mul(A::AbstractMatrix{Interval{T}}, B::AbstractVecOrMat{S}) where {T<:Real,S<:Real}
    NewType = promote_type(T, S)
    mA, rC, Csup = setrounding(NewType, RoundUp) do
        mA = (inf.(A) .+ sup.(A)) ./ 2
        rA = mA .- inf.(A)
        rC = rA * abs.(B)
        Csup = mul!(copy(rC), mA, B, true, true)
        return mA, rC, Csup
    end
    Cinf = setrounding(NewType, RoundDown) do
        return mul!(rC, mA, B, true, -1)
    end
    return Cinf, Csup
end
function __mul(A::AbstractMatrix{T}, B::AbstractVecOrMat{Interval{S}}) where {T<:Real,S<:Real}
    NewType = promote_type(T, S)
    mB, rC, Csup = setrounding(NewType, RoundUp) do
        mB = (inf.(B) .+ sup.(B)) ./ 2
        rB = mB .- inf.(B)
        rC = abs.(A) * rB
        Csup = mul!(copy(rC), A, mB, true, true)
        return mB, rC, Csup
    end
    Cinf = setrounding(NewType, RoundDown) do
        return mul!(rC, A, mB, true, -1)
    end
    return Cinf, Csup
end
