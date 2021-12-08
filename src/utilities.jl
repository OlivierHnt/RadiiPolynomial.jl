# Allocation free reshaping (cf. Issue #36313)
_no_alloc_reshape(a, dims) = invoke(Base._reshape, Tuple{AbstractArray,typeof(dims)}, a, dims)

# Implement fast interval matrix multiplication
# S. M. Rump, [Verification methods: Rigorous results using floating-point arithmetic](https://doi.org/10.1017/S096249291000005X), *Acta Numerica*, **19** (2010), 287-449.
@inline __mul!(C, A, B, α, β) = mul!(C, A, B, α, β)
for (T, S) ∈ ((:Interval, :Interval), (:Interval, :Any), (:Any, :Interval))
    @eval @inline function __mul!(C, A::AbstractMatrix{T}, B::AbstractVecOrMat{S}, α, β) where {T<:$T,S<:$S}
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
    @eval @inline function __mul!(C, A::AbstractMatrix{T}, B::AbstractVecOrMat{S}, α, β) where {T<:$T,S<:$S}
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
        @inline function __mul!(C, A::AbstractMatrix{T}, B::AbstractVecOrMat{S}, α, β) where {T<:$T,S<:$S}
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
        @inline function __mul!(C, A::AbstractMatrix{S}, B::AbstractVecOrMat{T}, α, β) where {S<:$S,T<:$T}
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
@inline function __mul(A::AbstractMatrix{Interval{T}}, B::AbstractVecOrMat{Interval{S}}) where {T<:Real,S<:Real}
    NewType = promote_type(T, S)
    Ainf = inf.(A)
    Asup = sup.(A)
    Binf = inf.(B)
    Bsup = sup.(B)
    mA, mB, R, Csup = setrounding(NewType, RoundUp) do
        mA = Ainf .+ (Asup .- Ainf) ./ 2
        mB = Binf .+ (Bsup .- Binf) ./ 2
        rA = mA - Ainf
        rB = mB - Binf
        R = abs.(mA) * rB .+ rA * (abs.(mB) .+ rB)
        Csup = mA * mB + R
        return mA, mB, R, Csup
    end
    Cinf = setrounding(NewType, RoundDown) do
        return mA * mB - R
    end
    return Cinf, Csup
end
@inline function __mul(A::AbstractMatrix{Interval{T}}, B::AbstractVecOrMat{S}) where {T<:Real,S<:Real}
    NewType = promote_type(T, S)
    Ainf = inf.(A)
    Asup = sup.(A)
    mA, R, Csup = setrounding(NewType, RoundUp) do
        mA = Ainf .+ (Asup .- Ainf) ./ 2
        rA = mA - Ainf
        R = rA * abs.(B)
        Csup = mA * B + R
        return mA, R, Csup
    end
    Cinf = setrounding(NewType, RoundDown) do
        return mA * B - R
    end
    return Cinf, Csup
end
@inline function __mul(A::AbstractMatrix{T}, B::AbstractVecOrMat{Interval{S}}) where {T<:Real,S<:Real}
    NewType = promote_type(T, S)
    Binf = inf.(B)
    Bsup = sup.(B)
    mB, R, Csup = setrounding(NewType, RoundUp) do
        mB = Binf .+ (Bsup .- Binf) ./ 2
        rB = mB - Binf
        R = abs.(A) * rB
        Csup = A * mB + R
        return mB, R, Csup
    end
    Cinf = setrounding(NewType, RoundDown) do
        return A * mB - R
    end
    return Cinf, Csup
end
