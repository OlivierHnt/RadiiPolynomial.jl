# safe equality

_safe_isequal(x, y) = x == y
_safe_isequal(x::Union{Interval,Complex{<:Interval}}, y::Union{Interval,Complex{<:Interval}}) =
    isequal_interval(x, y)

# allocation free reshaping (cf. Issue #36313)

_no_alloc_reshape(a, dims) = invoke(Base._reshape, Tuple{AbstractArray,typeof(dims)}, a, dims)

# implement fast interval matrix multiplication

@inline __mul!(C, A, B, α, β) = mul!(C, A, B, α, β)

for (T, S) ∈ ((:Interval, :Interval), (:Interval, :Any), (:Any, :Interval))
    @eval function __mul!(C, A::AbstractMatrix{<:$T}, B::AbstractVecOrMat{<:$S}, α, β)
        CoefType = eltype(C)
        if iszero(α)
            if iszero(β)
                C .= zero(CoefType)
            elseif !isone(β)
                C .*= β
            end
        else
            BoundType = IntervalArithmetic.numtype(CoefType)
            ABinf, ABsup = __mul(A, B)
            if isone(α)
                if iszero(β)
                    C .= interval.(BoundType, ABinf, ABsup)
                elseif isone(β)
                    C .+= interval.(BoundType, ABinf, ABsup)
                else
                    C .= interval.(BoundType, ABinf, ABsup) .+ C .* β
                end
            else
                if iszero(β)
                    C .= interval.(BoundType, ABinf, ABsup) .* α
                elseif isone(β)
                    C .+= interval.(BoundType, ABinf, ABsup) .* α
                else
                    C .= interval.(BoundType, ABinf, ABsup) .* α .+ C .* β
                end
            end
        end
        t = all(isguaranteed, A) & all(isguaranteed, B) & isguaranteed(α) & isguaranteed(β)
        C .= IntervalArithmetic._unsafe_interval.(getfield.(C, :bareinterval), decoration.(C), t)
        return C
    end
end

for (T, S) ∈ ((:(Complex{<:Interval}), :(Complex{<:Interval})),
        (:(Complex{<:Interval}), :Complex), (:Complex, :(Complex{<:Interval})))
    @eval function __mul!(C, A::AbstractMatrix{<:$T}, B::AbstractVecOrMat{<:$S}, α, β)
        CoefType = eltype(C)
        if iszero(α)
            if iszero(β)
                C .= zero(CoefType)
            elseif !isone(β)
                C .*= β
            end
        else
            BoundType = IntervalArithmetic.numtype(CoefType)
            A_real, A_imag = reim(A)
            B_real, B_imag = reim(B)
            ABinf_1, ABsup_1 = __mul(A_real, B_real)
            ABinf_2, ABsup_2 = __mul(A_imag, B_imag)
            ABinf_3, ABsup_3 = __mul(A_real, B_imag)
            ABinf_4, ABsup_4 = __mul(A_imag, B_real)
            if isone(α)
                if iszero(β)
                    C .= complex.(interval.(BoundType, ABinf_1, ABsup_1) .- interval.(BoundType, ABinf_2, ABsup_2),
                                  interval.(BoundType, ABinf_3, ABsup_3) .+ interval.(BoundType, ABinf_4, ABsup_4))
                elseif isone(β)
                    C .+= complex.(interval.(BoundType, ABinf_1, ABsup_1) .- interval.(BoundType, ABinf_2, ABsup_2),
                                   interval.(BoundType, ABinf_3, ABsup_3) .+ interval.(BoundType, ABinf_4, ABsup_4))
                else
                    C .= complex.(interval.(BoundType, ABinf_1, ABsup_1) .- interval.(BoundType, ABinf_2, ABsup_2),
                                  interval.(BoundType, ABinf_3, ABsup_3) .+ interval.(BoundType, ABinf_4, ABsup_4)) .+ C .* β
                end
            else
                if iszero(β)
                    C .= complex.(interval.(BoundType, ABinf_1, ABsup_1) .- interval.(BoundType, ABinf_2, ABsup_2),
                                  interval.(BoundType, ABinf_3, ABsup_3) .+ interval.(BoundType, ABinf_4, ABsup_4)) .* α
                elseif isone(β)
                    C .+= complex.(interval.(BoundType, ABinf_1, ABsup_1) .- interval.(BoundType, ABinf_2, ABsup_2),
                                   interval.(BoundType, ABinf_3, ABsup_3) .+ interval.(BoundType, ABinf_4, ABsup_4)) .* α
                else
                    C .= complex.(interval.(BoundType, ABinf_1, ABsup_1) .- interval.(BoundType, ABinf_2, ABsup_2),
                                  interval.(BoundType, ABinf_3, ABsup_3) .+ interval.(BoundType, ABinf_4, ABsup_4)) .* α .+ C .* β
                end
            end
        end
        t = all(isguaranteed, A) & all(isguaranteed, B) & isguaranteed(α) & isguaranteed(β)
        C .= complex.(
                IntervalArithmetic._unsafe_interval.(getfield.(real.(C), :bareinterval), decoration.(C), t),
                IntervalArithmetic._unsafe_interval.(getfield.(imag.(C), :bareinterval), decoration.(C), t))
        return C
    end
end

for (T, S) ∈ ((:(Complex{<:Interval}), :Interval), (:(Complex{<:Interval}), :Any), (:Complex, :Interval))
    @eval begin
        function __mul!(C, A::AbstractMatrix{<:$T}, B::AbstractVecOrMat{<:$S}, α, β)
            CoefType = eltype(C)
            if iszero(α)
                if iszero(β)
                    C .= zero(CoefType)
                elseif !isone(β)
                    C .*= β
                end
            else
                BoundType = IntervalArithmetic.numtype(CoefType)
                A_real, A_imag = reim(A)
                ABinf_real, ABsup_real = __mul(A_real, B)
                ABinf_imag, ABsup_imag = __mul(A_imag, B)
                if isone(α)
                    if iszero(β)
                        C .= complex.(interval.(BoundType, ABinf_real, ABsup_real), interval.(BoundType, ABinf_imag, ABsup_imag))
                    elseif isone(β)
                        C .+= complex.(interval.(BoundType, ABinf_real, ABsup_real), interval.(BoundType, ABinf_imag, ABsup_imag))
                    else
                        C .= complex.(interval.(BoundType, ABinf_real, ABsup_real), interval.(BoundType, ABinf_imag, ABsup_imag)) .+ C .* β
                    end
                else
                    if iszero(β)
                        C .= complex.(interval.(BoundType, ABinf_real, ABsup_real), interval.(BoundType, ABinf_imag, ABsup_imag)) .* α
                    elseif isone(β)
                        C .+= complex.(interval.(BoundType, ABinf_real, ABsup_real), interval.(BoundType, ABinf_imag, ABsup_imag)) .* α
                    else
                        C .= complex.(interval.(BoundType, ABinf_real, ABsup_real), interval.(BoundType, ABinf_imag, ABsup_imag)) .* α .+ C .* β
                    end
                end
            end
            t = all(isguaranteed, A) & all(isguaranteed, B) & isguaranteed(α) & isguaranteed(β)
            C .= complex.(
                IntervalArithmetic._unsafe_interval.(getfield.(real.(C), :bareinterval), decoration.(C), t),
                IntervalArithmetic._unsafe_interval.(getfield.(imag.(C), :bareinterval), decoration.(C), t))
            return C
        end

        function __mul!(C, A::AbstractMatrix{<:$S}, B::AbstractVecOrMat{<:$T}, α, β)
            CoefType = eltype(C)
            if iszero(α)
                if iszero(β)
                    C .= zero(CoefType)
                elseif !isone(β)
                    C .*= β
                end
            else
                BoundType = IntervalArithmetic.numtype(CoefType)
                B_real, B_imag = reim(B)
                ABinf_real, ABsup_real = __mul(A, B_real)
                ABinf_imag, ABsup_imag = __mul(A, B_imag)
                if isone(α)
                    if iszero(β)
                        C .= complex.(interval.(BoundType, ABinf_real, ABsup_real), interval.(BoundType, ABinf_imag, ABsup_imag))
                    elseif isone(β)
                        C .+= complex.(interval.(BoundType, ABinf_real, ABsup_real), interval.(BoundType, ABinf_imag, ABsup_imag))
                    else
                        C .= complex.(interval.(BoundType, ABinf_real, ABsup_real), interval.(BoundType, ABinf_imag, ABsup_imag)) .+ C .* β
                    end
                else
                    if iszero(β)
                        C .= complex.(interval.(BoundType, ABinf_real, ABsup_real), interval.(BoundType, ABinf_imag, ABsup_imag)) .* α
                    elseif isone(β)
                        C .+= complex.(interval.(BoundType, ABinf_real, ABsup_real), interval.(BoundType, ABinf_imag, ABsup_imag)) .* α
                    else
                        C .= complex.(interval.(BoundType, ABinf_real, ABsup_real), interval.(BoundType, ABinf_imag, ABsup_imag)) .* α .+ C .* β
                    end
                end
            end
            t = all(isguaranteed, A) & all(isguaranteed, B) & isguaranteed(α) & isguaranteed(β)
            C .= complex.(
                IntervalArithmetic._unsafe_interval.(getfield.(real.(C), :bareinterval), decoration.(C), t),
                IntervalArithmetic._unsafe_interval.(getfield.(imag.(C), :bareinterval), decoration.(C), t))
            return C
        end
    end
end

__mul(A::AbstractMatrix{<:Interval{<:Rational}}, B::AbstractVecOrMat{<:Interval{<:Rational}}) = A * B

function __mul(A::AbstractMatrix{T}, B::AbstractVecOrMat{S}) where {T,S}
    NewType = IntervalArithmetic.promote_numtype(T, S)
    return __mul(interval.(NewType, A), interval.(NewType, B))
end

function __mul(A::AbstractMatrix{Interval{T}}, B::AbstractMatrix{Interval{T}}) where {T<:AbstractFloat}
    mA = IntervalArithmetic._div_round.(IntervalArithmetic._add_round.(inf.(A), sup.(A), RoundUp), convert(T, 2), RoundUp) # (inf.(A) .+ sup.(A)) ./ 2
    rA = IntervalArithmetic._sub_round.(mA, inf.(A), RoundUp)
    mB = IntervalArithmetic._div_round.(IntervalArithmetic._add_round.(inf.(B), sup.(B), RoundUp), convert(T, 2), RoundUp) # (inf.(B) .+ sup.(B)) ./ 2
    rB = IntervalArithmetic._sub_round.(mB, inf.(B), RoundUp)

    Cinf = zeros(T, size(A, 1), size(B, 2))
    Csup = zeros(T, size(A, 1), size(B, 2))

    Threads.@threads for j ∈ axes(B, 2)
        for l ∈ axes(A, 2)
            @inbounds for i ∈ axes(A, 1)
                U_ij         = IntervalArithmetic._mul_round(abs(mA[i,l]), rB[l,j], RoundUp)
                V_ij         = IntervalArithmetic._mul_round(rA[i,l], IntervalArithmetic._add_round(abs(mB[l,j]), rB[l,j], RoundUp), RoundUp)
                rC_ij        = IntervalArithmetic._add_round(U_ij, V_ij, RoundUp)
                mAmB_up_ij   = IntervalArithmetic._mul_round(mA[i,l], mB[l,j], RoundUp)
                mAmB_down_ij = IntervalArithmetic._mul_round(mA[i,l], mB[l,j], RoundDown)

                Cinf[i,j] = IntervalArithmetic._add_round(IntervalArithmetic._sub_round(mAmB_down_ij, rC_ij, RoundDown), Cinf[i,j], RoundDown)
                Csup[i,j] = IntervalArithmetic._add_round(IntervalArithmetic._add_round(mAmB_up_ij,   rC_ij, RoundUp),   Csup[i,j], RoundUp)
            end
        end
    end

    return Cinf, Csup
end

function __mul(A::AbstractMatrix{Interval{T}}, B::AbstractMatrix{T}) where {T<:AbstractFloat}
    mA = IntervalArithmetic._div_round.(IntervalArithmetic._add_round.(inf.(A), sup.(A), RoundUp), convert(T, 2), RoundUp) # (inf.(A) .+ sup.(A)) ./ 2
    rA = IntervalArithmetic._sub_round.(mA, inf.(A), RoundUp)

    Cinf = zeros(T, size(A, 1), size(B, 2))
    Csup = zeros(T, size(A, 1), size(B, 2))

    Threads.@threads for j ∈ axes(B, 2)
        for l ∈ axes(A, 2)
            @inbounds for i ∈ axes(A, 1)
                rC_ij        = IntervalArithmetic._mul_round(rA[i,l], abs(B[l,j]), RoundUp)
                mAmB_up_ij   = IntervalArithmetic._mul_round(mA[i,l], B[l,j], RoundUp)
                mAmB_down_ij = IntervalArithmetic._mul_round(mA[i,l], B[l,j], RoundDown)

                Cinf[i,j] = IntervalArithmetic._add_round(IntervalArithmetic._sub_round(mAmB_down_ij, rC_ij, RoundDown), Cinf[i,j], RoundDown)
                Csup[i,j] = IntervalArithmetic._add_round(IntervalArithmetic._add_round(mAmB_up_ij,   rC_ij, RoundUp),   Csup[i,j], RoundUp)
            end
        end
    end

    return Cinf, Csup
end

function __mul(A::AbstractMatrix{T}, B::AbstractMatrix{Interval{T}}) where {T<:AbstractFloat}
    mB = IntervalArithmetic._div_round.(IntervalArithmetic._add_round.(inf.(B), sup.(B), RoundUp), convert(T, 2), RoundUp) # (inf.(B) .+ sup.(B)) ./ 2
    rB = IntervalArithmetic._sub_round.(mB, inf.(B), RoundUp)

    Cinf = zeros(T, size(A, 1), size(B, 2))
    Csup = zeros(T, size(A, 1), size(B, 2))

    Threads.@threads for j ∈ axes(B, 2)
        for l ∈ axes(A, 2)
            @inbounds for i ∈ axes(A, 1)
                rC_ij        = IntervalArithmetic._mul_round(abs(A[i,l]), rB[l,j], RoundUp)
                mAmB_up_ij   = IntervalArithmetic._mul_round(A[i,l], mB[l,j], RoundUp)
                mAmB_down_ij = IntervalArithmetic._mul_round(A[i,l], mB[l,j], RoundDown)

                Cinf[i,j] = IntervalArithmetic._add_round(IntervalArithmetic._sub_round(mAmB_down_ij, rC_ij, RoundDown), Cinf[i,j], RoundDown)
                Csup[i,j] = IntervalArithmetic._add_round(IntervalArithmetic._add_round(mAmB_up_ij,   rC_ij, RoundUp),   Csup[i,j], RoundUp)
            end
        end
    end

    return Cinf, Csup
end

function __mul(A::AbstractMatrix{Interval{T}}, B::AbstractVector{Interval{T}}) where {T<:AbstractFloat}
    mA = IntervalArithmetic._div_round.(IntervalArithmetic._add_round.(inf.(A), sup.(A), RoundUp), convert(T, 2), RoundUp) # (inf.(A) .+ sup.(A)) ./ 2
    rA = IntervalArithmetic._sub_round.(mA, inf.(A), RoundUp)
    mB = IntervalArithmetic._div_round.(IntervalArithmetic._add_round.(inf.(B), sup.(B), RoundUp), convert(T, 2), RoundUp) # (inf.(B) .+ sup.(B)) ./ 2
    rB = IntervalArithmetic._sub_round.(mB, inf.(B), RoundUp)

    Cinf = zeros(T, size(A, 1))
    Csup = zeros(T, size(A, 1))

    Threads.@threads for i ∈ axes(A, 1)
        @inbounds for l ∈ axes(A, 2)
            U_il         = IntervalArithmetic._mul_round(abs(mA[i,l]), rB[l], RoundUp)
            V_il         = IntervalArithmetic._mul_round(rA[i,l], IntervalArithmetic._add_round(abs(mB[l]), rB[l], RoundUp), RoundUp)
            rC_il        = IntervalArithmetic._add_round(U_il, V_il, RoundUp)
            mAmB_up_il   = IntervalArithmetic._mul_round(mA[i,l], mB[l], RoundUp)
            mAmB_down_il = IntervalArithmetic._mul_round(mA[i,l], mB[l], RoundDown)

            Cinf[i] = IntervalArithmetic._add_round(IntervalArithmetic._sub_round(mAmB_down_il, rC_il, RoundDown), Cinf[i], RoundDown)
            Csup[i] = IntervalArithmetic._add_round(IntervalArithmetic._add_round(mAmB_up_il,   rC_il, RoundUp),   Csup[i], RoundUp)
        end
    end

    return Cinf, Csup
end

function __mul(A::AbstractMatrix{Interval{T}}, B::AbstractVector{T}) where {T<:AbstractFloat}
    mA = IntervalArithmetic._div_round.(IntervalArithmetic._add_round.(inf.(A), sup.(A), RoundUp), convert(T, 2), RoundUp) # (inf.(A) .+ sup.(A)) ./ 2
    rA = IntervalArithmetic._sub_round.(mA, inf.(A), RoundUp)

    Cinf = zeros(T, size(A, 1))
    Csup = zeros(T, size(A, 1))

    Threads.@threads for i ∈ axes(A, 1)
        @inbounds for l ∈ axes(A, 2)
            rC_il       = IntervalArithmetic._mul_round(rA[i,l], abs(B[l]), RoundUp)
            mAB_up_il   = IntervalArithmetic._mul_round(mA[i,l], B[l], RoundUp)
            mAB_down_il = IntervalArithmetic._mul_round(mA[i,l], B[l], RoundDown)

            Cinf[i] = IntervalArithmetic._add_round(IntervalArithmetic._sub_round(mAB_down_il, rC_il, RoundDown), Cinf[i], RoundDown)
            Csup[i] = IntervalArithmetic._add_round(IntervalArithmetic._add_round(mAB_up_il,   rC_il, RoundUp),   Csup[i], RoundUp)
        end
    end

    return Cinf, Csup
end

function __mul(A::AbstractMatrix{T}, B::AbstractVector{Interval{T}}) where {T<:AbstractFloat}
    mB = IntervalArithmetic._div_round.(IntervalArithmetic._add_round.(inf.(B), sup.(B), RoundUp), convert(T, 2), RoundUp) # (inf.(B) .+ sup.(B)) ./ 2
    rB = IntervalArithmetic._sub_round.(mB, inf.(B), RoundUp)

    Cinf = zeros(T, size(A, 1))
    Csup = zeros(T, size(A, 1))

    Threads.@threads for i ∈ axes(A, 1)
        @inbounds for l ∈ axes(A, 2)
            rC_il       = IntervalArithmetic._mul_round(abs(A[i,l]), rB[l], RoundUp)
            AmB_up_il   = IntervalArithmetic._mul_round(A[i,l], mB[l], RoundUp)
            AmB_down_il = IntervalArithmetic._mul_round(A[i,l], mB[l], RoundDown)

            Cinf[i] = IntervalArithmetic._add_round(IntervalArithmetic._sub_round(AmB_down_il, rC_il, RoundDown), Cinf[i], RoundDown)
            Csup[i] = IntervalArithmetic._add_round(IntervalArithmetic._add_round(AmB_up_il,   rC_il, RoundUp),   Csup[i], RoundUp)
        end
    end

    return Cinf, Csup
end
