@testset "LinearOperator arithmetic" begin
    M₂, N₂, I₂ = [1.0 2.0 ; 3.0 4.0], [5.0 6.0 ; 7.0 8.0], [1.0 0.0 ; 0.0 1.0]
    M₂₃ = [1.0 2.0 3.0 ; 4.0 5.0 6.0]
    𝒯₀, 𝒯₁, 𝒯₂ = Taylor(0), Taylor(1), Taylor(2)
    𝕂 = ScalarSpace()
    A = LinearOperator(𝒯₁, 𝒯₁, M₂)
    B = LinearOperator(𝒯₁, 𝒯₁, N₂)

    @testset "unary +, - and lazy Negate" begin
        @test +A == A
        @test -A == LinearOperator(𝒯₁, 𝒯₁, -M₂)

        # the negation of a lazy composition stays lazy
        S = A ∘ B
        @test S isa ComposedOperator
        @test -S isa Negate
        @test -(-S) === S
        @test project(-S, 𝒯₁, 𝒯₁) == -(A * B) == LinearOperator(𝒯₁, 𝒯₁, -(M₂ * N₂))
    end

    @testset "+ and - adapt the domain and the codomain" begin
        @testset "$S" for S ∈ (Taylor, Chebyshev)
            s₁, s₂ = S(1), S(2)
            Aₛ = LinearOperator(s₁, s₁, M₂)
            Bₛ = LinearOperator(s₂, s₁, M₂₃)
            Dₛ = LinearOperator(s₁, s₂, [1.0 2.0 ; 3.0 4.0 ; 5.0 6.0])

            # out of place and `ladd!`/`lsub!`: the union of the spaces, Aₛ being folded into the shared indices
            @test domain(Aₛ + Bₛ) == s₂ && codomain(Aₛ + Bₛ) == s₁
            @test coefficients(Aₛ + Bₛ) == [2.0 4.0 3.0 ; 7.0 9.0 6.0]
            @test Aₛ + Bₛ == Aₛ - (-Bₛ) == ladd!(copy(Aₛ), copy(Bₛ)) == lsub!(copy(Aₛ), -Bₛ)
            @test coefficients(Aₛ + Dₛ) == [2.0 4.0 ; 6.0 8.0 ; 5.0 6.0]
            @test Aₛ + Dₛ == Aₛ - (-Dₛ)

            # `radd!`/`rsub!`: the spaces of Aₛ, the coefficients of Bₛ outside them being discarded
            @test coefficients(radd!(copy(Aₛ), Bₛ)) == [2.0 4.0 ; 7.0 9.0]
            @test radd!(copy(Aₛ), Bₛ) == rsub!(copy(Aₛ), -Bₛ)
        end

        @testset "Fourier" begin
            M₃, M₃₅ = Float64[1 2 3 ; 4 5 6 ; 7 8 9], Float64[1 2 3 4 5 ; 6 7 8 9 10 ; 11 12 13 14 15]
            Aℱ = LinearOperator(Fourier(1, 1.0), Fourier(1, 1.0), M₃)
            Bℱ = LinearOperator(Fourier(2, 1.0), Fourier(1, 1.0), M₃₅)
            # the frequencies -1, 0, 1 of Aℱ are the columns 2, 3, 4 of Bℱ
            expected = copy(M₃₅)
            expected[:, 2:4] .+= M₃
            @test coefficients(Aℱ + Bℱ) == expected
            @test Aℱ + Bℱ == Aℱ - (-Bℱ)
        end

        @testset "TensorSpace" begin
            𝒮 = 𝒯₁ ⊗ Chebyshev(1)
            Aₜ = LinearOperator(𝒮, 𝒮, Matrix(reshape(1.0:16.0, 4, 4)))
            Bₜ = LinearOperator(𝒮, 𝒮, Matrix{Float64}(I, 4, 4))
            @test coefficients(Aₜ + Bₜ) == coefficients(Aₜ) .+ coefficients(Bₜ)
            @test Aₜ + Bₜ == Aₜ - (-Bₜ)
        end

        @testset "SymmetricSpace" begin
            𝓈₂, 𝓈₁ = evensym(𝒯₂), evensym(𝒯₁) # indices (0, 2) and (0,)
            Aₑ = LinearOperator(𝓈₂, 𝓈₂, M₂)
            @test coefficients(Aₑ + LinearOperator(𝓈₂, 𝓈₂, I₂)) == [2.0 2.0 ; 3.0 5.0]
            @test coefficients(LinearOperator(𝓈₁, 𝓈₁, fill(5.0, 1, 1)) + Aₑ) == [6.0 2.0 ; 3.0 4.0]

            # the in-place product projects its operands and its output across symmetries, like `*`
            D = LinearOperator(𝒯₂, 𝒯₂, reshape(1.0:9.0, 3, 3))
            @test coefficients(mul!(zeros(𝓈₂, 𝓈₂), D, Aₑ, 1.0, 0.0)) ≈ coefficients(project(D * Aₑ, 𝓈₂, 𝓈₂))
            @test coefficients(mul!(zeros(𝒯₂, 𝒯₂), Aₑ, D, 1.0, 0.0)) ≈ coefficients(project(Aₑ * D, 𝒯₂, 𝒯₂))
        end
    end

    @testset "scalar *, / and their in-place forms" begin
        @test A * 3.0 == 3.0 * A == rmul!(copy(A), 3.0) == lmul!(3.0, copy(A)) == LinearOperator(𝒯₁, 𝒯₁, 3.0 .* M₂)
        @test A / 2.0 == 2.0 \ A == rdiv!(copy(A), 2.0) == ldiv!(2.0, copy(A)) == LinearOperator(𝒯₁, 𝒯₁, M₂ ./ 2.0)

        Ai = LinearOperator(𝒯₁, 𝒯₁, interval.(M₂))
        @test all(isequal_interval.(coefficients(2.0 * Ai), interval.(2.0 .* M₂)))
        @test all(isequal_interval.(coefficients(rmul!(copy(Ai), 2.0)), interval.(2.0 .* M₂)))

        Ac = LinearOperator(𝒯₁, 𝒯₁, ComplexF64[1.0+1.0im 2.0 ; 3.0 4.0-2.0im])
        @test 2.0 * Ac == Ac * 2.0 == rmul!(copy(Ac), 2.0) == lmul!(2.0, copy(Ac)) ==
            LinearOperator(𝒯₁, 𝒯₁, ComplexF64[2.0+2.0im 4.0 ; 6.0 8.0-4.0im])
    end

    @testset "* between LinearOperators and lazy composition" begin
        M₃ = [1.0 2.0 3.0 ; 4.0 5.0 6.0 ; 7.0 8.0 9.0]
        E = LinearOperator(𝒯₂, 𝒯₂, M₃)

        # the contracted dimension is truncated to domain(A) ∩ codomain(E), of order 1
        AE = M₂ * M₃[1:2, :]
        @test coefficients(A * E) == AE
        @test domain(A * E) == 𝒯₂ && codomain(A * E) == 𝒯₁
        @test A * E == mul!(similar(zeros(𝒯₂, 𝒯₁)), A, E, true, false)
        @test coefficients(E * A) == M₃[:, 1:2] * M₂

        C = A ∘ E
        @test C isa ComposedOperator
        @test project(C, domain(E), codomain(A)) == A * E

        # `mul!` into an output smaller than the product truncates it, here to its first two columns
        E₂₃ = LinearOperator(𝒯₂, 𝒯₁, M₂₃)
        for (X, C₀, α, β) ∈ ((E, Inf, 1.0, 0.0), (E, 50.0, 1.0, 2.0), (E₂₃, Inf, 1.0, 0.0), (E₂₃, 100.0, 2.0, 3.0))
            C = mul!(fill(C₀, 𝒯₁, 𝒯₁), A, X, α, β)
            @test coefficients(C) == (β == 0 ? 0.0 : β * C₀) .+ α .* AE[:, 1:2]
        end
    end

    @testset "^, inv, / and \\" begin
        @test A^0 == LinearOperator(𝒯₁, 𝒯₁, I₂) == one(A)
        @test A^1 == A
        @test A^2 == A * A == LinearOperator(𝒯₁, 𝒯₁, M₂ * M₂)
        @test A^3 == A * A * A == LinearOperator(𝒯₁, 𝒯₁, M₂ * M₂ * M₂)

        @test coefficients(inv(A)) ≈ inv(M₂)
        @test A^(-1) == inv(A)
        @test coefficients(A^(-2)) ≈ inv(M₂) * inv(M₂)

        # a literal exponent goes through `Base.literal_pow`, only a variable one reaches `^`
        for (n, expected) ∈ ((-2, inv(M₂) * inv(M₂)), (4, M₂ * M₂ * M₂ * M₂))
            @test coefficients(A^n) ≈ expected
        end

        @test coefficients(A / B) ≈ M₂ / N₂
        @test coefficients(A \ B) ≈ M₂ \ N₂
    end

    @testset "A ± I on sequence spaces" begin
        # the sum with a `UniformScaling` is lazy, its projection agrees with the in-place forms
        @test project(A + I, 𝒯₁, 𝒯₁) == project(I + A, 𝒯₁, 𝒯₁) ==
            radd!(copy(A), I) == ladd!(I, copy(A)) == LinearOperator(𝒯₁, 𝒯₁, M₂ + I₂)
        @test project(A - I, 𝒯₁, 𝒯₁) == rsub!(copy(A), I) == LinearOperator(𝒯₁, 𝒯₁, M₂ - I₂)
        @test project(I - A, 𝒯₁, 𝒯₁) == lsub!(I, copy(A)) == LinearOperator(𝒯₁, 𝒯₁, I₂ - M₂)

        @test radd!(copy(A), UniformScalingOperator(1.0)) == radd!(copy(A), I)
        @test radd!(copy(A), UniformScalingOperator(2.0)) == LinearOperator(𝒯₁, 𝒯₁, M₂ + 2.0 * I₂)
        @test radd!(LinearOperator(𝕂, 𝕂, fill(3.0, 1, 1)), I) == LinearOperator(𝕂, 𝕂, fill(4.0, 1, 1))

        # `I` is added on the indices shared by the domain and the codomain
        D = LinearOperator(𝒯₁, 𝒯₂, [1.0 2.0 ; 3.0 4.0 ; 5.0 6.0])
        @test coefficients(radd!(copy(D), I)) == [2.0 2.0 ; 3.0 5.0 ; 5.0 6.0]

        # between different symmetries `I` is the orthogonal projection
        Dsym = LinearOperator(Fourier(1, 1.0), evensym(Fourier(1, 1.0)), zeros(2, 3))
        @test coefficients(radd!(copy(Dsym), I)) == [0.0 1.0 0.0 ; 0.5 0.0 0.5]

        Dmix = LinearOperator(𝒯₁, Chebyshev(1), M₂)
        @test_throws ArgumentError radd!(copy(Dmix), I)
        @test_throws ArgumentError lsub!(I, copy(Dmix))
    end

    @testset "A ± I materializes on spaces built solely out of ScalarSpace" begin
        @testset "$s" for s ∈ (𝕂^2, 𝕂 × 𝕂)
            Aₛ = LinearOperator(s, s, copy(M₂))
            @test Aₛ + I == I + Aₛ == LinearOperator(s, s, M₂ + I₂)
            @test Aₛ - I == LinearOperator(s, s, M₂ - I₂)
            @test I - Aₛ == LinearOperator(s, s, I₂ - M₂)
            @test Aₛ + 2I == LinearOperator(s, s, M₂ + 2.0 * I₂)
        end

        As = LinearOperator(𝕂, 𝕂, fill(3.0, 1, 1))
        @test As + I == LinearOperator(𝕂, 𝕂, fill(4.0, 1, 1))
        @test I - As == LinearOperator(𝕂, 𝕂, fill(-2.0, 1, 1))

        M₃, I₃ = Float64[1 2 3 ; 4 5 6 ; 7 8 9], Matrix{Float64}(I, 3, 3)
        s = 𝕂^2 × 𝕂 # nested cartesian spaces
        C = LinearOperator(s, s, copy(M₃))
        @test C + I == LinearOperator(s, s, M₃ + I₃)
        @test I - C == LinearOperator(s, s, I₃ - M₃)

        # a `SequenceSpace` factor has a truncated tail, so the sum stays lazy
        smix = 𝕂 × 𝒯₁
        M = LinearOperator(smix, smix, copy(M₃))
        @test M + I isa Add
        @test project(M + I, smix, smix) == LinearOperator(smix, smix, M₃ + I₃)
        @test project(I - M, smix, smix) == LinearOperator(smix, smix, I₃ - M₃)

        @test_throws ArgumentError LinearOperator(𝕂^2, 𝕂^3, ones(3, 2)) + I

        Aint = LinearOperator(𝕂^2, 𝕂^2, [1 2 ; 3 4])
        @test eltype(Aint + I) == Int
        @test eltype(Aint - 2.5I) == Float64
    end

    @testset "add!, sub! and their one-sided forms" begin
        J = UniformScalingOperator(5.0)
        out() = fill(Inf, 𝒯₁, 𝒯₁)

        # an operand that is not a `LinearOperator` is projected onto the spaces of the output
        @test coefficients(add!(out(), J, A)) == coefficients(add!(out(), A, J)) == M₂ + 5.0 * I₂
        @test coefficients(radd!(copy(A), J)) == coefficients(ladd!(J, copy(A))) == M₂ + 5.0 * I₂

        # A shares the spaces of the output, B has a larger domain
        B₂₃ = LinearOperator(𝒯₂, 𝒯₁, M₂₃)
        @test coefficients(add!(out(), A, B₂₃)) == M₂ .+ M₂₃[:, 1:2]
        @test coefficients(sub!(out(), A, B₂₃)) == M₂ .- M₂₃[:, 1:2]

        # no two of A, B and the output share both their domain and codomain
        A₀ = LinearOperator(𝒯₀, 𝒯₀, fill(10.0, 1, 1))
        B₀ = LinearOperator(𝒯₁, 𝒯₀, [1.0 2.0])
        @test coefficients(add!(fill(Inf, 𝒯₂, 𝒯₀), A₀, B₀)) == [11.0 2.0 0.0]
        @test coefficients(sub!(fill(Inf, 𝒯₂, 𝒯₀), A₀, B₀)) == [9.0 -2.0 0.0]

        @test coefficients(lsub!(copy(A), copy(B))) == M₂ .- N₂
    end

    @testset "mul!(C, A, B, α, β)" begin
        @test mul!(fill(Inf, 𝒯₁, 𝒯₁), A, B, true, false) == A * B == LinearOperator(𝒯₁, 𝒯₁, M₂ * N₂)
        @test coefficients(mul!(fill(100.0, 𝒯₁, 𝒯₁), A, B, 2.0, 3.0)) == 2.0 .* (M₂ * N₂) .+ 300.0

        # operands that are not `LinearOperator`s are projected onto the spaces of the output
        @test coefficients(mul!(fill(Inf, 𝒯₁, 𝒯₁), UniformScalingOperator(2.0), UniformScalingOperator(3.0), 1.0, 0.0)) == 6.0 .* I₂
        @test coefficients(mul!(fill(Inf, 𝒯₁, 𝒯₁), UniformScalingOperator(3.0), B, 1.0, 0.0)) == 3.0 .* N₂
    end

    @testset "cartesian-block operator arithmetic" begin
        𝒯₀², 𝒯₁², 𝒯₂² = 𝒯₀^2, 𝒯₁^2, 𝒯₂^2
        I₄ = Matrix{Float64}(I, 4, 4)
        Z₂₂, Z₂₃ = zeros(2, 2), zeros(2, 3)
        # block diagonal operators whose blocks have different orders
        Ablk = LinearOperator(𝒯₁², 𝒯₁², [M₂ Z₂₂ ; Z₂₂ M₂])
        Bblk = LinearOperator(𝒯₂², 𝒯₁², [M₂₃ Z₂₃ ; Z₂₃ M₂₃])
        S₂₃ = [2.0 4.0 3.0 ; 7.0 9.0 6.0] # the sum of a Taylor(1) and a Taylor(2) block

        @testset "matching block structure" begin
            M₄ = [1.0 2.0 5.0 6.0 ; 3.0 4.0 7.0 8.0 ; 9.0 10.0 13.0 14.0 ; 11.0 12.0 15.0 16.0]
            A₄ = LinearOperator(𝒯₁², 𝒯₁², M₄)
            B₄ = LinearOperator(𝒯₁², 𝒯₁², I₄)

            @test coefficients(A₄ + B₄) == M₄ .+ I₄
            @test A₄ + B₄ == A₄ - (-B₄) == ladd!(copy(A₄), copy(B₄))
            @test coefficients(radd!(copy(A₄), B₄)) == M₄ .+ I₄
            @test coefficients(rsub!(copy(A₄), B₄)) == M₄ .- I₄
            @test coefficients(A₄ * B₄) == M₄
            @test mul!(fill(Inf, 𝒯₁², 𝒯₁²), A₄, B₄, true, false) == A₄ * B₄

            @test coefficients(component(A₄, 1, 1)) == M₄[1:2, 1:2]
            @test coefficients(component(A₄, 2, 2)) == M₄[3:4, 3:4]
        end

        @testset "mismatched suborders adapt per block" begin
            @test coefficients(Ablk + Bblk) == [S₂₃ Z₂₃ ; Z₂₃ S₂₃]
            @test coefficients(component(Ablk + Bblk, 1, 1)) == S₂₃
            @test all(iszero, coefficients(component(Ablk + Bblk, 1, 2)))
            @test coefficients(ladd!(copy(Ablk), copy(Bblk))) == [S₂₃ Z₂₃ ; Z₂₃ S₂₃]
            D₂₃ = [M₂ zeros(2, 1)] .- M₂₃
            @test coefficients(lsub!(copy(Ablk), copy(Bblk))) == [D₂₃ Z₂₃ ; Z₂₃ D₂₃]

            @test coefficients(Ablk * Bblk) == [M₂*M₂₃ Z₂₃ ; Z₂₃ M₂*M₂₃]

            # `mul!` into a smaller output truncates each block of the product to its first two columns
            P = M₂ * M₂₃[:, 1:2]
            @test coefficients(mul!(fill(Inf, 𝒯₁², 𝒯₁²), Ablk, Bblk, 1.0, 0.0)) == [P Z₂₂ ; Z₂₂ P]
            @test coefficients(mul!(fill(10.0, 𝒯₁², 𝒯₁²), Ablk, Bblk, 3.0, 2.0)) == 20.0 .+ 3.0 .* [P Z₂₂ ; Z₂₂ P]
        end

        @testset "products mixing plain and cartesian spaces" begin
            Bfull = LinearOperator(𝒯₁², 𝒯₁², Float64[1 0 2 0 ; 0 1 0 2 ; 3 0 1 0 ; 0 3 0 1])
            G = LinearOperator(𝒯₁², 𝕂, Float64[1 2 3 4])
            K = LinearOperator(𝕂, 𝒯₁², reshape(Float64[2, 3, 5, 7], 4, 1))
            H = LinearOperator(𝕂, 𝕂, fill(3.0, 1, 1))
            for (X, Y) ∈ ((H, G), (G, K), (K, H), (G, Bfull), (Bfull, K), (K, G), (Bfull, Bfull))
                @test coefficients(X * Y) == coefficients(X) * coefficients(Y)
                @test mul!(fill(Inf, domain(Y), codomain(X)), X, Y, true, false) == X * Y
            end
        end

        @testset "products mixing plain and cartesian spaces, assembled block by block" begin
            col, row = reshape([10.0, 20.0, 30.0, 40.0], 4, 1), [1.0 2.0 3.0 4.0]
            # the output has a space of a different order than the operand it would otherwise match,
            # only the coefficients of order 0 of the product are then written
            for (X, Y, dom, codom, expected) ∈ (
                    (LinearOperator(𝒯₀, 𝒯₁², reshape(row, 4, 1)), LinearOperator(𝒯₁², 𝒯₀, reshape(col, 1, 4)), 𝒯₁², 𝒯₀²,
                        [10.0 20.0 30.0 40.0 ; 30.0 60.0 90.0 120.0]),
                    (LinearOperator(𝒯₁², 𝒯₀, row), LinearOperator(𝒯₁², 𝒯₁², I₄), 𝒯₀², 𝒯₀, [1.0 3.0]),
                    (LinearOperator(𝒯₀, 𝒯₀, fill(5.0, 1, 1)), LinearOperator(𝒯₁², 𝒯₀, reshape(col, 1, 4)), 𝒯₀², 𝒯₀, [50.0 150.0]),
                    (LinearOperator(𝒯₁², 𝒯₀², [1.0 2.0 3.0 4.0 ; 5.0 6.0 7.0 8.0]), LinearOperator(𝒯₀, 𝒯₁², col), 𝒯₁, 𝒯₀²,
                        [300.0 0.0 ; 700.0 0.0]),
                    (LinearOperator(𝒯₀, 𝒯₁², reshape(row, 4, 1)), LinearOperator(𝒯₀, 𝒯₀, fill(10.0, 1, 1)), 𝒯₁, 𝒯₁²,
                        [10.0 0.0 ; 20.0 0.0 ; 30.0 0.0 ; 40.0 0.0]),
                    (LinearOperator(𝒯₁², 𝒯₀, row), LinearOperator(𝒯₀, 𝒯₁², col), 𝒯₁, 𝒯₀, [300.0 0.0]),
                    (LinearOperator(𝒯₀², 𝒯₀, [2.0 5.0]), LinearOperator(𝒯₀, 𝒯₁², col), 𝒯₀, 𝒯₀, fill(2.0 * 10.0 + 5.0 * 30.0, 1, 1)))
                @test coefficients(mul!(fill(Inf, dom, codom), X, Y, 1.0, 0.0)) == expected
                C₀ = reshape(collect(1.0:length(expected)), size(expected))
                @test coefficients(mul!(LinearOperator(dom, codom, copy(C₀)), X, Y, 1.0, 2.0)) == 2.0 .* C₀ .+ expected
            end
        end

        @testset "Diagonal of scalings, one per leaf block" begin
            M₄ = Float64[1 0 2 0 ; 0 1 0 2 ; 3 0 1 0 ; 0 3 0 1]
            D = RadiiPolynomial.LinearAlgebra.Diagonal
            @test coefficients(radd!(LinearOperator(𝒯₁², 𝒯₁², copy(M₄)), D([2.0 * I, 3.0 * I]))) == M₄ + D([2.0, 2.0, 3.0, 3.0])

            # the first block of the domain is itself cartesian, hence 3 leaf blocks
            dom = 𝒯₁² × 𝒯₀
            J = D([UniformScalingOperator(2.0), UniformScalingOperator(3.0), UniformScalingOperator(5.0)])
            @test coefficients(radd!(LinearOperator(dom, dom, Matrix{Float64}(I, 5, 5)), J)) == D([3.0, 3.0, 4.0, 4.0, 6.0])
        end
    end

    @testset "interval and complex coefficient promotion" begin
        Ai = interval(A)
        Bi = LinearOperator(𝒯₁, 𝒯₁, interval.(N₂))

        @test eltype(Ai + Bi) == Interval{Float64}
        @test all(isequal_interval.(coefficients(Ai + Bi), interval.(M₂ .+ N₂)))
        @test all(in_interval.(M₂ * N₂, coefficients(Ai * Bi)))
        @test all(in_interval.(M₂ * N₂, coefficients(mul!(fill(interval(1e10), 𝒯₁, 𝒯₁), Ai, Bi, true, false))))

        Ac = LinearOperator(𝒯₁, 𝒯₁, ComplexF64[1.0+1.0im 2.0 ; 3.0 4.0-2.0im])
        Bc = LinearOperator(𝒯₁, 𝒯₁, ComplexF64.(I₂))
        @test coefficients(Ac + Bc) == ComplexF64[2.0+1.0im 2.0 ; 3.0 5.0-2.0im]
        @test coefficients(Ac * Bc) == coefficients(Ac)

        Sci = LinearOperator(𝒯₁, 𝒯₁, complex.(interval.(M₂))) + Ai
        @test eltype(Sci) == Complex{Interval{Float64}}
        @test all(isequal_interval.(real.(coefficients(Sci)), interval.(2.0 .* M₂)))
        @test all(isequal_interval.(imag.(coefficients(Sci)), interval(0.0)))
    end

    @testset "incompatible spaces" begin
        𝒞 = Chebyshev(1)
        Bc = LinearOperator(𝒞, 𝒞, M₂)
        @test_throws ArgumentError A * Bc
        @test_throws ArgumentError A / Bc
        @test_throws ArgumentError A \ Bc
        @test_throws ArgumentError add!(zeros(𝒞, 𝒞), A, A)
        @test_throws ArgumentError sub!(zeros(𝒞, 𝒞), A, A)

        # cartesian spaces with a different number of blocks
        A2 = LinearOperator(𝒯₁^2, 𝒯₁^2, Matrix{Float64}(I, 4, 4))
        A3 = LinearOperator(𝒯₁^3, 𝒯₁^3, Matrix{Float64}(I, 6, 6))
        @test_throws ArgumentError A2 + A3
    end

    @testset "exact zeros read through a symmetry" begin
        # the cocycle factor of a zero coefficient is skipped: multiplied by it, an exact zero would be
        # a `Float64`, which cannot be stored as an `ExactReal`
        E, E₀ = evensym(Fourier(2, exact(1.0))), evensym(Fourier(0, exact(1.0)))
        Z = zeros(Complex{ExactReal{Float64}}, E₀, 𝕂)
        F = LinearOperator(E, 𝕂, [1.0+0im 2.0 3.0])
        @test Z + F == F
        @test F + Z == F
    end

    @testset "the sum allocates the type of the sum of the entries" begin
        Bb = LinearOperator(𝒯₁, 𝒯₁, Bool[1 0 ; 0 1])
        @test Bb + Bb == LinearOperator(𝒯₁, 𝒯₁, [2 0 ; 0 2])
        @test eltype(Bb + Bb) == Int
    end
end
