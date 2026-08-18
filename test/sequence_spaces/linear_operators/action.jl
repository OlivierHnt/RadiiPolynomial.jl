@testset "Action" begin

    @testset "call form A(a) == A*a" begin
        @testset "Taylor" begin
            𝒯 = Taylor(1)
            A = LinearOperator(𝒯, 𝒯, [1.0 2.0 ; 3.0 4.0])
            a = Sequence(𝒯, [10.0, 20.0])
            expected = Sequence(𝒯, [1*10.0+2*20.0, 3*10.0+4*20.0]) # [50, 110]
            @test A(a) == A * a == expected
        end

        @testset "Fourier" begin
            ℱ = Fourier(1, 1.5)
            A = LinearOperator(ℱ, ℱ, ComplexF64[1 2 3 ; 4 5 6 ; 7 8 9])
            a = Sequence(ℱ, ComplexF64[1, 2, 3])
            expected = Sequence(ℱ, coefficients(A) * coefficients(a))
            @test A(a) == A * a == expected
        end

        @testset "Chebyshev" begin
            𝒞 = Chebyshev(1)
            A = LinearOperator(𝒞, 𝒞, [1.0 2.0 ; 3.0 4.0])
            a = Sequence(𝒞, [10.0, 20.0])
            @test A(a) == A * a == Sequence(𝒞, [50.0, 110.0])
        end

        @testset "TensorSpace" begin
            𝒮 = Taylor(1) ⊗ Chebyshev(1)
            M = Matrix{Float64}(reshape(1:16, 4, 4)) # column-major
            A = LinearOperator(𝒮, 𝒮, M)
            a = Sequence(𝒮, [1.0, 2.0, 3.0, 4.0])
            # M*[1,2,3,4] = [1,2,3,4]+2[5,6,7,8]+3[9,10,11,12]+4[13,14,15,16] = [90,100,110,120]
            @test A(a) == A * a == Sequence(𝒮, [90.0, 100.0, 110.0, 120.0])
        end

        @testset "SymmetricSpace" begin
            es = evensym(Fourier(1, 1.0))
            A = LinearOperator(es, es, [1.0 2.0 ; 3.0 4.0])
            a = Sequence(es, [10.0, 20.0])
            @test A(a) == A * a == Sequence(es, [50.0, 110.0])
        end

        @testset "ComplexF64 coefficients, real sequence" begin
            𝒯 = Taylor(1)
            A = LinearOperator(𝒯, 𝒯, ComplexF64[1+1im 0 ; 0 1-1im])
            a = Sequence(𝒯, [2.0, 3.0])
            @test A(a) == Sequence(𝒯, ComplexF64[2+2im, 3-3im])
        end
    end

    @testset "action across mismatched-but-compatible spaces (order adaptation)" begin
        𝒯₁ = Taylor(1)
        A = LinearOperator(𝒯₁, 𝒯₁, [1.0 2.0 ; 3.0 4.0])

        # `a` has strictly more coefficients than `domain(A)`: the extra one is discarded
        a_big = Sequence(Taylor(2), [10.0, 20.0, 999.0])
        @test A * a_big == Sequence(𝒯₁, [50.0, 110.0]) # same as using only [10,20]

        # `a` has strictly fewer coefficients than `domain(A)`: the missing one acts as 0
        a_small = Sequence(Taylor(0), [10.0])
        @test A * a_small == Sequence(𝒯₁, [1*10.0, 3*10.0]) # [10, 30]

        # Fourier: matching frequency required, order may differ
        ℱ₁ = Fourier(1, 2.0)
        B = LinearOperator(ℱ₁, ℱ₁, Float64[1 2 3 ; 4 5 6 ; 7 8 9])
        b_big = Sequence(Fourier(2, 2.0), [1.0, 10.0, 20.0, 30.0, 1.0]) # only middle 3 matter
        @test B * b_big == Sequence(ℱ₁, coefficients(B) * [10.0, 20.0, 30.0])
    end

    @testset "mul!(c, A, a, α, β): BLAS-style c = α(A*a) + β*c" begin
        𝒯₁ = Taylor(1)
        A = LinearOperator(𝒯₁, 𝒯₁, [1.0 2.0 ; 3.0 4.0])
        a = Sequence(𝒯₁, [10.0, 20.0]) # A*a = [50, 110]

        @testset "β = 0: c is overwritten" begin
            c = Sequence(𝒯₁, [Inf, Inf])
            @test mul!(c, A, a, 2.0, 0.0) == Sequence(𝒯₁, [100.0, 220.0]) == c
        end

        @testset "β = 1: accumulate onto existing c" begin
            c = Sequence(𝒯₁, [1000.0, 2000.0])
            @test mul!(c, A, a, 1.0, 1.0) == Sequence(𝒯₁, [1050.0, 2110.0]) == c
        end

        @testset "general β: c is rescaled before accumulating" begin
            c = Sequence(𝒯₁, [100.0, 200.0])
            @test mul!(c, A, a, 3.0, 0.5) == Sequence(𝒯₁, 0.5 .* [100.0, 200.0] .+ 3.0 .* [50.0, 110.0]) == c
        end

        @testset "target space c larger than codomain(A): untouched entries follow β only" begin
            c = Sequence(Taylor(2), [1000.0, 1000.0, 7.0])
            mul!(c, A, a, 2.0, 0.5)
            # first two: 0.5*1000 + 2*(A*a) = [500,500] + [100,220] = [600,720]
            # third: β*7 = 3.5 (never touched by the mul, since 2 ∉ indices(codomain(A)))
            @test c == Sequence(Taylor(2), [600.0, 720.0, 3.5])
        end

        @testset "both domain(A) ≠ space(a) and codomain(A) ≠ space(c)" begin
            a3 = Sequence(Taylor(2), [10.0, 20.0, 99.0]) # extra coefficient ignored
            c = Sequence(Taylor(2), [1000.0, 1000.0, 7.0])
            mul!(c, A, a3, 2.0, 0.5)
            @test c == Sequence(Taylor(2), [600.0, 720.0, 3.5]) # identical to the case above
        end

        @testset "β = 0 exactly (rather than a general scaling): untouched entries are zeroed" begin
            # domain(A) == space(a), codomain(A) ≠ space(c)
            c = Sequence(Taylor(2), [999.0, 999.0, 7.0])
            mul!(c, A, a, 2.0, 0.0)
            @test c == Sequence(Taylor(2), [100.0, 220.0, 0.0])

            # domain(A) ≠ space(a3), codomain(A) ≠ space(c)
            a3 = Sequence(Taylor(2), [10.0, 20.0, 99.0])
            c2 = Sequence(Taylor(2), [999.0, 999.0, 7.0])
            mul!(c2, A, a3, 2.0, 0.0)
            @test c2 == Sequence(Taylor(2), [100.0, 220.0, 0.0])
        end
    end

    @testset "cartesian block action" begin
        @testset "both domain and codomain cartesian" begin
            domA = Taylor(1) × Taylor(0) # dims 2,1
            codomA = Taylor(0) × Taylor(1) # dims 1,2
            M = Float64[1 2 3 ; 4 5 6 ; 7 8 9]
            A = LinearOperator(domA, codomA, M)

            # b's first block is larger than the corresponding block of the domain
            b = Sequence(Taylor(2) × Taylor(0), [10.0, 20.0, 30.0, 40.0])
            # effective b (truncated to domA): component1 = [10,20], component2 = [40]
            # codomain component1 (dim1) = A[1,1]*[10,20] + A[1,2]*[40] = (1*10+2*20)+3*40 = 170
            # codomain component2 (dim2) = A[2:3,1:2]*[10,20] + A[2:3,3]*[40]
            #   = [4*10+5*20, 7*10+8*20] + [6,9]*40 = [140,230] + [240,360] = [380,590]
            @test A * b == Sequence(codomA, [170.0, 380.0, 590.0])
            @test A(b) == A * b
        end

        @testset "plain domain, cartesian codomain" begin
            A = LinearOperator(Taylor(1), Taylor(0)^2, [1.0 2.0 ; 3.0 4.0])
            b = Sequence(Taylor(2), [10.0, 20.0, 999.0]) # extra coefficient ignored
            @test A * b == Sequence(Taylor(0)^2, [1*10.0+2*20.0, 3*10.0+4*20.0]) # [50,110]
        end

        @testset "cartesian domain, plain codomain" begin
            A = LinearOperator(Taylor(0)^2, Taylor(1), [1.0 2.0 ; 3.0 4.0])
            # b's blocks are not the same spaces as those of the domain
            b = Sequence(Taylor(1) × Taylor(0), [100.0, 200.0, 300.0])
            # component(A,1) (col 1 = [1,3]) meets component(b,1) = Taylor(1) [100,200]:
            #   only the Taylor(0)-part (100) contributes: [1,3]*100 = [100,300]
            # component(A,2) (col 2 = [2,4]) meets component(b,2) = Taylor(0) [300]:
            #   [2,4]*300 = [600,1200]
            # total: [700, 1500]
            @test A * b == Sequence(Taylor(1), [700.0, 1500.0])
        end

        @testset "both cartesian, mismatched suborders, general β (≠ 0, 1)" begin
            CP1 = CartesianPower(Taylor(1), 2) # dim 4
            Amat = [1.0 2.0 5.0 6.0; 3.0 4.0 7.0 8.0; 9.0 10.0 13.0 14.0; 11.0 12.0 15.0 16.0]
            A = LinearOperator(CP1, CP1, Amat)
            CP2 = CartesianPower(Taylor(2), 2) # mismatched suborder (dim 6)
            b = Sequence(CP2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            c = Sequence(CP1, [100.0, 100.0, 100.0, 100.0])
            mul!(c, A, b, 2.0, 3.0)
            # b truncated per block (Taylor(2) → Taylor(1) drops the order-2 coefficient):
            # block 1 = [1,2], block 2 = [4,5]
            b1, b2 = [1.0, 2.0], [4.0, 5.0]
            A11, A12, A21, A22 = Amat[1:2,1:2], Amat[1:2,3:4], Amat[3:4,1:2], Amat[3:4,3:4]
            expected = 3.0 .* [100.0, 100.0, 100.0, 100.0] .+ 2.0 .* vcat(A11 * b1 .+ A12 * b2, A21 * b1 .+ A22 * b2)
            @test c == Sequence(CP1, expected)
        end

        @testset "plain domain, cartesian codomain: fast path when b's space matches domain(A) exactly" begin
            S = Taylor(0)
            CP = CartesianPower(Taylor(1), 2)
            A = LinearOperator(S, CP, reshape([1.0, 2.0, 3.0, 4.0], 4, 1))
            b = Sequence(S, [10.0])
            @test A * b == Sequence(CP, [10.0, 20.0, 30.0, 40.0])
        end

        @testset "cartesian domain, plain codomain, target space c ≠ codomain(A)" begin
            CP = CartesianPower(Taylor(1), 2) # dim 4
            S = Taylor(0)
            A = LinearOperator(CP, S, [1.0 2.0 3.0 4.0])
            b = Sequence(CP, [10.0, 20.0, 30.0, 40.0]) # matches domain(A) exactly
            dotAB = 1.0 * 10.0 + 2.0 * 20.0 + 3.0 * 30.0 + 4.0 * 40.0 # A ⋅ b = 300

            # β = 0: c is zeroed first, then only its overlapping (order-0) entry is written
            c = Sequence(Taylor(2), [999.0, 999.0, 7.0])
            mul!(c, A, b, 1.0, 0.0)
            @test c == Sequence(Taylor(2), [dotAB, 0.0, 0.0])

            # general β: untouched entries are rescaled instead of zeroed
            c2 = Sequence(Taylor(2), [1.0, 2.0, 3.0])
            mul!(c2, A, b, 1.0, 2.0)
            @test c2 == Sequence(Taylor(2), [2.0 * 1.0 + dotAB, 2.0 * 2.0, 2.0 * 3.0])
        end

        @testset "cartesian domain mismatched vs b, plain codomain, general β" begin
            S = Taylor(0)
            domA = CartesianPower(Taylor(1), 2) # dim 4
            A = LinearOperator(domA, S, [1.0 2.0 3.0 4.0])
            b = Sequence(CartesianPower(Taylor(0), 2), [10.0, 30.0]) # mismatched suborder vs domain(A)
            c = Sequence(S, [5.0])
            mul!(c, A, b, 1.0, 2.0)
            # per block, only the Taylor(0)-index (0) of each Taylor(1) block of A contributes
            val1 = 1.0 * 10.0 # A's block-1 index-0 coefficient times b's block-1 value
            val2 = 3.0 * 30.0 # A's block-2 index-0 coefficient times b's block-2 value
            @test c == Sequence(S, [2.0 * 5.0 + (val1 + val2)])
        end
    end

    @testset "lazy wrapper action" begin
        𝒯₁ = Taylor(1)
        A = LinearOperator(𝒯₁, 𝒯₁, [1.0 2.0 ; 3.0 4.0]) # A*a = [50,110]
        B = LinearOperator(𝒯₁, 𝒯₁, [5.0 6.0 ; 7.0 8.0]) # B*a = [170,230]
        a = Sequence(𝒯₁, [10.0, 20.0])

        @testset "UniformScalingOperator" begin
            J = UniformScalingOperator(2.0)
            @test J(a) == J * a == Sequence(𝒯₁, [20.0, 40.0])
        end

        @testset "UniformScaling (I)" begin
            @test (true * I) * a == a
            @test (2 * I) * a == Sequence(𝒯₁, [20.0, 40.0])
        end

        @testset "Negate" begin
            N = Negate(A)
            @test N(a) == N * a == -(A * a) == Sequence(𝒯₁, [-50.0, -110.0])
        end

        @testset "Add" begin
            S = Add(A, B)
            @test S(a) == S * a == A * a + B * a == Sequence(𝒯₁, [220.0, 340.0])
        end

        @testset "ComposedOperator" begin
            C = ComposedOperator(A, B) # outer = A, inner = B: C*a = A*(B*a)
            @test C(a) == C * a == A * (B * a) == Sequence(𝒯₁, [630.0, 1430.0])
        end

        @testset "generic 4-argument mul! fallback for non-LinearOperator operators" begin
            J = UniformScalingOperator(2.0)
            c = Sequence(𝒯₁, [100.0, 200.0])
            mul!(c, J, a, 3.0, 0.5)
            # 0.5*[100,200] + 3.0*(2.0*a) = [50,100] + [60,120] = [110,220]
            @test c == Sequence(𝒯₁, [110.0, 220.0])

            N = Negate(A)
            c2 = Sequence(𝒯₁, [1000.0, 2000.0])
            mul!(c2, N, a, 1.0, 0.0)
            @test c2 == Sequence(𝒯₁, [-50.0, -110.0])
        end
    end

    @testset "Matrix, Diagonal, and Vector actions" begin
        @testset "Matrix * Sequence (ScalarSpace cartesian block)" begin
            b = Sequence(ScalarSpace()^3, [1.0, 2.0, 3.0])
            M = [1 2 3 ; 4 5 6]
            @test M * b == Sequence(ScalarSpace()^2, [1*1.0+2*2.0+3*3.0, 4*1.0+5*2.0+6*3.0]) # [14,32]
        end

        @testset "Diagonal * Sequence" begin
            b = Sequence(ScalarSpace()^3, [1.0, 2.0, 3.0])
            D = RadiiPolynomial.LinearAlgebra.Diagonal([2.0, 3.0, 4.0])
            @test D * b == Sequence(ScalarSpace()^3, [2.0, 6.0, 12.0])
        end

        @testset "LinearOperator * Vector{Sequence}" begin
            s1 = Sequence(Taylor(1), [1.0, 2.0])
            s2 = Sequence(Taylor(0), [3.0])
            v = [s1, s2]
            A = LinearOperator(Taylor(1) × Taylor(0), Taylor(0), [1.0 2.0 3.0])
            @test A * v == Sequence(Taylor(0), [1*1.0+2*2.0+3*3.0]) # [14]
        end

        @testset "LinearOperator * Vector mixing Sequence and Number" begin
            s1 = Sequence(Taylor(1), [1.0, 2.0])
            v = [s1, 5.0]
            A = LinearOperator(Taylor(1) × ScalarSpace(), Taylor(0), [1.0 2.0 3.0])
            @test A * v == Sequence(Taylor(0), [1*1.0+2*2.0+3*5.0]) # [20]
        end

        @testset "generic AbstractLinearOperator * Vector (non-LinearOperator)" begin
            s1 = Sequence(Taylor(1), [1.0, 2.0])
            v = [s1, 5.0]
            J = UniformScalingOperator(3.0)
            @test J * v == Sequence(Taylor(1) × ScalarSpace(), [3.0, 6.0, 15.0])
        end
    end

    @testset "interval and complex coefficients through mul!" begin
        𝒯₁ = Taylor(1)
        A = LinearOperator(𝒯₁, 𝒯₁, interval.([1.0 2.0 ; 3.0 4.0]))
        a = Sequence(𝒯₁, interval.([10.0, 20.0]))
        c = Sequence(𝒯₁, [interval(1000.0), interval(1000.0)])
        mul!(c, A, a, interval(2.0), interval(0.0))
        # 2*(A*a) = 2*[50,110] = [100,220], both exactly representable
        @test issubset_interval(interval(100.0), c[0])
        @test issubset_interval(interval(220.0), c[1])

        Ac = complex(A)
        ac = complex(a)
        @test Ac * ac == Sequence(𝒯₁, ComplexF64[50.0, 110.0])
    end

end
