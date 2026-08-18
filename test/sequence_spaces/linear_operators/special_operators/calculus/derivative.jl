@testset "Derivative" begin

    @testset "Constructors and algebra" begin
        @test order(Derivative(1)) == 1
        @test order(Derivative(1, 2)) == (1, 2) == order(Derivative((1, 2)))
        @test Derivative(1, 2) == Derivative((1, 2))
        @test Derivative(1) * Derivative(1) == Derivative(1)^2 == Derivative(2)
        @test Derivative(2) * Derivative(3) == Derivative(5)
        @test Derivative((1, 2)) * Derivative((3, 1)) == Derivative((4, 3))
        @test Derivative(2)^3 == Derivative(6)
        @test Derivative(1)^0 == Derivative(0)
        @test Derivative((1, 2))^3 == Derivative((3, 6))
        @test Derivative((1, 2))^(2, 3) == Derivative((2, 6))
        # guards: only non-negative orders, and at least one order in a tuple
        @test_throws DomainError Derivative(-1)
        @test_throws DomainError Derivative((1, -1))
        @test_throws ArgumentError Derivative(())
        @test_throws ArgumentError Derivative()
    end

    @testset "Taylor" begin
        ∂¹ = Derivative(1)
        ∂⁴ = Derivative(4)

        # domain/codomain: ∂ⁿ maps Taylor(N) to Taylor(max(0,N-n)); domain is the inverse map (via Integral)
        @test codomain(∂¹, Taylor(3)) == Taylor(2)
        @test codomain(∂⁴, Taylor(3)) == Taylor(0) # order too high ⇒ truncated to the constant space
        @test domain(Derivative(2), Taylor(3)) == Taylor(5) == codomain(Integral(2), Taylor(3))
        @test domain(Derivative(0), Taylor(3)) == Taylor(3)
        @test codomain(Derivative(0), Taylor(3)) == Taylor(3)

        # n = 0: identity, coefficient type preserved (unlike Fourier)
        a_int = Sequence(Taylor(2), [1, 2, 3])
        c_int = differentiate(a_int, 0)
        @test eltype(c_int) == Int
        @test c_int == a_int

        # n = 1: d/dx(1 - x + x²) = -1 + 2x
        a_𝒯 = Sequence(Taylor(2), [1.0, -1.0, 1.0])
        @test ∂¹(a_𝒯) == project(∂¹, Taylor(2), Taylor(1), Float64)(a_𝒯) ==
            differentiate!(Sequence(Taylor(1), [Inf, Inf]), a_𝒯) ==
            mul!(Sequence(Taylor(1), [Inf, Inf]), ∂¹, a_𝒯) == Sequence(Taylor(1), [-1.0, 2.0])

        # order too high (n > order(a)): result is the zero constant
        @test ∂⁴(a_𝒯) == project(∂⁴, Taylor(2), Taylor(0), Float64)(a_𝒯) ==
            differentiate!(Sequence(Taylor(0), [Inf]), a_𝒯, 4) ==
            mul!(Sequence(Taylor(0), [Inf]), ∂⁴, a_𝒯) == Sequence(Taylor(0), [0.0])

        # composition of two first-order derivatives: d/dx(-1+2x) = 2
        @test ∂¹(∂¹(a_𝒯)) == Sequence(Taylor(0), [2.0])

        # spaces must match in the in-place forms
        @test_throws ArgumentError differentiate!(Sequence(Taylor(5), zeros(6)), a_𝒯)

        a_𝒯c = Sequence(Taylor(2), ComplexF64[1 + 1im, -1.0, 2im])
        @test ∂¹(a_𝒯c) == Sequence(Taylor(1), ComplexF64[-1.0, 4im])

        a_𝒯i = Sequence(Taylor(2), interval.([1.0, -1.0, 1.0]))
        @test ∂¹(a_𝒯i) == Sequence(Taylor(1), interval.([-1.0, 2.0]))

        @testset "order ≥ 2" begin
            a2 = Sequence(Taylor(3), [1.0, 2.0, 3.0, 4.0]) # 1+2x+3x²+4x³ ⇒ d²/dx² = 6+24x
            expected = Sequence(Taylor(1), [6.0, 24.0])
            @test differentiate(a2, 2) == differentiate!(Sequence(Taylor(1), [Inf, Inf]), a2, 2) ==
                mul!(Sequence(Taylor(1), [Inf, Inf]), Derivative(2), a2) ==
                project(Derivative(2), Taylor(3), Taylor(1), Float64)(a2) == expected
        end
    end

    @testset "Fourier" begin
        # domain/codomain are always the same Fourier space, for every order
        @test domain(Derivative(3), Fourier(2, 1.0)) == Fourier(2, 1.0)
        @test codomain(Derivative(3), Fourier(2, 1.0)) == Fourier(2, 1.0)

        # (Dⁿa)_j = (iωj)ⁿ a_j; iⁿ cycles with period 4: 1, i, -1, -i
        ω = 1.5
        a_ℱ = Sequence(Fourier(2, ω), ComplexF64[1.0, 2.0, 3.0, -2.0, 0.5]) # j = -2,-1,0,1,2

        # n = 0: identity, but coefficient type is always promoted to Complex
        a_real = Sequence(Fourier(2, ω), [1.0, 2.0, 3.0, -2.0, 0.5])
        c0 = differentiate(a_real, 0)
        @test eltype(c0) == ComplexF64
        @test c0 == a_ℱ

        ∂¹, ∂², ∂³, ∂⁴ = Derivative(1), Derivative(2), Derivative(3), Derivative(4)

        @test ∂¹(a_ℱ) == project(∂¹, Fourier(2, ω), Fourier(2, ω), ComplexF64)(a_ℱ) ==
            differentiate!(Sequence(Fourier(2, ω), fill(complex(Inf), 5)), a_ℱ) ==
            mul!(Sequence(Fourier(2, ω), fill(complex(Inf), 5)), ∂¹, a_ℱ) ==
            Sequence(Fourier(2, ω), ComplexF64[-3.0im, -3.0im, 0.0, -3.0im, 1.5im])

        @test ∂²(a_ℱ) == differentiate!(Sequence(Fourier(2, ω), fill(complex(Inf), 5)), a_ℱ, 2) ==
            mul!(Sequence(Fourier(2, ω), fill(complex(Inf), 5)), ∂², a_ℱ) ==
            Sequence(Fourier(2, ω), ComplexF64[-9.0, -4.5, 0.0, 4.5, -4.5])

        # n = 3 and n = 4 complete the i^n cycle
        @test ∂³(a_ℱ) == project(∂³, Fourier(2, ω), Fourier(2, ω), ComplexF64)(a_ℱ) ==
            Sequence(Fourier(2, ω), ComplexF64[27.0im, 6.75im, 0.0, 6.75im, -13.5im])

        @test ∂⁴(a_ℱ) == project(∂⁴, Fourier(2, ω), Fourier(2, ω), ComplexF64)(a_ℱ) ==
            Sequence(Fourier(2, ω), ComplexF64[81.0, 10.125, 0.0, -10.125, 40.5])

        @test ∂¹(∂¹(a_ℱ)) == ∂²(a_ℱ)

        @test_throws ArgumentError differentiate!(Sequence(Fourier(1, ω), fill(complex(Inf), 3)), a_ℱ)

        # 1/3 is not exactly representable, hence the enclosure checks
        x = interval(1) / interval(3)
        a_ℱi = Sequence(Fourier(1, 1.0), [complex(x), complex(interval(0.0)), complex(x)])
        d_ℱi = ∂¹(a_ℱi)
        @test in_interval(1/3, imag(d_ℱi[1]))
        @test in_interval(-1/3, imag(d_ℱi[-1]))
        @test isequal_interval(real(d_ℱi[0]), interval(0.0)) & isequal_interval(imag(d_ℱi[0]), interval(0.0))
    end

    @testset "Chebyshev" begin
        ∂¹ = Derivative(1)

        # only the order-0 derivative has a well-defined domain on a Chebyshev space
        @test domain(∂¹, Chebyshev(3)) == UndefSpace()
        @test domain(Derivative(0), Chebyshev(3)) == Chebyshev(3)
        @test codomain(∂¹, Chebyshev(3)) == Chebyshev(2)
        @test codomain(Derivative(2), Chebyshev(3)) == Chebyshev(1)

        # n = 0: identity, coefficient type preserved
        a_int = Sequence(Chebyshev(2), [1, 2, 3])
        c_int = differentiate(a_int, 0)
        @test eltype(c_int) == Int
        @test c_int == a_int

        # n = 1: c[i] = 2 Σ_{j=i+1, step 2}^{ord} j a[j]; a = T0+2T1+3T2+4T3
        a_𝒞 = Sequence(Chebyshev(3), [1.0, 2.0, 3.0, 4.0])
        @test ∂¹(a_𝒞) == project(∂¹, Chebyshev(3), Chebyshev(2), Float64)(a_𝒞) ==
            differentiate!(Sequence(Chebyshev(2), [Inf, Inf, Inf]), a_𝒞) ==
            mul!(Sequence(Chebyshev(2), [Inf, Inf, Inf]), ∂¹, a_𝒞) == Sequence(Chebyshev(2), [28.0, 12.0, 24.0])

        # T₀ + T₁/2 + T₂/2
        a_𝒞2 = Sequence(Chebyshev(2), [1.0, 0.5, 0.5])
        @test ∂¹(a_𝒞2) == Sequence(Chebyshev(1), [1.0, 2.0])

        @test_throws ArgumentError differentiate!(Sequence(Chebyshev(5), zeros(6)), a_𝒞)

        a_𝒞c = Sequence(Chebyshev(3), ComplexF64[1 + 1im, 2.0, 3.0, 4im])
        @test ∂¹(a_𝒞c) == Sequence(Chebyshev(2), ComplexF64[4 + 24im, 12.0, 24.0im])
        a_𝒞i = Sequence(Chebyshev(2), interval.([1.0, 0.5, 0.5]))
        @test ∂¹(a_𝒞i) == Sequence(Chebyshev(1), interval.([1.0, 2.0]))

        # the order-0 derivative materializes to the identity
        @test project(Derivative(0), Chebyshev(3), Chebyshev(3), Float64)(a_𝒞) == a_𝒞

        # order(a) < n = 1 ⇒ the lone coefficient is the zero constant
        a_𝒞0 = Sequence(Chebyshev(0), [5.0])
        @test differentiate(a_𝒞0, 1) == Sequence(Chebyshev(0), [0.0])

        # derivatives of order ≥ 2 are not supported on a Chebyshev space
        @test_throws DomainError differentiate(a_𝒞2, 2)

        # the same restriction applies when materializing the operator
        @test_throws DomainError project(Derivative(2), Chebyshev(3), Chebyshev(1), Float64)
    end

    @testset "Tensor space" begin
        # all orders = 1: ∂ₓ∂ᵧ∂_z of a monomial x^i e^{iωjy} T_k(z)
        # only survives terms with i=1 (Taylor), j≠0 (Fourier, else the mode vanishes) and k=1 (Chebyshev)
        s = Taylor(1) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(1)
        a_𝑇 = Sequence(s, collect(1.0:12.0))
        ∂₁₁₁ = Derivative((1, 1, 1))
        c_𝑇 = differentiate(a_𝑇, (1, 1, 1))
        @test project(∂₁₁₁, s, codomain(∂₁₁₁, s), ComplexF64) * a_𝑇 == c_𝑇 ==
            Sequence(Taylor(0) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(0), ComplexF64[-16.0im, 0.0, 24.0im])

        # identity (all orders zero)
        @test differentiate(a_𝑇, (0, 0, 0)) == Sequence(s, ComplexF64.(collect(1.0:12.0)))

        # coefficient type promotes to Complex as soon as a Fourier factor is present
        b1 = Sequence(Taylor(1) ⊗ Taylor(1), collect(1.0:4.0))
        @test eltype(differentiate(b1, (1, 1))) == Float64
        b2 = Sequence(Taylor(1) ⊗ Fourier(1, 1.0), collect(1.0:6.0))
        @test eltype(differentiate(b2, (1, 1))) == ComplexF64

        # a Chebyshev factor with a nonzero order propagates UndefSpace through the whole tensor domain
        s2 = Taylor(1) ⊗ Chebyshev(2)
        @test domain(Derivative((0, 1)), s2) == UndefSpace()
        @test codomain(Derivative((0, 1)), s2) == Taylor(1) ⊗ Chebyshev(1)

        a_𝑇i = Sequence(Taylor(1) ⊗ Taylor(1), interval.(collect(1.0:4.0)))
        @test differentiate(a_𝑇i, (1, 1)) == Sequence(Taylor(0) ⊗ Taylor(0), interval.([4.0]))

        # with no undefined factor, the domain is the tensor product of the per-factor domains
        @test domain(Derivative((1, 2)), Taylor(1) ⊗ Taylor(2)) == Taylor(2) ⊗ Taylor(4)

        @testset "non-first factor uses the array-returning `_apply`, not the in-place `_apply!`" begin
            # the first factor of a tensor product is differentiated in place, every other
            # factor through a separate path

            # a non-first Taylor factor of order 0 is copied through
            b1t = Sequence(Taylor(1) ⊗ Taylor(2), collect(1.0:6.0))
            @test differentiate(b1t, (1, 0)) == Sequence(Taylor(0) ⊗ Taylor(2), [2.0, 4.0, 6.0])

            # a non-first Taylor factor whose own order is below the derivative order gives zero
            b2t = Sequence(Taylor(1) ⊗ Taylor(0), collect(1.0:2.0))
            @test differentiate(b2t, (1, 1)) == Sequence(Taylor(0) ⊗ Taylor(0), [0.0])

            # Fourier as the first factor, n = 1 then odd n ≥ 3, on the same j = -2,...,2 values
            ω = 1.5
            vals = ComplexF64[1.0, 2.0, 3.0, -2.0, 0.5]
            sF1 = Fourier(2, ω) ⊗ Taylor(0)
            @test differentiate(Sequence(sF1, vals), (1, 0)) ==
                Sequence(sF1, ComplexF64[-3.0im, -3.0im, 0.0, -3.0im, 1.5im])
            @test differentiate(Sequence(sF1, vals), (3, 0)) ==
                Sequence(sF1, ComplexF64[27.0im, 6.75im, 0.0, 6.75im, -13.5im])

            # Fourier as a non-first factor, odd n ≥ 3
            sF2 = Taylor(0) ⊗ Fourier(2, ω)
            @test differentiate(Sequence(sF2, vals), (0, 3)) ==
                Sequence(sF2, ComplexF64[27.0im, 6.75im, 0.0, 6.75im, -13.5im])

            # a non-first Chebyshev factor whose own order is below the derivative order gives zero
            sC1 = Taylor(0) ⊗ Chebyshev(0)
            @test differentiate(Sequence(sC1, [3.0]), (0, 1)) == Sequence(sC1, [0.0])
        end

        @testset "Taylor order ≥ 2 through tensor recursion" begin
            # Taylor order ≥ 2 as the first factor
            sT1 = Taylor(3) ⊗ Fourier(1, 1.0)
            aT1 = Sequence(sT1, collect(1.0:12.0))
            D1 = Derivative((2, 0))
            codom1 = codomain(D1, sT1)
            expectedT1 = project(D1, sT1, codom1, ComplexF64)(aT1) # cross-check via materialization
            @test differentiate(aT1, (2, 0)) ==
                mul!(Sequence(codom1, fill(complex(Inf), dimension(codom1))), D1, aT1) ==
                expectedT1 == Sequence(codom1, ComplexF64[6.0, 24.0, 14.0, 48.0, 22.0, 72.0])

            # Taylor order ≥ 2 as a non-first factor
            sT2 = Fourier(1, 1.0) ⊗ Taylor(3)
            aT2 = Sequence(sT2, collect(1.0:12.0))
            D2 = Derivative((0, 2))
            codom2 = codomain(D2, sT2)
            expectedT2 = project(D2, sT2, codom2, ComplexF64)(aT2)
            @test differentiate(aT2, (0, 2)) == expectedT2 ==
                Sequence(codom2, ComplexF64[14.0, 16.0, 18.0, 60.0, 66.0, 72.0])

            # Chebyshev order ≥ 2 as a non-first factor is not supported either
            sC2 = Taylor(0) ⊗ Chebyshev(3)
            aC2 = Sequence(sC2, collect(1.0:4.0))
            @test_throws DomainError differentiate(aC2, (0, 2))
        end
    end

    @testset "Cartesian space" begin
        # Derivative applies the same operator independently to each component, whatever its base space
        a = Sequence(Taylor(2)^2 × Fourier(1, 1.0) × Chebyshev(2), collect(1.0:6.0+3.0+3.0))
        ∂¹ = Derivative(1)
        c = ∂¹(a)
        @test c == differentiate!(Sequence(Taylor(1)^2 × Fourier(1, 1.0) × Chebyshev(1), fill(complex(Inf), 2*2+3+2)), a) ==
            mul!(Sequence(Taylor(1)^2 × Fourier(1, 1.0) × Chebyshev(1), fill(complex(Inf), 2*2+3+2)), ∂¹, a) ==
            Sequence(Taylor(1)^2 × Fourier(1, 1.0) × Chebyshev(1),
                ComplexF64[2.0, 6.0, 5.0, 12.0, -7.0im, 0.0, 9.0im, 22.0, 48.0])
        @test component(c, 1) == differentiate(component(a, 1))
        @test component(c, 3) == differentiate(component(a, 3))

        # the domain and codomain adapt component by component
        @test domain(∂¹, Taylor(2)^2) == Taylor(3)^2
        @test codomain(∂¹, Taylor(2)^2) == Taylor(1)^2
        @test codomain(∂¹, Taylor(2) × Fourier(1, 1.0)) == Taylor(1) × Fourier(1, 1.0)
    end

    @testset "Symmetric space" begin
        # only Fourier symmetries carry a derivative
        sE = evensym(Fourier(2, 1.0))
        sO = oddsym(Fourier(2, 1.0))
        ∂¹ = Derivative(1)

        # ∂ maps evensym Fourier to oddsym Fourier: the group generator's amplitude picks up a
        # factor (-1)^order(∂) since the lattice automorphism matrix is [-1] for both symmetries
        @test codomain(∂¹, sE) == sO
        @test domain(∂¹, sO) == sE
        @test codomain(Derivative(2), sE) == sE # (-1)² = 1: order-2 derivative preserves the symmetry

        a = Sequence(sE, ComplexF64[1.0, 2.0, 3.0]) # a₀=1, a₁=a₋₁=2, a₂=a₋₂=3 (even ⇒ cosine-like)
        out = Sequence(sO, fill(complex(Inf), dimension(sO)))
        expected = Sequence(sO, ComplexF64[2.0im, 6.0im]) # (iω·1)a₁ = 2i, (iω·2)a₂ = 6i
        @test ∂¹(a) == project(∂¹, sE, sO, ComplexF64)(a) == differentiate!(out, a) ==
            mul!(Sequence(sO, fill(complex(Inf), dimension(sO))), ∂¹, a) == expected

        # ∂∘∂ == ∂² and maps back into evensym
        c2 = differentiate(expected) # oddsym → evensym
        @test space(c2) == sE
        @test c2 == differentiate(a, 2)
    end

    @testset "InfiniteSequence error propagation" begin
        X = Ell1(GeometricWeight(2.0))

        # Taylor derivative: finite max_{α≤k≤N} k!/(k-α)! ν^{-k}, tail sup_{k>N}, total sup_{k≥α}
        c = InfiniteSequence(Sequence(Taylor(2), [1.0, 1.0, 1.0]), 1.0, 1.0, 2.0, X)
        Dc = differentiate(c)
        @test finite_error(Dc) == 0.5
        @test tail_error(Dc) == 0.375
        @test total_error(Dc) == min(0.5 + 0.375, 0.5 * 2.0)
        @test banachspace(Dc) == Ell1() # the differentiated sequence always carries the unweighted norm

        # α = 0 is the identity factor for every base space
        @test RadiiPolynomial._derivative_finite_error(X, Taylor(3), 0) == 1.0
        @test RadiiPolynomial._derivative_tail_error(X, Taylor(3), 0) == 1.0
        @test RadiiPolynomial._derivative_total_error(X, Taylor(3), 0) == 1.0

        # Fourier derivative: finite/tail/total factors are |ω|·(a sup of j ν^{-j}), no extra factor 2
        @test RadiiPolynomial._derivative_finite_error(X, Fourier(3, 1.0), 1) == 0.5
        @test RadiiPolynomial._derivative_tail_error(X, Fourier(3, 1.0), 1) == 0.25
        @test RadiiPolynomial._derivative_total_error(X, Fourier(3, 1.0), 1) == 0.5

        f = InfiniteSequence(Sequence(Fourier(3, 1.0), [0, 0, 0.5im, 0, -0.5im, 0, 0]), 0.5, 0.3, 0.9, X)
        Df = differentiate(f)
        @test sequence(Df) == differentiate(sequence(f))
        @test finite_error(Df) == 0.5 * 0.5
        @test tail_error(Df) == 0.25 * 0.3
        @test total_error(Df) == min(0.5 * 0.5 + 0.25 * 0.3, 0.5 * 0.9)

        # guaranteed-zero errors skip the factors: error-free Fourier differentiation works
        b = InfiniteSequence(Sequence(Fourier(1, 1.0), [0.5im, 0.0, -0.5im]), X)
        Db = differentiate(b)
        @test sequence(Db) == differentiate(sequence(b))
        @test iszero(total_error(Db))

        # tensor tail factor covers the box complement: max_i tailᵢ · ∏_{l≠i} totalₗ
        X² = Ell1((GeometricWeight(2.0), GeometricWeight(2.0)))
        s² = Taylor(2) ⊗ Taylor(2)
        @test RadiiPolynomial._derivative_finite_error(X², s², (1, 1)) == 0.5 * 0.5
        @test RadiiPolynomial._derivative_tail_error(X², s², (1, 1)) == 0.375 * 0.5
        @test RadiiPolynomial._derivative_total_error(X², s², (1, 1)) == 0.5 * 0.5

        a² = InfiniteSequence(Sequence(s², ones(9)), 0.0, 1.0, 1.0, X²)
        Da² = differentiate(a², (1, 1))
        @test tail_error(Da²) == 0.375 * 0.5
        @test sequence(Da²) == Sequence(Taylor(1) ⊗ Taylor(1), [1.0, 2.0, 2.0, 4.0]) # ∂ₓ∂ᵧ Σxⁱyʲ, i,j∈0:2

        # guarded errors: Fourier only supports α ≤ 1
        @test_throws DomainError RadiiPolynomial._derivative_tail_error(X, Fourier(3, 1.0), 2)

        # every Taylor/Fourier operator separates supports, so the cross column vanishes and the
        # triangular scheme collapses to the diagonal one
        @test RadiiPolynomial._derivative_cross_error(X, Taylor(2), 1) == 0.0
        @test RadiiPolynomial._derivative_cross_error(X, Fourier(3, 1.0), 1) == 0.0
        @test RadiiPolynomial._derivative_cross_error(Ell1((GeometricWeight(2.0), GeometricWeight(2.0))),
                                                      Taylor(2) ⊗ Taylor(2), (1, 1)) == 0.0

        @testset "Chebyshev: triangular propagation" begin
            #= D is upper triangular in the halved convention, (Du)_i = Σ_{j>i, j-i odd} 2j u_j, so a
               head column j ≤ N reaches only rows i ≤ N-1 = N' while a tail column j > N reaches head
               rows too. Support separation fails in the tail → finite direction only, so four
               separate constants are needed. Each is a restricted column norm, and is checked
               here against directly measured columns. =#
            ν = 2.0
            Xc = Ell1(GeometricWeight(ν))
            for N ∈ (1, 3, 8)
                s = Chebyshev(N)
                # measure ‖Π• D e_j‖_{Ell1()} / w(j) directly, w(0) = 1, w(k) = 2νᵏ
                big, J = Chebyshev(80), 80
                col(j) = begin
                    e = Sequence(big, zeros(Float64, J+1)); e[j] = 1.0
                    img = differentiate(e)
                    head = project(img, Chebyshev(max(0, N-1)))
                    w = norm(e, Xc)
                    (norm(head, Ell1()) / w, (norm(img, Ell1()) - norm(head, Ell1())) / w, norm(img, Ell1()) / w)
                end
                cols = [col(j) for j ∈ 0:J-2]
                @test RadiiPolynomial._derivative_finite_error(Xc, s, 1) ≈ maximum(c[1] for c ∈ cols[1:N+1])   rtol=1e-12
                @test RadiiPolynomial._derivative_cross_error(Xc, s, 1)  ≈ maximum(c[1] for c ∈ cols[N+2:end]) rtol=1e-12
                @test RadiiPolynomial._derivative_tail_error(Xc, s, 1)   ≈ maximum(c[2] for c ∈ cols[N+2:end]) rtol=1e-12
                @test RadiiPolynomial._derivative_total_error(Xc, s, 1)  ≈ maximum(c[3] for c ∈ cols)          rtol=1e-12
            end

            # the head and tail row counts of a tail column add up to the full column: b_N(j) + 2⌈(j-N)/2⌉ = j
            @test all(RadiiPolynomial._cheb_b(N, j) + 2*cld(j-N, 2) == j for N ∈ 1:12, j ∈ 13:40 if j > N)

            # the finite error of the image now sees the tail error of the input
            a_cheb = InfiniteSequence(Sequence(Chebyshev(3), [1.0, 0.5, 0.2, 0.05]), 0.5, 0.25, 0.75, Xc)
            Da = differentiate(a_cheb)
            κf = RadiiPolynomial._derivative_finite_error(Xc, Chebyshev(3), 1)
            κc = RadiiPolynomial._derivative_cross_error(Xc, Chebyshev(3), 1)
            κt = RadiiPolynomial._derivative_tail_error(Xc, Chebyshev(3), 1)
            κo = RadiiPolynomial._derivative_total_error(Xc, Chebyshev(3), 1)
            @test finite_error(Da) == κf * 0.5 + κc * 0.25
            @test tail_error(Da) == κt * 0.25
            @test total_error(Da) == min(κo * 0.75, κf * 0.5 + κc * 0.25 + κt * 0.25)
            @test banachspace(Da) == Ell1()

            # α = 0 is the identity; α ≥ 2, ν ≤ 1 and order 0 are out of scope
            @test RadiiPolynomial._derivative_cross_error(Xc, Chebyshev(2), 0) == 0.0
            @test RadiiPolynomial._derivative_tail_error(Xc, Chebyshev(2), 0) == 1.0
            @test_throws DomainError RadiiPolynomial._derivative_finite_error(Xc, Chebyshev(2), 2)
            @test_throws DomainError RadiiPolynomial._derivative_total_error(Ell1(GeometricWeight(1.0)), Chebyshev(2), 1)
            @test_throws DomainError RadiiPolynomial._derivative_cross_error(Xc, Chebyshev(0), 1)

            #= Tensor rule. For a *product* row set the column norm factorizes, the finite rows F_{N'}
               are a product and the columns outside the box F_N are the union of the slabs
               Aᵢ = {j : jᵢ > Nᵢ}, whence
                   κ_cross = max_i ( κ_cross^{(i)} ∏_{l≠i} max(κ_fin^{(l)}, κ_cross^{(l)}) ),
               the inner max being a sup over *all* j_l. Checked against measured columns below. =#
            X2 = Ell1((GeometricWeight(ν), GeometricWeight(ν)))
            for N ∈ (2, 4)
                s1 = Chebyshev(N)
                hfin = map(0:40) do j
                    e = Sequence(Chebyshev(60), zeros(Float64, 61)); e[j] = 1.0
                    norm(project(differentiate(e), Chebyshev(max(0, N-1))), Ell1()) / norm(e, Ell1(GeometricWeight(ν)))
                end
                meas = maximum(hfin[j1+1] * hfin[j2+1] for j1 ∈ 0:40, j2 ∈ 0:40 if j1 > N || j2 > N)
                @test RadiiPolynomial._derivative_cross_error(X2, s1 ⊗ s1, (1, 1)) ≈ meas rtol=1e-12
            end
        end

        # the shared Taylor/Fourier factor requires a genuine geometric rate ν > 1
        @test_throws DomainError RadiiPolynomial._geom_kfact_sup(1.0, 1, 3)

        # f(k) = k!/(k-α)!·ν^{-k} peaks near k* = (να-ν+1)/(ν-1);
        # starting well below k* (here k* ≈ 8) forces several strict ascents (running max updates) before
        # the certified decrease, unlike the α = 1 cases above which peak (or decrease) immediately
        @test RadiiPolynomial._geom_kfact_sup(1.5, 3, 0) == (8*7*6)/1.5^8 # f(8) is the running max (f(9) does not exceed it)
    end

end
