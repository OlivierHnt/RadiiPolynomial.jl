@testset "Derivative" begin

    @testset "Constructors and algebra" begin
        _calculus_test_algebra(Derivative)
    end

    @testset "Taylor" begin
        ∂¹ = Derivative(1)
        @test codomain(∂¹, Taylor(3)) == Taylor(2)
        @test codomain(Derivative(4), Taylor(3)) == Taylor(0)
        @test domain(Derivative(2), Taylor(3)) == Taylor(5) == codomain(Integral(2), Taylor(3))
        @test domain(Derivative(0), Taylor(3)) == codomain(Derivative(0), Taylor(3)) == Taylor(3)

        # the order 0 keeps the coefficient type
        a_int = Sequence(Taylor(2), [1, 2, 3])
        @test eltype(differentiate(a_int, 0)) == Int
        @test differentiate(a_int, 0) == a_int

        a = Sequence(Taylor(2), [1.0, -1.0, 1.0])
        test_action(∂¹, a, Sequence(Taylor(1), [-1.0, 2.0]); inplace! = differentiate!)
        # an order above the order of the sequence gives the zero constant
        test_action(Derivative(4), a, Sequence(Taylor(0), [0.0]); inplace! = (c, a) -> differentiate!(c, a, 4))
        test_action(Derivative(2), Sequence(Taylor(3), [1.0, 2.0, 3.0, 4.0]), Sequence(Taylor(1), [6.0, 24.0]);
            inplace! = (c, a) -> differentiate!(c, a, 2))
        @test differentiate(Sequence(Taylor(3), [1.0, 2.0, 3.0, 4.0]), 2) == Sequence(Taylor(1), [6.0, 24.0])
        @test ∂¹(∂¹(a)) == Sequence(Taylor(0), [2.0])
        @test_throws ArgumentError differentiate!(zeros(Taylor(5)), a)

        @test ∂¹(Sequence(Taylor(2), ComplexF64[1 + 1im, -1.0, 2im])) == Sequence(Taylor(1), ComplexF64[-1.0, 4im])
        @test ∂¹(interval.(a)) == Sequence(Taylor(1), interval.([-1.0, 2.0]))
    end

    @testset "Fourier" begin
        ω = 1.5
        s = Fourier(2, ω)
        @test domain(Derivative(3), s) == codomain(Derivative(3), s) == s

        # the order 0 still promotes the coefficient type to complex
        a = Sequence(s, ComplexF64[1.0, 2.0, 3.0, -2.0, 0.5])
        c0 = differentiate(Sequence(s, [1.0, 2.0, 3.0, -2.0, 0.5]), 0)
        @test eltype(c0) == ComplexF64
        @test c0 == a

        # (∂ⁿa)ₖ = (iωk)ⁿ aₖ, the four orders covering the cycle of iⁿ
        for (n, expected) ∈ (1 => [-3.0im, -3.0im, 0.0, -3.0im, 1.5im], 2 => [-9.0, -4.5, 0.0, 4.5, -4.5],
                             3 => [27.0im, 6.75im, 0.0, 6.75im, -13.5im], 4 => [81.0, 10.125, 0.0, -10.125, 40.5])
            test_action(Derivative(n), a, Sequence(s, ComplexF64.(expected)); inplace! = (c, a) -> differentiate!(c, a, n))
        end
        @test Derivative(1)(Derivative(1)(a)) == Derivative(2)(a)
        @test_throws ArgumentError differentiate!(fill(complex(Inf), Fourier(1, ω)), a)

        # 1/3 is not representable, so the result is checked as an enclosure
        x = interval(1) / interval(3)
        d = Derivative(1)(Sequence(Fourier(1, 1.0), [complex(x), complex(interval(0.0)), complex(x)]))
        @test in_interval(1/3, imag(d[1]))
        @test in_interval(-1/3, imag(d[-1]))
        @test isequal_interval(real(d[0]), interval(0.0)) & isequal_interval(imag(d[0]), interval(0.0))
    end

    @testset "Chebyshev" begin
        ∂¹ = Derivative(1)
        # only the order 0 has a well-defined domain
        @test domain(∂¹, Chebyshev(3)) == UndefSpace()
        @test domain(Derivative(0), Chebyshev(3)) == Chebyshev(3)
        @test codomain(∂¹, Chebyshev(3)) == Chebyshev(2)
        @test codomain(Derivative(2), Chebyshev(3)) == Chebyshev(1)

        a_int = Sequence(Chebyshev(2), [1, 2, 3])
        @test eltype(differentiate(a_int, 0)) == Int
        @test differentiate(a_int, 0) == a_int

        # cᵢ = 2 Σ j aⱼ over j > i with j - i odd
        a = Sequence(Chebyshev(3), [1.0, 2.0, 3.0, 4.0])
        test_action(∂¹, a, Sequence(Chebyshev(2), [28.0, 12.0, 24.0]); inplace! = differentiate!)
        @test project(Derivative(0), Chebyshev(3), Chebyshev(3), Float64)(a) == a
        @test_throws ArgumentError differentiate!(zeros(Chebyshev(5)), a)

        a2 = Sequence(Chebyshev(2), [1.0, 0.5, 0.5])
        @test ∂¹(a2) == Sequence(Chebyshev(1), [1.0, 2.0])
        @test ∂¹(interval.(a2)) == Sequence(Chebyshev(1), interval.([1.0, 2.0]))
        @test ∂¹(Sequence(Chebyshev(3), ComplexF64[1 + 1im, 2.0, 3.0, 4im])) == Sequence(Chebyshev(2), ComplexF64[4 + 24im, 12.0, 24.0im])
        @test differentiate(Sequence(Chebyshev(0), [5.0]), 1) == Sequence(Chebyshev(0), [0.0])

        # orders ≥ 2 are not supported, for the action and for the materialized operator
        @test_throws DomainError differentiate(a2, 2)
        @test_throws DomainError project(Derivative(2), Chebyshev(3), Chebyshev(1), Float64)
    end

    @testset "Tensor space" begin
        # ∂ₓ∂ᵧ∂_z keeps the modes with a Taylor index 1, a nonzero Fourier index and a Chebyshev index 1
        s = Taylor(1) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(1)
        a = Sequence(s, collect(1.0:12.0))
        ∂₁₁₁ = Derivative((1, 1, 1))
        @test project(∂₁₁₁, s, codomain(∂₁₁₁, s), ComplexF64) * a == differentiate(a, (1, 1, 1)) ==
            Sequence(Taylor(0) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(0), ComplexF64[-16.0im, 0.0, 24.0im])
        @test differentiate(a, (0, 0, 0)) == Sequence(s, ComplexF64.(collect(1.0:12.0)))

        # a Fourier factor promotes the coefficient type to complex
        @test eltype(differentiate(Sequence(Taylor(1) ⊗ Taylor(1), collect(1.0:4.0)), (1, 1))) == Float64
        @test eltype(differentiate(Sequence(Taylor(1) ⊗ Fourier(1, 1.0), collect(1.0:6.0)), (1, 1))) == ComplexF64

        # a Chebyshev factor of nonzero order makes the whole domain undefined
        @test domain(Derivative((0, 1)), Taylor(1) ⊗ Chebyshev(2)) == UndefSpace()
        @test codomain(Derivative((0, 1)), Taylor(1) ⊗ Chebyshev(2)) == Taylor(1) ⊗ Chebyshev(1)
        @test domain(Derivative((1, 2)), Taylor(1) ⊗ Taylor(2)) == Taylor(2) ⊗ Taylor(4)

        @test differentiate(Sequence(Taylor(1) ⊗ Taylor(1), interval.(collect(1.0:4.0))), (1, 1)) ==
            Sequence(Taylor(0) ⊗ Taylor(0), interval.([4.0]))

        @testset "each base space as a first and as a later factor" begin
            # a Taylor factor of order 0 is copied through
            @test differentiate(Sequence(Taylor(1) ⊗ Taylor(2), collect(1.0:6.0)), (1, 0)) ==
                Sequence(Taylor(0) ⊗ Taylor(2), [2.0, 4.0, 6.0])
            # a factor whose order is below the derivative order gives zero
            @test differentiate(Sequence(Taylor(1) ⊗ Taylor(0), collect(1.0:2.0)), (1, 1)) == Sequence(Taylor(0) ⊗ Taylor(0), [0.0])
            @test differentiate(Sequence(Taylor(0) ⊗ Chebyshev(0), [3.0]), (0, 1)) == Sequence(Taylor(0) ⊗ Chebyshev(0), [0.0])

            ω = 1.5
            vals = ComplexF64[1.0, 2.0, 3.0, -2.0, 0.5]
            ∂¹vals, ∂³vals = ComplexF64[-3.0im, -3.0im, 0.0, -3.0im, 1.5im], ComplexF64[27.0im, 6.75im, 0.0, 6.75im, -13.5im]
            s₁, s₂ = Fourier(2, ω) ⊗ Taylor(0), Taylor(0) ⊗ Fourier(2, ω)
            @test differentiate(Sequence(s₁, vals), (1, 0)) == Sequence(s₁, ∂¹vals)
            @test differentiate(Sequence(s₁, vals), (3, 0)) == Sequence(s₁, ∂³vals)
            @test differentiate(Sequence(s₂, vals), (0, 3)) == Sequence(s₂, ∂³vals)

            # Taylor order 2 as the first factor
            sT₁ = Taylor(3) ⊗ Fourier(1, 1.0)
            aT₁ = Sequence(sT₁, collect(1.0:12.0))
            test_action(Derivative((2, 0)), aT₁,
                Sequence(codomain(Derivative((2, 0)), sT₁), ComplexF64[6.0, 24.0, 14.0, 48.0, 22.0, 72.0]))
            @test differentiate(aT₁, (2, 0)) == Derivative((2, 0))(aT₁)

            # Taylor order 2 as a later factor
            sT₂ = Fourier(1, 1.0) ⊗ Taylor(3)
            aT₂ = Sequence(sT₂, collect(1.0:12.0))
            test_action(Derivative((0, 2)), aT₂,
                Sequence(codomain(Derivative((0, 2)), sT₂), ComplexF64[14.0, 16.0, 18.0, 60.0, 66.0, 72.0]))

            @test_throws DomainError differentiate(Sequence(Taylor(0) ⊗ Chebyshev(3), collect(1.0:4.0)), (0, 2))
        end
    end

    @testset "Cartesian space" begin
        # each component is differentiated on its own, whatever its space
        a = Sequence(Taylor(2)^2 × Fourier(1, 1.0) × Chebyshev(2), collect(1.0:12.0))
        ∂¹ = Derivative(1)
        c = ∂¹(a)
        test_action(∂¹, a, Sequence(Taylor(1)^2 × Fourier(1, 1.0) × Chebyshev(1),
            ComplexF64[2.0, 6.0, 5.0, 12.0, -7.0im, 0.0, 9.0im, 22.0, 48.0]); inplace! = differentiate!)
        @test component(c, 1) == differentiate(component(a, 1))
        @test component(c, 3) == differentiate(component(a, 3))

        @test domain(∂¹, Taylor(2)^2) == Taylor(3)^2
        @test codomain(∂¹, Taylor(2)^2) == Taylor(1)^2
        @test codomain(∂¹, Taylor(2) × Fourier(1, 1.0)) == Taylor(1) × Fourier(1, 1.0)
    end

    @testset "Symmetric space" begin
        sE, sO = evensym(Fourier(2, 1.0)), oddsym(Fourier(2, 1.0))
        ∂¹ = Derivative(1)

        # an odd order swaps the even and odd symmetries, an even order preserves them
        @test codomain(∂¹, sE) == sO
        @test domain(∂¹, sO) == sE
        @test codomain(Derivative(2), sE) == sE

        # (iωk)aₖ on the representatives k = 1, 2
        a = Sequence(sE, ComplexF64[1.0, 2.0, 3.0])
        expected = Sequence(sO, ComplexF64[2.0im, 6.0im])
        test_action(∂¹, a, expected; inplace! = differentiate!)

        c2 = differentiate(expected)
        @test space(c2) == sE
        @test c2 == differentiate(a, 2)
    end

    @testset "InfiniteSequence error propagation" begin
        factors = (RadiiPolynomial._derivative_finite_error, RadiiPolynomial._derivative_cross_error,
                   RadiiPolynomial._derivative_tail_error, RadiiPolynomial._derivative_total_error)
        finite, cross, tail, total = factors
        X = Ell1(GeometricWeight(2.0))
        X² = Ell1((GeometricWeight(2.0), GeometricWeight(2.0)))

        # Taylor: finite max_{α≤k≤N} k!/(k-α)! ν⁻ᵏ, tail sup_{k>N}, total sup_{k≥α}
        Dc = differentiate(InfiniteSequence(Sequence(Taylor(2), [1.0, 1.0, 1.0]), 1.0, 1.0, 2.0, X))
        @test finite_error(Dc) == 0.5
        @test tail_error(Dc) == 0.375
        @test total_error(Dc) == min(0.5 + 0.375, 0.5 * 2.0)
        @test banachspace(Dc) == Ell1() # the image carries the unweighted norm

        # the order 0 is the identity
        @test finite(X, Taylor(3), 0) == tail(X, Taylor(3), 0) == total(X, Taylor(3), 0) == 1.0

        # Fourier: the factors are |ω| times a sup of k ν⁻ᵏ
        @test finite(X, Fourier(3, 1.0), 1) == 0.5
        @test tail(X, Fourier(3, 1.0), 1) == 0.25
        @test total(X, Fourier(3, 1.0), 1) == 0.5
        @test_throws DomainError tail(X, Fourier(3, 1.0), 2)

        f = InfiniteSequence(Sequence(Fourier(3, 1.0), [0, 0, 0.5im, 0, -0.5im, 0, 0]), 0.5, 0.3, 0.9, X)
        Df = differentiate(f)
        @test sequence(Df) == differentiate(sequence(f))
        @test finite_error(Df) == 0.5 * 0.5
        @test tail_error(Df) == 0.25 * 0.3
        @test total_error(Df) == min(0.5 * 0.5 + 0.25 * 0.3, 0.5 * 0.9)

        # a sequence without error is differentiated without error
        b = InfiniteSequence(Sequence(Fourier(1, 1.0), [0.5im, 0.0, -0.5im]), X)
        @test sequence(differentiate(b)) == differentiate(sequence(b))
        @test iszero(total_error(differentiate(b)))

        # the tensor tail factor covers the complement of the box: max_i tailᵢ ∏_{l≠i} totalₗ
        s² = Taylor(2) ⊗ Taylor(2)
        @test finite(X², s², (1, 1)) == 0.5 * 0.5
        @test tail(X², s², (1, 1)) == 0.375 * 0.5
        @test total(X², s², (1, 1)) == 0.5 * 0.5
        Da² = differentiate(InfiniteSequence(Sequence(s², ones(9)), 0.0, 1.0, 1.0, X²), (1, 1))
        @test tail_error(Da²) == 0.375 * 0.5
        @test sequence(Da²) == Sequence(Taylor(1) ⊗ Taylor(1), [1.0, 2.0, 2.0, 4.0])

        # the Taylor and Fourier derivatives separate supports, so the cross factor vanishes
        @test cross(X, Taylor(2), 1) == cross(X, Fourier(3, 1.0), 1) == cross(X², s², (1, 1)) == 0.0

        @testset "Chebyshev: triangular propagation" begin
            # (Du)ᵢ = Σ 2j uⱼ over j > i with j - i odd, so a tail column reaches the head rows. The
            # image is measured unweighted and its head is Chebyshev(max(0, N-1)).
            for N ∈ (0, 1, 3, 8)
                cols = _calculus_chebyshev_columns(differentiate, Chebyshev(max(0, N-1)), X, Ell1())
                _calculus_test_chebyshev_factors(factors, X, N, cols)
            end

            # a tail column j > N reaches b_N(j) head rows and 2⌈(j-N)/2⌉ tail rows, j in total
            @test all(RadiiPolynomial._cheb_b(N, j) + 2*cld(j-N, 2) == j for N ∈ 1:12, j ∈ 1:40 if j > N)
            # at N = 0 the head is the row 0, reached exactly by the odd columns
            @test all(RadiiPolynomial._cheb_b(0, j) == isodd(j) for j ∈ 1:40)

            a = InfiniteSequence(Sequence(Chebyshev(3), [1.0, 0.5, 0.2, 0.05]), 0.5, 0.25, 0.75, X)
            _calculus_test_triangular(differentiate(a), factors, X, Chebyshev(3))
            @test banachspace(differentiate(a)) == Ell1()

            @test cross(X, Chebyshev(2), 0) == 0.0
            @test finite(X, Chebyshev(2), 0) == tail(X, Chebyshev(2), 0) == 1.0
            # at ν = 1 the columns are unbounded
            @test_throws DomainError total(Ell1(GeometricWeight(1.0)), Chebyshev(2), 1)
            @test_throws DomainError differentiate(a, 2)

            # tensor rule: κ_cross = max_i κ_crossⁱ ∏_{l≠i} max(κ_finˡ, κ_crossˡ), against the measured
            # products of head column norms over the columns outside the box
            for N ∈ (2, 4)
                h = first.(_calculus_chebyshev_columns(differentiate, Chebyshev(max(0, N-1)), X, Ell1(); J = 42))
                measured = maximum(h[j₁+1] * h[j₂+1] for j₁ ∈ 0:40, j₂ ∈ 0:40 if j₁ > N || j₂ > N)
                @test cross(X², Chebyshev(N) ⊗ Chebyshev(N), (1, 1)) ≈ measured rtol=1e-12
            end
        end

        # the Taylor and Fourier factors require a geometric rate ν > 1
        @test_throws DomainError RadiiPolynomial._geom_kfact_sup(1.0, 1, 3)
        # k!/(k-α)! ν⁻ᵏ peaks at k = 8 here, well above the starting index, so the search must ascend
        @test RadiiPolynomial._geom_kfact_sup(1.5, 3, 0) == (8*7*6)/1.5^8
    end

end
