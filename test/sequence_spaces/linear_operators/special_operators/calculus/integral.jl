@testset "Integral" begin

    @testset "Constructors and algebra" begin
        _calculus_test_algebra(Integral)
    end

    @testset "Taylor" begin
        ∫¹, ∫² = Integral(1), Integral(2)
        @test domain(∫², Taylor(3)) == Taylor(1)
        @test codomain(∫², Taylor(3)) == Taylor(5)

        # the order 0 still promotes the coefficient type to floating point
        c_int = integrate(Sequence(Taylor(2), [1, 2, 3]), 0)
        @test eltype(c_int) == Float64
        @test c_int == Sequence(Taylor(2), [1.0, 2.0, 3.0])

        # cᵢ₊₁ = aᵢ/(i+1), c₀ = 0
        test_action(∫¹, Sequence(Taylor(1), [2.0, 4.0]), Sequence(Taylor(2), [0.0, 2.0, 2.0]); inplace! = integrate!)
        a = Sequence(Taylor(2), [1.0, -1.0, 1.0])
        test_action(∫¹, a, Sequence(Taylor(3), [0.0, 1.0, -1/2, 1/3]); inplace! = integrate!)
        test_action(∫², a, Sequence(Taylor(4), [0.0, 0.0, 1/2, -1/6, 1/12]); inplace! = (c, a) -> integrate!(c, a, 2))
        @test ∫¹(∫¹(a)) == ∫²(a)
        @test differentiate(∫¹(a)) == a
        @test_throws ArgumentError integrate!(fill(Inf, Taylor(1)), a)

        @test ∫¹(Sequence(Taylor(2), ComplexF64[2 + 2im, -4.0, 6im])) == Sequence(Taylor(3), ComplexF64[0.0, 2 + 2im, -2.0, 2im])
        @test integrate(Sequence(Taylor(1), interval.([2.0, 4.0]))) == Sequence(Taylor(2), interval.([0.0, 2.0, 2.0]))
        # 1/3 is not representable, so that coefficient is checked as an enclosure
        c = ∫¹(interval.(a))
        @test all(isequal_interval.(coefficients(c)[1:3], interval.([0.0, 1.0, -0.5])))
        @test in_interval(1//3, c[3])
    end

    @testset "Fourier" begin
        s = Fourier(1, 1.0)
        ∫¹, ∫², ∫³, ∫⁴ = Integral(1), Integral(2), Integral(3), Integral(4)

        # the mean-free antiderivative is diagonal, the zero mode of the input being checked by the action
        @test domain(∫¹, Fourier(3, 1.0)) == codomain(∫¹, Fourier(3, 1.0)) == Fourier(3, 1.0)
        @test domain(Integral(0), Fourier(3, 1.0)) == Fourier(3, 1.0)

        # a nonzero mean is accepted by the order 0 only
        a_bad = Sequence(s, [0.5, 1.0, 0.5])
        @test integrate(a_bad, 0) == Sequence(s, ComplexF64[0.5, 1.0, 0.5])
        @test_throws DomainError integrate(a_bad)
        @test_throws DomainError ∫¹(a_bad)

        # the antiderivatives of cos are sin, -cos, -sin, cos
        a = Sequence(s, [0.5, 0.0, 0.5])
        cos_, sin_ = Sequence(s, ComplexF64[0.5, 0.0, 0.5]), Sequence(s, [0.5im, 0.0, -0.5im])
        test_action(∫¹, a, sin_; inplace! = integrate!)
        for (∫ⁿ, expected) ∈ ((Integral(0), cos_), (∫², -cos_), (∫³, -sin_), (∫⁴, cos_))
            test_action(∫ⁿ, a, expected)
        end
        @test ∫¹(∫¹(a)) == project(∫¹, s, s, ComplexF64)^2 * a == integrate!(fill(complex(Inf), s), integrate(a, 1)) == -a
        @test ∫³(a) == ∫¹(∫²(a))
        @test ∫⁴(a) == ∫²(∫²(a))
        @test differentiate(∫¹(a)) == a
        @test_throws ArgumentError integrate!(fill(complex(Inf), Fourier(2, 1.0)), a)

        c = integrate(interval.(a))
        @test all(isequal_interval.(real.(coefficients(c)), interval(0.0)))
        @test all(isequal_interval.(imag.(coefficients(c)), interval.([0.5, 0.0, -0.5])))
    end

    @testset "Chebyshev" begin
        ∫¹ = Integral(1)

        # the T₀ row of the antiderivative is dense, so no finite domain exists
        @test domain(∫¹, Chebyshev(3)) == UndefSpace()
        @test domain(Integral(0), Chebyshev(3)) == Chebyshev(3)
        @test codomain(∫¹, Chebyshev(3)) == Chebyshev(4)

        a_int = Sequence(Chebyshev(2), [1, 2, 3])
        @test eltype(integrate(a_int, 0)) == Float64
        @test integrate(a_int, 0) == a_int
        @test project(Integral(0), Chebyshev(2), Chebyshev(2), Float64)(a_int) == Sequence(Chebyshev(2), [1.0, 2.0, 3.0])

        test_action(∫¹, Sequence(Chebyshev(0), [2.0]), Sequence(Chebyshev(1), [2.0, 1.0]); inplace! = integrate!)
        test_action(∫¹, Sequence(Chebyshev(1), [1.0, 3.0]), Sequence(Chebyshev(2), [-0.5, 0.5, 0.75]); inplace! = integrate!)

        # the row 0 of the materialized matrix holds the inexact entries 2/(1-j²), so its product with
        # a sequence only approximates the recursion
        a = Sequence(Chebyshev(2), [1.0, 0.5, 0.5])
        @test project(∫¹, Chebyshev(2), Chebyshev(3), Float64)(a) ≈ mul!(fill(Inf, Chebyshev(3)), ∫¹, a) ==
            ∫¹(a) == integrate!(fill(Inf, Chebyshev(3)), a)
        a3 = Sequence(Chebyshev(3), [1.0, 2.0, 3.0, 4.0])
        @test project(∫¹, Chebyshev(3), Chebyshev(4), Float64)(a3) ≈ ∫¹(a3) == integrate!(fill(Inf, Chebyshev(4)), a3) ==
            mul!(fill(Inf, Chebyshev(4)), ∫¹, a3) == Sequence(Chebyshev(4), [-1.0, -1.0, -0.5, 0.5, 0.5])
        @test differentiate(∫¹(a3)) == a3

        @test_throws DomainError integrate(a, 2)

        @test ∫¹(Sequence(Chebyshev(1), ComplexF64[2 + 2im, 4.0])) == Sequence(Chebyshev(2), ComplexF64[0 + 2im, 1 + 1im, 1.0])
        @test integrate(Sequence(Chebyshev(1), interval.([1.0, 3.0]))) == Sequence(Chebyshev(2), interval.([-0.5, 0.5, 0.75]))
    end

    @testset "Tensor space" begin
        # the materialized operator agrees with the action, the Fourier direction `dim` being mean-free
        function check(s, dim, orders)
            a = Sequence(s, collect(1.0:dimension(s)))
            selectdim(a, dim, 0) .= 0.0
            for α ∈ orders
                ℐ = Integral(α)
                @test project(ℐ, s, codomain(ℐ, s), ComplexF64) * a ≈ integrate(a, α) == ℐ(a)
            end
            return a
        end

        a = check(Taylor(2) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(2), 2, ((1, 1, 1),))
        @test differentiate(integrate(a, (1, 1, 1)), (1, 1, 1)) ≈ a

        # each base space as a first and as a later factor, over the orders 0, 1, odd ≥ 3 and even ≥ 2,
        # with Chebyshev factors of order 0, 1 and 3
        check(Taylor(1) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(1), 2, ((0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 3, 1), (1, 2, 1)))
        check(Fourier(1, 1.0) ⊗ Taylor(1) ⊗ Chebyshev(3), 1, ((0, 2, 1), (1, 0, 0), (3, 0, 0), (2, 0, 0)))
        check(Taylor(1) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(0), 2, ((1, 1, 1),))

        # a nonzero Fourier mean throws in any direction, and the error names the direction at fault
        @test_throws DomainError integrate(Sequence(Taylor(1) ⊗ Fourier(1, 1.0), collect(1.0:6.0)), (0, 1))
        @test_throws DomainError integrate(Sequence(Fourier(1, 1.0) ⊗ Taylor(1), collect(1.0:6.0)), (1, 0))
        err = try integrate(Sequence(Taylor(2) ⊗ Fourier(2, 1.0), ones(ComplexF64, 15)), (0, 1)) catch e; e end
        @test err isa DomainError && occursin("dimension 2", sprint(showerror, err))

        # a middle direction with a Chebyshev factor of order 0 agrees with the base space operator
        s = Taylor(0) ⊗ Chebyshev(0) ⊗ Fourier(2, 1.0)
        a = Sequence(s, [1.0, 2.0, 3.0, 4.0, 5.0])
        r = Integral((0, 1, 0)) * a
        @test space(r) == Taylor(0) ⊗ Chebyshev(1) ⊗ Fourier(2, 1.0)
        for k ∈ -2:2
            r₁ = Integral(1) * Sequence(Chebyshev(0), [a[(0, 0, k)]])
            @test r[(0, 0, k)] == r₁[0] && r[(0, 1, k)] == r₁[1]
        end

        # each direction agrees with the base space operator on a line of coefficients
        b = Sequence(Chebyshev(3) ⊗ Chebyshev(3) ⊗ Chebyshev(3), [sin(1.7i) for i ∈ 1:64])
        for d ∈ 1:3
            r = Integral(ntuple(i -> Int(i == d), 3)) * b
            line = Sequence(Chebyshev(3), [b[ntuple(i -> i == d ? j : 2, 3)] for j ∈ 0:3])
            @test [r[ntuple(i -> i == d ? j : 2, 3)] for j ∈ 0:4] ≈ coefficients(Integral(1) * line)
        end
    end

    @testset "Cartesian space" begin
        a = Sequence(Taylor(2)^2 × Fourier(1, 1.0) × Chebyshev(2), collect(1.0:12.0))
        component(a, 2)[0] = 0.0 # mean-free Fourier component
        ∫¹ = Integral(1)
        c = ∫¹(a)
        codom = Taylor(3)^2 × Fourier(1, 1.0) × Chebyshev(3)
        @test c == integrate!(fill(complex(Inf), codom), a) == mul!(fill(complex(Inf), codom), ∫¹, a)
        @test component(c, 1) == integrate(component(a, 1))
        @test component(c, 3) == integrate(component(a, 3))
    end

    @testset "Symmetric space" begin
        sE, sO = evensym(Fourier(3, 1.0)), oddsym(Fourier(3, 1.0))
        ∫¹ = Integral(1)

        # an odd order swaps the even and odd symmetries, an even order preserves them
        @test codomain(∫¹, sE) == domain(∫¹, sE) == sO
        @test domain(∫¹, sO) == sE
        @test codomain(Integral(2), sE) == domain(Integral(2), sE) == sE
        @test domain(Integral(2), sO) == sO

        # aₖ/(iωk) on the representatives k = 1, 2, 3
        a = Sequence(sE, ComplexF64[0.0, 2.0, 0.0, 3.0])
        expected = Sequence(sO, ComplexF64[-2.0im, 0.0, -1.0im])
        test_action(∫¹, a, expected; inplace! = integrate!)

        # the same computation on the full Fourier space
        full(x) = Sequence(desymmetrize(space(x)), [RadiiPolynomial.getcoefficient(x, (desymmetrize(space(x)), j)) for j ∈ -3:3])
        @test integrate(full(a)) == full(expected)

        @test differentiate(∫¹(a)) == a

        a_bad = Sequence(sE, ComplexF64[1.0, 2.0, 0.0, 3.0])
        @test_throws DomainError integrate(a_bad)
        @test_throws DomainError ∫¹(a_bad)
    end

    @testset "InfiniteSequence error propagation" begin
        factors = (RadiiPolynomial._integral_finite_error, RadiiPolynomial._integral_cross_error,
                   RadiiPolynomial._integral_tail_error, RadiiPolynomial._integral_total_error)
        finite, cross, tail, total = factors
        X = Ell1(GeometricWeight(2.0))
        X² = Ell1((GeometricWeight(2.0), GeometricWeight(2.0)))

        # Taylor: finite and total νᵅ/α!, tail νᵅ (N+1)!/(N+α+1)!
        ∫a = integrate(InfiniteSequence(Sequence(Taylor(2), [1.0, 1.0, 1.0]), 0.5, 0.25, 0.75, X))
        @test finite_error(∫a) == 2.0 * 0.5
        @test tail_error(∫a) == 0.5 * 0.25
        @test total_error(∫a) == min(2.0 * 0.5 + 0.5 * 0.25, 2.0 * 0.75)
        @test banachspace(∫a) == X # the image keeps the norm of the input

        # a sequence without error is integrated without error, even with a weight for which no
        # error factor is defined
        for Y ∈ (X, Ell1(IdentityWeight()))
            b = InfiniteSequence(Sequence(Fourier(1, 1.0), [0.5im, 0.0, -0.5im]), Y)
            @test sequence(integrate(b)) == integrate(sequence(b))
            @test iszero(total_error(integrate(b)))
            @test banachspace(integrate(b)) == Y
        end

        # the tensor tail factor covers the complement of the box: max_i tailᵢ ∏_{l≠i} totalₗ
        s² = Taylor(2) ⊗ Taylor(2)
        @test finite(X², s², (1, 1)) == total(X², s², (1, 1)) == 2.0 * 2.0
        @test tail(X², s², (1, 1)) == 0.5 * 2.0
        @test tail_error(integrate(InfiniteSequence(Sequence(s², ones(9)), 0.0, 1.0, 1.0, X²), (1, 1))) == 1.0

        # Fourier: finite ω⁻ᵅ (0 when N = 0), tail (ω(N+1))⁻ᵅ, total ω⁻ᵅ
        f = InfiniteSequence(Sequence(Fourier(3, 0.5), [0, 0, 0.5im, 0, -0.5im, 0, 0]), 0.25, 1.0, 1.25, X)
        ∫f = integrate(f)
        @test sequence(∫f) == integrate(sequence(f))
        @test finite_error(∫f) == 2.0 * 0.25
        @test tail_error(∫f) == 0.5 * 1.0
        @test total_error(∫f) == min(2.0 * 0.25 + 0.5 * 1.0, 2.0 * 1.25)
        @test tail(X, Fourier(3, 0.5), 2) == 0.25
        @test finite(X, Fourier(0, 0.5), 1) == 0.0
        @test total(X, Fourier(0, 0.5), 1) == inv(0.5)

        # the Taylor and Fourier integrals separate supports, so the cross factor vanishes
        @test cross(X, Taylor(2), 1) == cross(X, Fourier(3, 1.0), 1) == 0.0

        @testset "Chebyshev: triangular propagation" begin
            # The row 0 of the antiderivative normalized at t = -1 is dense, so a tail column reaches
            # that head row. Both norms carry the weight and the head of the image is Chebyshev(N+1).
            for ν ∈ (1.0, 1.5, 2.0), N ∈ (0, 1, 3, 8)
                Xν = Ell1(GeometricWeight(ν))
                cols = _calculus_chebyshev_columns(integrate, Chebyshev(N+1), Xν, Xν)
                _calculus_test_chebyshev_factors(factors, Xν, N, cols)
                # the finite and total factors are attained at j = 0
                @test finite(Xν, Chebyshev(N), 1) == total(Xν, Chebyshev(N), 1) == 1.0 + ν
            end

            a = InfiniteSequence(Sequence(Chebyshev(3), [1.0, 0.5, 0.2, 0.05]), 0.5, 0.25, 0.75, X)
            _calculus_test_triangular(integrate(a), factors, X, Chebyshev(3))
            @test banachspace(integrate(a)) == X

            # ν = 1 is admissible
            @test finite(Ell1(GeometricWeight(1.0)), Chebyshev(2), 1) == 2.0
            @test cross(X, Chebyshev(2), 0) == 0.0
            @test finite(X, Chebyshev(2), 0) == tail(X, Chebyshev(2), 0) == 1.0
            @test_throws DomainError integrate(a, 2)
        end
    end

end
