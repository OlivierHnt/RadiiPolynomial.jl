@testset "Integral" begin

    # no condition: the coefficient of the constants is zero
    E₀ = Evaluation(nothing)
    # one condition per dimension, `nothing` along the dimensions that are not integrated
    conditions(α) = map(n -> n == 0 ? nothing : E₀, α)

    @testset "Constructors" begin
        @test Integral() == Integral(E₀)
        @test RadiiPolynomial.condition(Integral()) == E₀
        @test RadiiPolynomial.condition(Integral(Evaluation(-1.0))) == Evaluation(-1.0)
        @test Integral(Evaluation(-1.0), nothing) == Integral((Evaluation(-1.0), nothing))
        @test RadiiPolynomial.condition(Integral(Evaluation(-1.0), nothing)) == (Evaluation(-1.0), nothing)

        # the order of integration: 1 along the dimensions with a condition, 0 along the other ones
        @test order(Integral()) == order(Integral(Evaluation(-1.0))) == 1
        @test order(Integral(nothing)) == 0
        @test order(Integral(Evaluation(-1.0), nothing, E₀)) == (1, 0, 1)

        @test_throws ArgumentError Integral(())
        # one evaluation per dimension
        @test_throws MethodError Integral(Evaluation(1.0, 2.0))
    end

    @testset "Taylor" begin
        ∫ = Integral()
        @test domain(∫, Taylor(3)) == Taylor(2)
        @test codomain(∫, Taylor(3)) == Taylor(4)

        # the coefficient type is promoted to floating point
        @test eltype(integrate(Sequence(Taylor(2), [1, 2, 3]))) == Float64

        # cᵢ₊₁ = aᵢ/(i+1), c₀ = 0
        test_action(∫, Sequence(Taylor(1), [2.0, 4.0]), Sequence(Taylor(2), [0.0, 2.0, 2.0]); inplace! = integrate!)
        a = Sequence(Taylor(2), [1.0, -1.0, 1.0])
        test_action(∫, a, Sequence(Taylor(3), [0.0, 1.0, -1/2, 1/3]); inplace! = integrate!)
        @test ∫(∫(a)) == Sequence(Taylor(4), [0.0, 0.0, 1/2, -1/6, 1/12])
        @test differentiate(∫(a)) == a
        @test_throws ArgumentError integrate!(fill(Inf, Taylor(1)), a)

        @test ∫(Sequence(Taylor(2), ComplexF64[2 + 2im, -4.0, 6im])) == Sequence(Taylor(3), ComplexF64[0.0, 2 + 2im, -2.0, 2im])
        @test integrate(Sequence(Taylor(1), interval.([2.0, 4.0]))) == Sequence(Taylor(2), interval.([0.0, 2.0, 2.0]))
        # 1/3 is not representable, so that coefficient is checked as an enclosure
        c = ∫(interval.(a))
        @test all(isequal_interval.(coefficients(c)[1:3], interval.([0.0, 1.0, -0.5])))
        @test in_interval(1//3, c[3])
    end

    @testset "Fourier" begin
        s = Fourier(1, 1.0)
        ∫ = Integral()

        # the mean-free antiderivative is diagonal, and it sends the zero mode of the input to zero
        @test domain(∫, Fourier(3, 1.0)) == codomain(∫, Fourier(3, 1.0)) == Fourier(3, 1.0)

        a_mean = Sequence(s, [0.5, 1.0, 0.5])
        @test integrate(a_mean) == ∫(a_mean) == integrate(Sequence(s, [0.5, 0.0, 0.5]))
        @test ∫(a_mean) == project(∫, s, s, ComplexF64) * a_mean
        @test differentiate(∫(a_mean)) == a_mean - 1

        # the antiderivatives of cos are sin and -cos
        a = Sequence(s, [0.5, 0.0, 0.5])
        sin_ = Sequence(s, [0.5im, 0.0, -0.5im])
        test_action(∫, a, sin_; inplace! = integrate!)
        @test ∫(∫(a)) == project(∫, s, s, ComplexF64)^2 * a == -a
        @test differentiate(∫(a)) == a
        @test_throws ArgumentError integrate!(fill(complex(Inf), Fourier(2, 1.0)), a)

        c = integrate(interval.(a))
        @test all(isequal_interval.(real.(coefficients(c)), interval(0.0)))
        @test all(isequal_interval.(imag.(coefficients(c)), interval.([0.5, 0.0, -0.5])))
    end

    @testset "Chebyshev" begin
        ∫ = Integral()

        # the row i reads the columns i-1 and i+1: the rows of Chebyshev(3) are exact on that domain
        @test domain(∫, Chebyshev(3)) == Chebyshev(4)
        b = Sequence(Chebyshev(7), collect(1.0:8.0))
        @test (Projection(Chebyshev(3)) * ∫) * b ≈ project(∫(b), Chebyshev(3))
        @test codomain(∫, Chebyshev(3)) == Chebyshev(4)

        @test eltype(integrate(Sequence(Chebyshev(2), [1, 2, 3]))) == Float64

        # cᵢ = (aᵢ₋₁ - aᵢ₊₁)/(2i), c₀ = 0
        test_action(∫, Sequence(Chebyshev(0), [2.0]), Sequence(Chebyshev(1), [0.0, 1.0]); inplace! = integrate!)
        test_action(∫, Sequence(Chebyshev(1), [1.0, 3.0]), Sequence(Chebyshev(2), [0.0, 0.5, 0.75]); inplace! = integrate!)
        a3 = Sequence(Chebyshev(3), [1.0, 2.0, 3.0, 4.0])
        test_action(∫, a3, Sequence(Chebyshev(4), [0.0, -1.0, -0.5, 0.5, 0.5]); inplace! = integrate!)
        @test differentiate(∫(a3)) == a3

        # each integration sets the coefficient of T₀ to zero, and writes on T₁
        a = Sequence(Chebyshev(2), [1.0, 0.5, 0.5])
        @test ∫(∫(a))[0] == 0.0 && ∫(∫(a))[1] != 0.0
        @test differentiate(∫(∫(a3)), 2) ≈ a3

        @test ∫(Sequence(Chebyshev(1), ComplexF64[2 + 2im, 4.0])) == Sequence(Chebyshev(2), ComplexF64[0.0, 1 + 1im, 1.0])
        @test integrate(Sequence(Chebyshev(1), interval.([1.0, 3.0]))) == Sequence(Chebyshev(2), interval.([0.0, 0.5, 0.75]))
    end

    @testset "No integration" begin
        ∫⁰ = Integral(nothing)
        for s ∈ (Taylor(2), Fourier(1, 1.0), Chebyshev(2))
            a = Sequence(s, [0.5, 1.0, 0.5])
            @test domain(∫⁰, s) == codomain(∫⁰, s) == s
            test_action(∫⁰, a, a; inplace! = (c, a) -> integrate!(c, a, nothing))
        end
        # the coefficient type is kept
        @test eltype(integrate(Sequence(Taylor(2), [1, 2, 3]), nothing)) == Int
    end

    @testset "Tensor space" begin
        # the materialized operator agrees with the action, the Fourier direction `dim` being mean-free
        function check(s, dim, orders)
            a = Sequence(s, collect(1.0:dimension(s)))
            selectdim(a, dim, 0) .= 0.0
            for α ∈ orders
                ℐ = Integral(conditions(α))
                @test order(ℐ) == α
                @test project(ℐ, s, codomain(ℐ, s), ComplexF64) * a ≈ integrate(a, conditions(α)) == ℐ(a)
                @test differentiate(ℐ(a), α) ≈ a
            end
            return a
        end

        # each base space as a first and as a later factor, integrated or not, with Chebyshev factors
        # of order 0, 1, 2 and 3
        check(Taylor(2) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(2), 2, ((1, 1, 1), (0, 0, 0)))
        check(Taylor(1) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(1), 2, ((0, 1, 1), (1, 0, 1), (1, 1, 0), (0, 0, 1)))
        check(Fourier(1, 1.0) ⊗ Taylor(1) ⊗ Chebyshev(3), 1, ((0, 1, 1), (1, 0, 0), (1, 1, 0)))
        check(Taylor(1) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(0), 2, ((1, 1, 1),))
        check(Chebyshev(3) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(2), 2, ((1, 0, 0), (0, 0, 1), (1, 1, 1)))

        # a nonzero Fourier mean is sent to zero in any direction, as by the materialized operator
        for (s, dim, ℰ) ∈ ((Taylor(1) ⊗ Fourier(1, 1.0), 2, (nothing, E₀)), (Fourier(1, 1.0) ⊗ Taylor(1), 1, (E₀, nothing)),
                           (Taylor(2) ⊗ Fourier(2, 1.0), 2, (nothing, E₀)))
            a = Sequence(s, collect(1.0:dimension(s)))
            a₀ = copy(a)
            selectdim(a₀, dim, 0) .= 0.0
            @test integrate(a, ℰ) == integrate(a₀, ℰ)
            @test integrate(a, ℰ) ≈ project(Integral(ℰ), s, codomain(Integral(ℰ), s), ComplexF64) * a
        end

        # a middle direction with a Chebyshev factor of order 0 agrees with the base space operator
        s = Taylor(0) ⊗ Chebyshev(0) ⊗ Fourier(2, 1.0)
        a = Sequence(s, [1.0, 2.0, 3.0, 4.0, 5.0])
        r = Integral(nothing, E₀, nothing) * a
        @test space(r) == Taylor(0) ⊗ Chebyshev(1) ⊗ Fourier(2, 1.0)
        for k ∈ -2:2
            r₁ = Integral() * Sequence(Chebyshev(0), [a[(0, 0, k)]])
            @test r[(0, 0, k)] == r₁[0] && r[(0, 1, k)] == r₁[1]
        end

        # each direction agrees with the base space operator on a line of coefficients
        b = Sequence(Chebyshev(3) ⊗ Chebyshev(3) ⊗ Chebyshev(3), [sin(1.7i) for i ∈ 1:64])
        for d ∈ 1:3
            r = Integral(ntuple(i -> i == d ? E₀ : nothing, 3)) * b
            line = Sequence(Chebyshev(3), [b[ntuple(i -> i == d ? j : 2, 3)] for j ∈ 0:3])
            @test [r[ntuple(i -> i == d ? j : 2, 3)] for j ∈ 0:4] ≈ coefficients(Integral() * line)
        end
    end

    @testset "Condition" begin
        # the integral from -1 on Chebyshev
        ∫₋₁ = Integral(Evaluation(-1.0))
        inplace!(c, a) = integrate!(c, a, Evaluation(-1.0))
        @test domain(∫₋₁, Chebyshev(3)) == UndefSpace() # the T₀ row is dense, so no finite domain exists
        @test codomain(∫₋₁, Chebyshev(3)) == Chebyshev(4)
        test_action(∫₋₁, Sequence(Chebyshev(0), [2.0]), Sequence(Chebyshev(1), [2.0, 1.0]); inplace! = inplace!)
        test_action(∫₋₁, Sequence(Chebyshev(1), [1.0, 3.0]), Sequence(Chebyshev(2), [-0.5, 0.5, 0.75]); inplace! = inplace!)
        test_action(∫₋₁, Sequence(Chebyshev(3), [1.0, 2.0, 3.0, 4.0]), Sequence(Chebyshev(4), [-1.0, -1.0, -0.5, 0.5, 0.5]); inplace! = inplace!)
        @test ∫₋₁(Sequence(Chebyshev(1), ComplexF64[2 + 2im, 4.0])) == Sequence(Chebyshev(2), ComplexF64[0 + 2im, 1 + 1im, 1.0])
        @test integrate(Sequence(Chebyshev(1), interval.([1.0, 3.0])), Evaluation(interval(-1.0))) == Sequence(Chebyshev(2), interval.([-0.5, 0.5, 0.75]))

        # only the coefficient of the constants differs from the integral without condition: the
        # integral vanishes at the value of the condition, and the materialized operator agrees
        for x ∈ (-1.0, 0.3, 1.0), s ∈ (Taylor(4), Fourier(3, 1.0), Chebyshev(4))
            ℐ = Integral(Evaluation(x))
            a = Sequence(s, collect(1.0:dimension(s)))
            s isa Fourier && (a[0] = 0.0)
            c, c₀ = ℐ(a), Integral()(a)
            @test all(c[k] == c₀[k] for k ∈ indices(space(c)) if k != 0)
            @test abs(c(x)) ≤ 1e-13
            @test differentiate(c) ≈ a
            @test project(ℐ, s, space(c), eltype(c)) * a ≈ c == mul!(fill(eltype(c)(Inf), space(c)), ℐ, a)
        end
        @test Integral(Evaluation(0.0))(Sequence(Taylor(2), [1.0, -1.0, 1.0])) == Integral()(Sequence(Taylor(2), [1.0, -1.0, 1.0]))

        # the iterated integral from -1 vanishes at -1 with its derivative
        u = Sequence(Chebyshev(2), [1.0, 0.5, 0.25])
        W = ∫₋₁(∫₋₁(u))
        @test (∫₋₁ * ∫₋₁) * u == W
        @test abs(W(-1.0)) ≤ 1e-13 && abs(differentiate(W)(-1.0)) ≤ 1e-13
        @test differentiate(W, 2) ≈ u

        # one condition per dimension: a value, none, or no integration
        s = Taylor(2) ⊗ Chebyshev(2) ⊗ Fourier(1, 1.0)
        a = Sequence(s, ComplexF64.(1.0:dimension(s)))
        selectdim(a, 3, 0) .= 0.0
        for ℰ ∈ ((Evaluation(0.5), Evaluation(-1.0), Evaluation(0.2)), (E₀, Evaluation(1.0), nothing),
                 (nothing, E₀, Evaluation(0.7)), (Evaluation(0.25), nothing, nothing))
            ℐ = Integral(ℰ)
            c = ℐ(a)
            @test project(ℐ, s, codomain(ℐ, s), ComplexF64) * a ≈ c == integrate(a, ℰ)
            @test differentiate(c, order(ℐ)) ≈ a
            for (d, xᵢ) ∈ enumerate(RadiiPolynomial._integral_value(ℰ))
                isnothing(xᵢ) || @test maximum(abs, coefficients(evaluate(c, ntuple(i -> i == d ? xᵢ : nothing, 3)))) ≤ 1e-12
            end
        end

        # on a symmetric space, the constant added by the condition keeps the symmetries of the constants:
        # the integral of an odd series stays even, the integral of an even series loses its symmetry
        sE, sO = evensym(Fourier(3, 1.0)), oddsym(Fourier(3, 1.0))
        full(x) = Sequence(desymmetrize(space(x)), [RadiiPolynomial.getcoefficient(x, (desymmetrize(space(x)), j)) for j ∈ -3:3])
        for (s, codom) ∈ ((sO, sE), (sE, RadiiPolynomial._compatible_space_with_constant_index(sO)))
            ℐ = Integral(Evaluation(0.3))
            a = Sequence(s, ComplexF64[0.0, 2.0, 0.0, 3.0][1:dimension(s)])
            @test domain(ℐ, s) == UndefSpace()
            @test codomain(ℐ, s) == codom
            @test full(ℐ(a)) ≈ ℐ(full(a))
            @test abs(ℐ(a)(0.3)) ≤ 1e-13
        end

        @testset "InfiniteSequence error propagation" begin
            finite, tail = RadiiPolynomial._integral_condition_finite_error, RadiiPolynomial._integral_condition_tail_error
            X = Ell1(GeometricWeight(2.0))

            # the factors bound the column norms |(∫ eⱼ)(x)| / wⱼ of the row filled by the condition, over
            # the head columns j ≤ N and over the tail columns; on Chebyshev at ±1 they are attained
            for s ∈ (Taylor(3), Fourier(3, 0.5), Chebyshev(0), Chebyshev(1), Chebyshev(3)), x ∈ (-1.0, 0.3, 1.0, 0.5im)
                s₊ = s isa Fourier ? Fourier(40, 0.5) : typeof(s)(40)
                col = map(indices(s₊)) do j
                    e = zeros(ComplexF64, s₊)
                    e[j] = 1.0
                    abs(Integral(Evaluation(x))(e)[0]) / norm(e, X)
                end
                head = [c for (j, c) ∈ zip(indices(s₊), col) if abs(j) ≤ order(s)]
                rest = [c for (j, c) ∈ zip(indices(s₊), col) if abs(j) > order(s)]
                @test maximum(head) ≤ finite(X, s, x) * (1 + 1e-12)
                @test maximum(rest) ≤ tail(X, s, x) * (1 + 1e-12)
                if s isa Chebyshev && isreal(x) && abs(x) == 1
                    @test maximum(head) ≈ finite(X, s, x)
                    @test maximum(rest) ≈ tail(X, s, x)
                end
            end

            # on Chebyshev, the condition at -1 gives the factors of the integral from -1: finite and
            # total 1 + ν, cross 1/(2ν(j-1)) + 1/(νʲ(j²-1)) at j = N+1
            a = InfiniteSequence(Sequence(Chebyshev(3), [1.0, 0.5, 0.2, 0.05]), 0.5, 0.25, 0.75, X)
            r₀, r = integrate(a), integrate(a, Evaluation(-1.0))
            @test sequence(r) == integrate(sequence(a), Evaluation(-1.0))
            @test finite_error(r) ≈ (1 + 2.0) * 0.5 + (1 / (2 * 2.0 * 3) + 1 / (2.0^4 * 15)) * 0.25
            @test tail_error(r) == tail_error(r₀)
            @test total_error(r) ≈ min(finite_error(r) + tail_error(r), (1 + 2.0) * 0.75)
            @test banachspace(r) == X

            @test_throws DomainError integrate(a, Evaluation(3.0)) # outside the Bernstein ellipse
            # a sequence without error is integrated without error
            b = InfiniteSequence(Sequence(Chebyshev(1), [1.0, 3.0]), Ell1(IdentityWeight()))
            @test iszero(total_error(integrate(b, Evaluation(-1.0))))

            # in one dimension, the condition adds its head columns to the finite factor and its tail
            # columns to the cross factor
            factors = (RadiiPolynomial._integral_finite_error, RadiiPolynomial._integral_cross_error,
                       RadiiPolynomial._integral_tail_error, RadiiPolynomial._integral_total_error)
            κf, κc, κt, κo = factors
            for s ∈ (Taylor(3), Fourier(3, 0.5), Chebyshev(3)), x ∈ (-1.0, 0.3)
                ℰ = Evaluation(x)
                @test κf(X, s, ℰ) == κf(X, s, E₀) + finite(X, s, x)
                @test κc(X, s, ℰ) == κc(X, s, E₀) + tail(X, s, x)
                @test κt(X, s, ℰ) == κt(X, s, E₀)
                @test κo(X, s, ℰ) == κo(X, s, E₀) + max(finite(X, s, x), tail(X, s, x))
            end

            # on a tensor space the factors are those of the product of the integrals along each dimension:
            # they bound the column norms, restricted to the head rows and to the other rows
            X² = Ell1((GeometricWeight(2.0), GeometricWeight(1.5)))
            for (s, big) ∈ ((Taylor(2) ⊗ Chebyshev(2), Taylor(9) ⊗ Chebyshev(9)), (Fourier(1, 1.0) ⊗ Taylor(2), Fourier(6, 1.0) ⊗ Taylor(9))),
                ℰ ∈ ((Evaluation(0.5), E₀), (E₀, Evaluation(-1.0)), (Evaluation(0.25), Evaluation(1.0)), (nothing, Evaluation(0.5)), (Evaluation(-0.5), nothing))
                head = codomain(Integral(ℰ), s)
                fin = cr = tl = tot = 0.0
                for j ∈ indices(big)
                    all(abs.(j) .≤ Tuple(order(big)) .- 2) || continue
                    e = zeros(ComplexF64, big)
                    e[j] = 1.0
                    img = Integral(ℰ)(e)
                    w = norm(e, X²)
                    h, t = norm(project(img, head), X²) / w, norm(img, X²) / w
                    if all(abs.(j) .≤ Tuple(order(s)))
                        fin = max(fin, t)
                    else
                        cr, tl = max(cr, h), max(tl, t - h)
                    end
                    tot = max(tot, t)
                end
                @test fin ≤ κf(X², s, ℰ) * (1 + 1e-12)
                @test cr ≤ κc(X², s, ℰ) * (1 + 1e-12)
                @test tl ≤ κt(X², s, ℰ) * (1 + 1e-12)
                @test tot ≤ κo(X², s, ℰ) * (1 + 1e-12)
            end
            Y² = Ell1((GeometricWeight(2.0), GeometricWeight(2.0)))
            c = InfiniteSequence(Sequence(Taylor(2) ⊗ Taylor(2), ones(9)), 0.5, 0.25, 0.75, Y²)
            t = integrate(c, (Evaluation(0.5), E₀))
            @test finite_error(t) == (2.0 + 0.5) * 2.0 * 0.5 + 0.5 * 0.25^3 / 4 * 2.0 * 0.25
            @test total_error(t) == min(finite_error(t) + tail_error(t), (2.0 + 0.5) * 2.0 * 0.75)
        end
    end

    @testset "Cartesian space" begin
        a = Sequence(Taylor(2)^2 × Fourier(1, 1.0) × Chebyshev(2), collect(1.0:12.0))
        component(a, 2)[0] = 0.0 # mean-free Fourier component
        ∫ = Integral()
        c = ∫(a)
        codom = Taylor(3)^2 × Fourier(1, 1.0) × Chebyshev(3)
        @test c == integrate!(fill(complex(Inf), codom), a) == mul!(fill(complex(Inf), codom), ∫, a)
        @test component(c, 1) == integrate(component(a, 1))
        @test component(c, 3) == integrate(component(a, 3))
    end

    @testset "Symmetric space" begin
        sE, sO = evensym(Fourier(3, 1.0)), oddsym(Fourier(3, 1.0))
        ∫ = Integral()

        # the integral swaps the even and odd symmetries
        @test codomain(∫, sE) == domain(∫, sE) == sO
        @test codomain(∫, sO) == domain(∫, sO) == sE

        # aₖ/(iωk) on the representatives k = 1, 2, 3
        a = Sequence(sE, ComplexF64[0.0, 2.0, 0.0, 3.0])
        expected = Sequence(sO, ComplexF64[-2.0im, 0.0, -1.0im])
        test_action(∫, a, expected; inplace! = integrate!)

        # the same computation on the full Fourier space
        full(x) = Sequence(desymmetrize(space(x)), [RadiiPolynomial.getcoefficient(x, (desymmetrize(space(x)), j)) for j ∈ -3:3])
        @test integrate(full(a)) == full(expected)

        @test differentiate(∫(a)) == a

        # no integration
        @test domain(Integral(nothing), sE) == codomain(Integral(nothing), sE) == sE
        @test Integral(nothing) * a == integrate(a, nothing) == a

        # the mean is sent to zero
        a_mean = Sequence(sE, ComplexF64[1.0, 2.0, 0.0, 3.0])
        @test integrate(a_mean) == ∫(a_mean) == expected
    end

    @testset "InfiniteSequence error propagation" begin
        factors = (RadiiPolynomial._integral_finite_error, RadiiPolynomial._integral_cross_error,
                   RadiiPolynomial._integral_tail_error, RadiiPolynomial._integral_total_error)
        finite, cross, tail, total = factors
        X = Ell1(GeometricWeight(2.0))
        X² = Ell1((GeometricWeight(2.0), GeometricWeight(2.0)))

        # Taylor: finite and total ν, tail ν/(N+2)
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

        # no integration, no error
        d = InfiniteSequence(Sequence(Taylor(2), [1.0, 1.0, 1.0]), 0.5, 0.25, 0.75, X)
        ∫⁰d = integrate(d, nothing)
        @test sequence(∫⁰d) == sequence(d)
        @test (finite_error(∫⁰d), tail_error(∫⁰d), total_error(∫⁰d)) == (0.5, 0.25, 0.75)

        # the tensor tail factor covers the complement of the box: max_i tailᵢ ∏_{l≠i} totalₗ
        s² = Taylor(2) ⊗ Taylor(2)
        @test finite(X², s², (E₀, E₀)) == total(X², s², (E₀, E₀)) == 2.0 * 2.0
        @test tail(X², s², (E₀, E₀)) == 0.5 * 2.0
        @test tail_error(integrate(InfiniteSequence(Sequence(s², ones(9)), 0.0, 1.0, 1.0, X²), (E₀, E₀))) == 1.0
        # along one dimension only
        @test finite(X², s², (E₀, nothing)) == total(X², s², (E₀, nothing)) == 2.0
        @test finite_error(integrate(InfiniteSequence(Sequence(s², ones(9)), 0.5, 0.0, 0.5, X²), (E₀, nothing))) == 2.0 * 0.5

        # Fourier: finite 1/ω (0 when N = 0), tail 1/(ω(N+1)), total 1/ω
        f = InfiniteSequence(Sequence(Fourier(3, 0.5), [0, 0, 0.5im, 0, -0.5im, 0, 0]), 0.25, 1.0, 1.25, X)
        ∫f = integrate(f)
        @test sequence(∫f) == integrate(sequence(f))
        @test finite_error(∫f) == 2.0 * 0.25
        @test tail_error(∫f) == 0.5 * 1.0
        @test total_error(∫f) == min(2.0 * 0.25 + 0.5 * 1.0, 2.0 * 1.25)
        @test finite(X, Fourier(0, 0.5), E₀) == 0.0
        @test total(X, Fourier(0, 0.5), E₀) == inv(0.5)

        # the Taylor and Fourier integrals separate supports, so the cross factor vanishes
        @test cross(X, Taylor(2), E₀) == cross(X, Fourier(3, 1.0), E₀) == 0.0

        @testset "Chebyshev: triangular propagation" begin
            # A tail column j reaches the head row j-1. Both norms carry the weight and the head of
            # the image is Chebyshev(N+1).
            for ν ∈ (1.0, 1.5, 2.0), N ∈ (0, 1, 3, 8)
                Xν = Ell1(GeometricWeight(ν))
                cols = _calculus_chebyshev_columns(integrate, Chebyshev(N+1), Xν, Xν)
                _calculus_test_chebyshev_factors(factors, Xν, N, cols, E₀)
                # the finite and total factors are attained at j = 0
                @test finite(Xν, Chebyshev(N), E₀) == total(Xν, Chebyshev(N), E₀) == ν
            end

            a = InfiniteSequence(Sequence(Chebyshev(3), [1.0, 0.5, 0.2, 0.05]), 0.5, 0.25, 0.75, X)
            _calculus_test_triangular(integrate(a), factors, X, Chebyshev(3), E₀)
            @test banachspace(integrate(a)) == X

            # ν = 1 is admissible
            @test finite(Ell1(GeometricWeight(1.0)), Chebyshev(2), E₀) == 1.0
            @test cross(X, Chebyshev(2), nothing) == 0.0
            @test finite(X, Chebyshev(2), nothing) == tail(X, Chebyshev(2), nothing) == 1.0
        end
    end

    @testset "Composition" begin
        ∫ = Integral()
        a = Sequence(Taylor(2), [1.0, -1.0, 1.0])

        # higher orders are compositions
        ∫² = ∫ * ∫
        @test ∫²(a) == ∫(∫(a)) == Sequence(Taylor(4), [0.0, 0.0, 1/2, -1/6, 1/12])
        @test codomain(∫², Taylor(3)) == Taylor(5)
        @test project(∫², Taylor(2), Taylor(4), Float64) * a == ∫²(a)

        # the power is the repeated composition
        @test ∫^1 == ∫
        @test ∫^2 == ∫²
        @test (∫^3)(a) == ∫(∫(∫(a)))
        @test (∫^0)(a) == a
        @test_throws DomainError ∫^(-1)
        n = -1
        @test_throws DomainError ∫^n
        ∫₋₁ = Integral(Evaluation(-1))
        b = Sequence(Chebyshev(2), [1.0, 0.5, 0.5])
        @test (∫₋₁^2)(b) == ∫₋₁(∫₋₁(b))

        # there is no integer order
        @test_throws MethodError Integral(1)
        @test_throws MethodError Integral((1, 0))
        @test_throws MethodError integrate(a, 1)
        @test_throws MethodError integrate!(fill(Inf, Taylor(4)), a, 2)
    end

end
