# the evaluation of `a` at `x` agrees with the value `v` across its entry points
function test_evaluation(a, x, v; cmp = ==)
    s₀ = codomain(Evaluation(x), space(a))
    P = project(Evaluation(x), space(a), s₀, typeof(v))
    c = Sequence(s₀, [v])
    @testset let x = x, dom = space(a)
        @test cmp(P(a), c)
        @test cmp(evaluate!(fill(typeof(v)(Inf), s₀), a, x), c)
        @test cmp(mul!(fill(typeof(v)(Inf), s₀), P, a), c)
        @test cmp(a(x), v)
        @test cmp(evaluate(a, x), v)
        @test cmp(Evaluation(x) * a, v)
    end
    return nothing
end

# the evaluation on a symmetric space agrees with the evaluation of the desymmetrized sequence,
# including the coefficients reconstructed from the symmetry of the codomain
function test_symmetric_evaluation(a, x)
    full = Projection(desymmetrize(space(a))) * a
    c = evaluate(a, x)
    cfull = evaluate(full, x)
    @test desymmetrize(space(c)) == space(cfull)
    @test all(k -> isapprox(c[k], cfull[k]; atol = 1e-12), indices(space(c)))
    @test all(k -> isapprox(RadiiPolynomial.getcoefficient(c, (space(cfull), k)), cfull[k]; atol = 1e-12), indices(space(cfull)))
    return c
end

@testset "Evaluation" begin

    @testset "Constructors and value" begin
        @test Evaluation(1.0) isa Evaluation{Float64}
        @test value(Evaluation(1.0)) == 1.0
        @test Evaluation(nothing) isa Evaluation{Nothing}
        @test value(Evaluation(nothing)) === nothing
        @test Evaluation(1.0, nothing, 2.0) == Evaluation((1.0, nothing, 2.0))
        @test value(Evaluation(1.0, nothing, 2.0)) == (1.0, nothing, 2.0)
        @test_throws ArgumentError Evaluation()
    end

    @testset "domain, codomain and identity on $s" for (s, s₀, x) ∈ (
            (Taylor(2), Taylor(0), 2.0), (Fourier(1, 1.0), Fourier(0, 1.0), 0.0), (Chebyshev(2), Chebyshev(0), 0.4))
        @test domain(Evaluation(nothing), s) == codomain(Evaluation(nothing), s) == s
        @test domain(Evaluation(x), s) == UndefSpace()
        @test codomain(Evaluation(x), s) == s₀
        for T ∈ (Float64, ComplexF64)
            a = Sequence(s, T[1.0, 2.0, 3.0])
            @test evaluate(a, nothing) == a
            @test mul!(fill(T(Inf), s), Evaluation(nothing), a) == a
        end
    end

    @testset "Taylor" begin
        𝒯 = Taylor(2)
        a = Sequence(𝒯, [1.0, -2.0, 3.0]) # p(t) = 1 - 2t + 3t²

        test_evaluation(a, 2.0, 9.0)
        @test evaluate!([Inf], a, 2.0) == [9.0]
        @test a(0.0) == 1.0
        @test coefficients(project(Evaluation(2.0), 𝒯, Taylor(0), Float64)) == [1.0 2.0 4.0]

        @test_throws ArgumentError evaluate!(Sequence(Taylor(1), [Inf, Inf]), a, 2.0)

        z = 1.0 + 2.0im
        @test a(z) ≈ 1.0 - 2.0*z + 3.0*z^2
        @test Sequence(𝒯, ComplexF64[1.0, -2.0, 3.0])(2.0) == 9.0 + 0.0im

        a_i = Sequence(𝒯, interval.([1.0, -2.0, 3.0]))
        @test in_interval(9.0, a_i(interval(2.0)))
        @test in_interval(9.0, a_i(interval(1.9, 2.1)))
    end

    @testset "Fourier" begin
        ω = 1.0
        ℱ = Fourier(1, ω)
        a = Sequence(ℱ, ComplexF64[1.0+1.0im, 2.0+0.0im, 1.0-1.0im])

        # cis(0) = 1 exactly, so a(0) = a₋₁ + a₀ + a₁
        test_evaluation(a, 0.0, 4.0 + 0.0im)
        @test coefficients(project(Evaluation(0.0), ℱ, Fourier(0, ω), ComplexF64)) == ComplexF64[1 1 1]

        # cis(±π) = -1, so a(π) = a₀ - a₁ - a₋₁
        @test a(π) ≈ 0.0 atol=1e-12

        x = 0.7
        P = project(Evaluation(x), ℱ, Fourier(0, ω), ComplexF64)
        @test coefficients(P) ≈ [cis(-ω*x) cis(0.0*x) cis(ω*x)]
        @test (P * a)[0] ≈ a(x)

        T = 2π/ω
        @test a(0.37) ≈ a(0.37 + T)
        @test a(0.37) ≈ a(0.37 - 2T)

        a0 = Sequence(Fourier(0, ω), ComplexF64[3.0])
        @test a0(1.234) == a0(-5.6) == 3.0 + 0.0im

        @test Sequence(ℱ, [0.5, 1.0, 0.5])(0.3) isa ComplexF64

        @test in_interval(4.0 + 0.0im, Sequence(ℱ, interval.([1.0, 2.0, 1.0]))(interval(0.0)))
    end

    @testset "Chebyshev" begin
        # f(x) = a₀ + 2∑ₖ aₖ Tₖ(x)
        𝒞 = Chebyshev(2)
        a = Sequence(𝒞, [1.0, 2.0, 3.0])
        Tk(k, x) = cos(k*acos(x))

        test_evaluation(a, 0.4, a[0] + 2*a[1]*Tk(1, 0.4) + 2*a[2]*Tk(2, 0.4); cmp = ≈)

        # the values and the row operators are exact at x = 0, ±1
        for (x, v, row) ∈ ((1.0, 11.0, [1.0 2.0 2.0]), (-1.0, 3.0, [1.0 -2.0 2.0]), (0.0, -5.0, [1.0 0.0 -2.0]))
            @test a(x) == v
            @test coefficients(project(Evaluation(x), 𝒞, Chebyshev(0), Float64)) == row
        end

        a0 = Sequence(Chebyshev(0), [5.0])
        @test a0(0.3) == a0(-0.9) == 5.0
        @test Sequence(Chebyshev(1), [1.0, 2.0])(0.3) ≈ 1.0 + 2*2*0.3

        z = 0.3 + 0.2im
        @test a(z) ≈ a[0] + 2*a[1]*Tk(1, z) + 2*a[2]*Tk(2, z)

        @test in_interval(11.0, Sequence(𝒞, interval.([1.0, 2.0, 3.0]))(interval(1.0)))
    end

    @testset "TensorSpace (partial evaluation)" begin
        # the first index runs fastest: a(0,-1)=1, a(1,-1)=2, a(0,0)=3, a(1,0)=4, a(0,1)=5, a(1,1)=6
        a = Sequence(Taylor(1) ⊗ Fourier(1, 1.0), collect(1.0:6.0))
        @test a(2.0, nothing) == evaluate(a, (2.0, nothing)) == Sequence(Taylor(0) ⊗ Fourier(1, 1.0), [5.0, 11.0, 17.0])
        @test evaluate(a, (nothing, 0.0)) == Sequence(Taylor(1) ⊗ Fourier(0, 1.0), ComplexF64[9.0, 12.0])
        @test evaluate(a, (2.0, 0.0)) == 33.0 + 0.0im # a fully specified tuple gives a scalar
        @test evaluate(a, (nothing, nothing)) == a

        # Fourier as the leading factor
        d = Sequence(Fourier(1, 1.0) ⊗ Taylor(1), collect(1.0:6.0))
        @test evaluate(d, (0.0, nothing)) == Sequence(Fourier(0, 1.0) ⊗ Taylor(1), ComplexF64[6.0, 15.0])

        # Chebyshev as a trailing factor of order 0, 1 and 2, at x = 0, ±1 and at a generic x
        𝒯𝒞 = Taylor(1) ⊗ Chebyshev(0)
        e2 = Sequence(Taylor(1) ⊗ Chebyshev(2), collect(1.0:6.0))
        @test evaluate(e2, (nothing, 0.5)) == Sequence(𝒯𝒞, [-1.0, 0.0]) # T₁(0.5) = 0.5, T₂(0.5) = -0.5
        @test evaluate(e2, (nothing, 0.0)) == Sequence(𝒯𝒞, [-9.0, -10.0])
        e1 = Sequence(Taylor(1) ⊗ Chebyshev(1), collect(1.0:4.0))
        @test evaluate(e1, (nothing, -1.0)) == Sequence(𝒯𝒞, [-5.0, -6.0])
        @test evaluate(e1, (nothing, 1.0)) == Sequence(𝒯𝒞, [7.0, 10.0])
        @test evaluate(e1, (nothing, 0.3)) ≈ Sequence(𝒯𝒞, [1.0 + 2*0.3*3.0, 2.0 + 2*0.3*4.0])
        e0 = Sequence(𝒯𝒞, [1.0, 2.0])
        @test evaluate(e0, (nothing, 0.3)) == e0
    end

    @testset "CartesianSpace" begin
        a = Sequence(Taylor(1)^2 × Fourier(1, 1.0), collect(1.0:7.0))
        v = evaluate(a, 0.5)
        @test v isa Vector{ComplexF64}
        @test v[1:2] == [2.0, 5.0] # the Taylor components 1 + 2x and 3 + 4x
        @test v == vcat(component(a, 1)(0.5), component(a, 2)(0.5))
        @test evaluate!(fill(complex(Inf), 3), a, 0.5) == v
        @test evaluate(a, nothing) == a
    end

    @testset "SymmetricSpace" begin
        # p(t) = 1 + 2t² + 3t⁴
        s = evensym(Taylor(4))
        @test indices(s) == 0:2:4
        a = Sequence(s, [1.0, 2.0, 3.0])
        full = Projection(desymmetrize(s)) * a
        @test full == Sequence(Taylor(4), [1.0, 0.0, 2.0, 0.0, 3.0])
        cs = codomain(Evaluation(0.3), s)
        @test desymmetrize(cs) == Taylor(0)
        @test indices(cs) == 0:0
        @test a(0.3) == full(0.3) == 1.0 + 2*0.3^2 + 3*0.3^4
        @test evaluate(a, nothing) == a

        a2 = Sequence(oddsym(Fourier(2, 1.0)), ComplexF64[1.0, 2.0])
        @test a2(0.4) ≈ (Projection(Fourier(2, 1.0)) * a2)(0.4)

        a3 = Sequence(oddsym(Chebyshev(3)), [1.0, 2.0])
        full3 = Projection(Chebyshev(3)) * a3
        @test full3 == Sequence(Chebyshev(3), [0.0, 1.0, 0.0, 2.0])
        @test a3(0.5) == full3(0.5) == -3.0

        s4 = d4sym(Fourier(1, 1.0) ⊗ Fourier(1, 1.0))
        a4 = Sequence(s4, ComplexF64[1.0, 2.0, 3.0])
        x = (0.3, 0.4)
        @test a4(x...) ≈ (Projection(desymmetrize(s4)) * a4)(x...)
        cs4 = codomain(Evaluation(x), s4)
        @test desymmetrize(cs4) == Fourier(0, 1.0) ⊗ Fourier(0, 1.0)
        @test indices(cs4) == [(0, 0)]

        # on a symmetric tensor space only the tuple of `nothing` has a domain
        @test domain(Evaluation((0.3, nothing)), s4) == UndefSpace()
        @test domain(Evaluation((nothing, nothing)), s4) == s4

        s5 = Chebyshev(3) ⊗ Chebyshev(2) ⊗ evensym(Fourier(2, 1.0))
        a5 = Sequence(s5, [sin(k) for k ∈ 1:dimension(s5)])
        full5 = Projection(desymmetrize(s5)) * a5
        for x ∈ ((0.3, -0.4, nothing), (0.3, nothing, nothing), (nothing, -0.4, nothing),
                 (nothing, nothing, 0.7), (0.3, nothing, 0.7), (nothing, -0.4, 0.7))
            test_symmetric_evaluation(a5, x)
        end
        @test a5(0.3, -0.4, 0.7) ≈ full5(0.3, -0.4, 0.7)
        @test evaluate(a5, (nothing, nothing, nothing)) == a5

        # evaluating a variable whose space has order 0 is the identity
        a6 = Sequence(Chebyshev(0) ⊗ evensym(Fourier(2, 1.0)), [1.0, 2.0, 3.0])
        @test evaluate(a6, (0.3, nothing)) == a6

        # the sign carried by an odd symmetry is applied along the orbit
        s7 = oddsym(Chebyshev(3)) ⊗ evensym(Fourier(2, 1.0))
        a7 = Sequence(s7, ComplexF64[1.0, 2.0im, 3.0, 4.0im, 5.0, 6.0im])
        test_symmetric_evaluation(a7, (0.5, nothing))
        test_symmetric_evaluation(a7, (nothing, 0.4))
        @test a7(0.5, 0.4) ≈ (Projection(desymmetrize(s7)) * a7)(0.5, 0.4)

        # the codomain inherits the symmetries acting trivially on the evaluated variables, whatever
        # their amplitude, so an odd symmetry in the remaining variable is kept
        a9 = Sequence(Chebyshev(2) ⊗ oddsym(Fourier(2, 1.0)), ComplexF64[1.0, 2.0im, 3.0, 4.0im, 5.0, 6.0im])
        @test indices(space(test_symmetric_evaluation(a9, (0.3, nothing)))) == [(0, 1), (0, 2)]

        # an element carrying a phase on the evaluated variable is dropped: keeping the product of the
        # two odd reflections would declare the result even in the remaining variable
        a10 = Sequence(oddsym(Chebyshev(3)) ⊗ oddsym(Fourier(2, 1.0)), ComplexF64[1.0, 2.0im, 3.0, 4.0im])
        @test indices(space(test_symmetric_evaluation(a10, (0.5, nothing)))) == [(0, 1), (0, 2)]
        @test indices(space(test_symmetric_evaluation(a10, (nothing, 0.4)))) == [(1, 0), (3, 0)]

        # the rotations and the diagonal reflection of D₄ mix the variables and are dropped, the axis
        # reflection is kept
        s11 = d4sym(Fourier(2, 1.0) ⊗ Fourier(2, 1.0))
        a11 = Sequence(s11, ComplexF64[(1 + 0.4im) / (1 + sum(abs, k))^2 for k ∈ indices(s11)])
        @test indices(space(test_symmetric_evaluation(a11, (0.3, nothing)))) == [(0, 0), (0, 1), (0, 2)]
        @test indices(space(test_symmetric_evaluation(a11, (nothing, 0.3)))) == [(0, 0), (1, 0), (2, 0)]

        aI = Sequence(interval(s5), interval.(coefficients(a5)))
        @test in_interval(real(full5(0.3, -0.4, 0.7)), real(aI(interval(0.3), interval(-0.4), interval(0.7))))
        cI = evaluate(aI, (interval(0.3), nothing, interval(0.7)))
        c5 = evaluate(a5, (0.3, nothing, 0.7))
        @test all(k -> in_interval(real(c5[k]), real(cI[k])), indices(space(cI)))
    end

    @testset "InfiniteSequence: the point must lie in the domain of convergence" begin
        ν₂, ν₁ = Ell1(GeometricWeight(2.0)), Ell1()
        exact_error(a, X) = InfiniteSequence(a, 0.0, 0.0, 0.0, X)
        𝒯, ℱ, 𝒞 = Sequence(Taylor(2), [1.0, 1.0, 1.0]), Sequence(Fourier(1, 1.0), ComplexF64[0.5, 1.0, 0.5]), Sequence(Chebyshev(2), [1.0, 1.0, 1.0])
        encloses(v) = y -> in_interval(v, y)
        # Taylor: disk of radius ν. Fourier: strip |ω Im(x)| ≤ log(ν). Chebyshev: Bernstein ellipse of
        # parameter ν, whose semi-minor axis is 0.75 for ν = 2. The weight ν = 1 gives the unit disk,
        # the real axis and the segment [-1, 1].
        for (a, x, check, outside) ∈ (
                (exact_error(𝒯, ν₂), 1.5, encloses(1.0 + 1.5 + 1.5^2), (3.0,)),
                (InfiniteSequence(𝒯, ν₁), 1.0, encloses(3.0), (1.5,)),
                (exact_error(ℱ, ν₂), 0.3 + 0.5im, y -> y isa Complex, (0.3 + 1.0im,)),
                (InfiniteSequence(ℱ, ν₁), 0.3, y -> y isa Complex, (0.3 + 0.1im,)),
                (exact_error(𝒞, ν₂), 0.5, encloses(1.0), (2.0, 2.0im)),
                (InfiniteSequence(𝒞, ν₁), 1.0, encloses(5.0), (0.5im, 1.5)),
                (exact_error(Sequence(evensym(Taylor(4)), [1.0, 2.0, 3.0]), ν₂), 1.5, encloses(1.0 + 2*1.5^2 + 3*1.5^4), (3.0,)))
            @test check(evaluate(a, x))
            for y ∈ outside
                @test_throws DomainError evaluate(a, y)
            end
        end

        # `nothing` passes whatever the Banach space
        @test space(evaluate(InfiniteSequence(𝒯, ν₁), nothing)) == Taylor(2)

        # each factor of a tensor space is checked
        t = exact_error(Sequence(Taylor(1) ⊗ Taylor(1), [1.0, 1.0, 1.0, 1.0]), Ell1((GeometricWeight(2.0), GeometricWeight(2.0))))
        @test space(evaluate(t, (0.5, nothing))) == Taylor(0) ⊗ Taylor(1)
        @test_throws DomainError evaluate(t, (3.0, 0.5))

        # an algebraic weight defines no evaluation domain
        g = InfiniteSequence(𝒯, Ell1(AlgebraicWeight(2.0)))
        @test_throws DomainError evaluate(g, 0.5)
        @test space(evaluate(g, nothing)) == Taylor(2)
    end

end
