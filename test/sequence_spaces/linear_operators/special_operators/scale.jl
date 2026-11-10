@testset "Scale" begin
    test_scale(a, γ, expected) = test_action(Scale(γ), a, expected; inplace! = (c, a) -> scale!(c, a, γ))

    # scaling commutes with reading the sequence on the full lattice
    scale_commutes(a, γ; cmp = ==) =
        cmp(project(scale(a, γ), desymmetrize(codomain(Scale(γ), space(a)))), scale(project(a, desymmetrize(space(a))), γ))

    @testset "constructors and arithmetic" begin
        test_valued_operator(Scale)
        @test Scale(2.0) * Scale(3.0) == Scale(6.0)
        @test Scale((2.0, 3.0)) * Scale((4.0, 5.0)) == Scale((8.0, 15.0))
        @test Scale(2.0)^3 == Scale(8.0)
        @test Scale((2.0, 3.0))^2 == Scale((4.0, 9.0))
        @test Scale((2.0, 3.0))^(2, 3) == Scale((4.0, 27.0))
    end

    @testset "Taylor: cₖ = γᵏ aₖ" begin
        a = Sequence(Taylor(3), [1.0, 2.0, 4.0, 8.0])
        @test domain(Scale(0.5), Taylor(3)) == codomain(Scale(0.5), Taylor(3)) == Taylor(3)
        test_scale(a, 0.5, ones(Taylor(3)))
        @test scale(a, 0.5) == ones(Taylor(3))
        @test scale(a, 1.0) == a
        @test scale(ones(Taylor(2)), 2.0) == Sequence(Taylor(2), [1.0, 2.0, 4.0])
        @test scale(Sequence(Taylor(2), ComplexF64[1.0+1.0im, 2.0, 3.0-1.0im]), 2.0) ==
            Sequence(Taylor(2), ComplexF64[1.0+1.0im, 4.0, 12.0-4.0im])

        γ = interval(0.5)
        @test scale(interval(a), γ) == Scale(γ)(interval(a)) == interval(ones(Taylor(3)))

        @test (Scale(2.0) * Scale(3.0))(ones(Taylor(2))) == Scale(2.0)(Scale(3.0)(ones(Taylor(2)))) == Sequence(Taylor(2), [1.0, 6.0, 36.0])

        @test_throws ArgumentError scale!(fill(Inf, Taylor(2)), a, 0.5)
    end

    @testset "Fourier: c_{γk} = aₖ for an integer γ" begin
        a = Sequence(Fourier(1, 1.0), [0.5, 1.0, 0.5])
        @test domain(Scale(2), Fourier(4, 1.0)) == Fourier(2, 1.0)
        @test codomain(Scale(2), Fourier(2, 1.0)) == Fourier(4, 1.0)
        expected = Sequence(Fourier(2, 1.0), [0.5, 0.0, 1.0, 0.0, 0.5])
        test_scale(a, 2, expected)
        @test scale(a, 2) == scale(a, 2.0) == expected
        @test scale(a, 1) == a

        @test_throws DomainError domain(Scale(0.5), Fourier(2, 1.0))
        @test_throws DomainError codomain(Scale(0.5), Fourier(2, 1.0))
        @test_throws DomainError scale(a, 0.5)

        # a negative γ reverses time
        b = Sequence(Fourier(1, 1.0), [10.0, 20.0, 30.0])
        @test scale(b, -1) == Sequence(Fourier(1, 1.0), [30.0, 20.0, 10.0])
        @test scale(b, -2) == Sequence(Fourier(2, 1.0), [30.0, 0.0, 20.0, 0.0, 10.0])

        @test (Scale(2) * Scale(3))(a) == Scale(2)(Scale(3)(a))
    end

    @testset "Chebyshev: only γ = 1 is supported" begin
        a = Sequence(Chebyshev(2), [1.0, 2.0, 3.0])
        @test domain(Scale(1.0), Chebyshev(2)) == codomain(Scale(1.0), Chebyshev(2)) == Chebyshev(2)
        test_scale(a, 1.0, a)
        @test scale(a, 1.0) == a
        @test_throws DomainError scale(a, 2.0)
        @test_throws DomainError project(Scale(2.0), Chebyshev(2), Chebyshev(2), Float64)
    end

    @testset "TensorSpace: one value per factor" begin
        𝒯ℱ = Taylor(1) ⊗ Fourier(1, 1.0)
        # a Fourier factor is resized along its own dimension, in leading and trailing position
        for (s, γ, expected) ∈ (
                (Taylor(1) ⊗ Chebyshev(1), (2.0, 1), Sequence(Taylor(1) ⊗ Chebyshev(1), [1.0, 4.0, 3.0, 8.0])),
                (𝒯ℱ, (1, 2), Sequence(Taylor(1) ⊗ Fourier(2, 1.0), [1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 5.0, 6.0])),
                (𝒯ℱ, (2.0, 1), Sequence(𝒯ℱ, [1.0, 4.0, 3.0, 8.0, 5.0, 12.0])),
                (Chebyshev(1) ⊗ Taylor(1), (1.0, 1.0), Sequence(Chebyshev(1) ⊗ Taylor(1), collect(1.0:4.0))),
                (Taylor(0) ⊗ Fourier(1, 1.0), (1.0, -1), Sequence(Taylor(0) ⊗ Fourier(1, 1.0), [3.0, 2.0, 1.0])),
                (Fourier(1, 1.0) ⊗ Taylor(0), (-1, 1.0), Sequence(Fourier(1, 1.0) ⊗ Taylor(0), [3.0, 2.0, 1.0])),
                (Fourier(1, 1.0) ⊗ Taylor(1), (2, 3.0), Sequence(Fourier(2, 1.0) ⊗ Taylor(1), [1.0, 0.0, 2.0, 0.0, 3.0, 12.0, 0.0, 15.0, 0.0, 18.0])))
            a = Sequence(s, collect(1.0:dimension(s)))
            test_scale(a, γ, expected)
            @test scale(a, γ) == expected
        end

        s = Taylor(3) ⊗ Taylor(2)
        test_scale(Sequence(s, [2.0 ^ sum(α) for α ∈ indices(s)]), (0.5, 0.5), ones(s))

        s₃ = Fourier(1, 1.0) ⊗ Taylor(1) ⊗ Chebyshev(1)
        a₃ = Sequence(s₃, collect(1.0:dimension(s₃)))
        test_scale(a₃, (-2, 3.0, 1), scale(a₃, (-2, 3.0, 1)))

        @test_throws DomainError scale(ones(Taylor(1) ⊗ Chebyshev(1)), (1.0, 2.0))
    end

    @testset "Symmetric space" begin
        F = Fourier(2, 1.0)
        sE = evensym(F)
        sO = oddsym(F)
        a = Sequence(sE, [1.0, 2.0, 3.0])
        reflection(F, angle) = SymmetricSpace(F, Group(GroupElement(LatticeAut([-1;;]), Cocycle(0//1, Rational{Int}[angle]))))
        swap = Group(GroupElement(LatticeAut([0 1; 1 0]), Cocycle(0//1, Rational{Int}[0, 0])))
        decaying(s) = Sequence(s, [1.0 / (1 + sum(abs, k))^2 for k ∈ indices(s)])

        # a reflection k ↦ -k keeps the space and flips the phase, a dilation divides the phase by γ
        @test codomain(Scale(-1), sE) == sE
        @test codomain(Scale(-1), sO) == sO
        @test codomain(Scale(2), sE) == evensym(Fourier(4, 1.0))
        @test domain(Scale(2), evensym(Fourier(4, 1.0))) == sE
        @test scale_commutes(a, -1)
        @test scale_commutes(a, 2)
        @test project(Scale(2), sE, evensym(Fourier(4, 1.0)), Float64)(a) == scale(a, 2)
        s3 = reflection(F, 4//3)
        @test codomain(Scale(2), s3) == reflection(Fourier(4, 1.0), 2//3)
        @test domain(Scale(2), codomain(Scale(2), s3)) == s3
        @test codomain(Scale(-1), s3) == reflection(F, -4//3)
        @test scale_commutes(Sequence(s3, ComplexF64[1, 2, 3]), -1; cmp = ≈)
        @test_throws DomainError codomain(Scale(1//2), sE)
        @test_throws DomainError codomain(Scale(0.5), sE)

        # mixed tensors: a Chebyshev axis only admits γ = 1, a Taylor axis keeps its cocycle
        m = evensym(Chebyshev(2)) ⊗ sE
        @test codomain(Scale((1, -1)), m) == m
        @test codomain(Scale((1, 2)), m) == evensym(Chebyshev(2)) ⊗ evensym(Fourier(4, 1.0))
        @test scale_commutes(decaying(m), (1, -1))
        @test scale_commutes(decaying(m), (1, 2))
        @test_throws DomainError codomain(Scale((2, 1)), m)
        mt = oddsym(Taylor(3)) ⊗ sE
        @test codomain(Scale((0.5, -1)), mt) == mt
        @test scale_commutes(decaying(mt), (0.5, -1); cmp = ≈)
        @test domain(Scale((0.5, 2)), codomain(Scale((0.5, 2)), mt)) == mt

        # D₄: the quarter turn exchanges the axes, so both must carry the same scale up to a sign
        t = d4sym(F ⊗ F)
        t₂ = d4sym(Fourier(4, 1.0) ⊗ Fourier(4, 1.0))
        e = Sequence(t, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        for (γ, codom) ∈ (((2, 2), t₂), ((-1, -1), t), ((-1, 1), t), ((2, -2), t₂))
            @test codomain(Scale(γ), t) == codom
            @test scale_commutes(e, γ)
        end
        # unequal scales break the quarter turn and only the axis reflections survive
        t21 = codomain(Scale((2, 1)), t)
        @test desymmetrize(t21) == Fourier(4, 1.0) ⊗ F
        @test length(elements(symmetry(t21))) == 4
        @test all(g -> g.lattice_aut.matrix[1,2] == g.lattice_aut.matrix[2,1] == 0, elements(symmetry(t21)))
        @test scale_commutes(e, (2, 1))
        @test domain(Scale((2, 1)), t21) == SymmetricSpace(desymmetrize(t), symmetry(t21))

        # the dilation does not see the frequencies, so axes of different frequencies may be exchanged
        tω = SymmetricSpace(F ⊗ Fourier(2, 2.0), swap)
        @test codomain(Scale((2, 2)), tω) == SymmetricSpace(Fourier(4, 1.0) ⊗ Fourier(4, 2.0), swap)
        @test scale_commutes(decaying(tω), (2, 2))
        @test symmetry(codomain(Scale((2, 1)), tω)) == RadiiPolynomial._trivial_symmetry(Val(2))

        # Taylor axes exchanged by the group must carry the same scale
        tt = SymmetricSpace(Taylor(2) ⊗ Taylor(2), swap)
        @test codomain(Scale((0.5, 0.5)), tt) == tt
        @test symmetry(codomain(Scale((0.5, 2.0)), tt)) == RadiiPolynomial._trivial_symmetry(Val(2))
        @test scale_commutes(Sequence(tt, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), (0.5, 0.5))

        # materializing `Scale` on a symmetric space averages over the orbits, in floating point
        s = evensym(Fourier(2, exact(1.0)))
        @test eltype(Projection(s) * Scale(-1)) == Float64
        @test Projection(s) * Scale(-1) == project(Scale(-1), s, s, Float64)
    end

    @testset "InfiniteSequence" begin
        geo(ν...) = Ell1(length(ν) == 1 ? GeometricWeight(ν[1]) : map(GeometricWeight, ν))
        rates(a) = map(rate, weight(banachspace(a)))
        isometry(a, b) = norm(sequence(b), banachspace(b)) ≈ norm(sequence(a), banachspace(a))
        errors(a) = (finite_error(a), tail_error(a), total_error(a))

        # Fourier: the dilation k ↦ γk pulls the rate back to ν^{1/|γ|}
        X = geo(4.0)
        a = InfiniteSequence(Sequence(Fourier(2, 1.0), ComplexF64[1, 2, 3, 2, 1]), 0.1, 0.2, 0.3, X)
        b = scale(a, 2)
        @test sequence(b) == scale(sequence(a), 2)
        @test errors(b) == (0.1, 0.2, 0.3)
        @test rate(weight(banachspace(b))) ≈ 2.0
        @test isometry(a, b)
        @test banachspace(scale(a, -1)) == X
        @test isometry(a, scale(a, -1))
        @test sequence(Scale(-1) * a) == sequence(scale(a, -1))
        @test banachspace(scale(InfiniteSequence(sequence(a), 0.1, 0.2, 0.3, Ell1()), 2)) == Ell1()
        @test_throws DomainError scale(InfiniteSequence(sequence(a), 0.1, 0.2, 0.3, Ell1(AlgebraicWeight(1.0))), 2)

        # Taylor: γᵏ multiplies the weight, so the rate becomes ν/|γ| and must stay at least one
        t = InfiniteSequence(Sequence(Taylor(2), [1.0, 2.0, 3.0]), 0.1, 0.2, 0.3, X)
        tb = scale(t, 0.5)
        @test sequence(tb) == scale(sequence(t), 0.5)
        @test rate(weight(banachspace(tb))) == 8.0
        @test isometry(t, tb)
        @test rate(weight(banachspace(scale(t, 4.0)))) == 1.0
        @test_throws DomainError scale(t, 5.0)
        @test banachspace(scale(t, -1)) == X

        # tensor: one weight per axis, a single weight being applied to each axis
        m = InfiniteSequence(Sequence(Fourier(1, 1.0) ⊗ Taylor(1) ⊗ Chebyshev(1), ComplexF64.(1:12)), 0.1, 0.2, 0.3, geo(4.0, 2.0, 3.0))
        mb = scale(m, (2, 0.5, 1))
        @test sequence(mb) == scale(sequence(m), (2, 0.5, 1))
        @test rates(mb) == (2.0, 4.0, 3.0)
        @test isometry(m, mb)
        @test total_error(mb) == 0.3
        ms = InfiniteSequence(Sequence(Fourier(1, 1.0) ⊗ Fourier(1, 1.0), ComplexF64.(1:9)), 0.1, 0.2, 0.3, X)
        @test rates(scale(ms, (2, -1))) == (2.0, 4.0)

        # interval rates stay rigorous
        ai = InfiniteSequence(interval(sequence(a)), interval(0.1), interval(0.2), interval(0.3), geo(interval(4.0)))
        @test in_interval(2.0, rate(weight(banachspace(scale(ai, 2)))))
        @test rate(weight(banachspace(scale(ai, -1)))) === interval(4.0)

        # on a symmetric space the weight is pulled back through the underlying space
        Xs = Ell1((GeometricWeight(1.1), IdentityWeight()))
        s = Chebyshev(2) ⊗ evensym(Fourier(3, 1.0))
        c = scale(InfiniteSequence(Sequence(s, collect(1.0:dimension(s))), Xs; total_error = 0.1), (1, -1))
        @test space(c) == s
        @test banachspace(c) == Xs
        @test total_error(c) == 0.1
    end
end
