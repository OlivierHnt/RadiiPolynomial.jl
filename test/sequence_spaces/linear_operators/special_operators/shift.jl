@testset "Shift" begin
    test_shift(a, τ, expected; cmp = ==) = test_action(Shift(τ), a, expected; inplace! = (c, a) -> shift!(c, a, τ), cmp = cmp)

    @testset "constructors and arithmetic" begin
        test_valued_operator(Shift)
        @test Shift(0.3) * Shift(0.7) == Shift(1.0)
        @test Shift((0.3, 0.5)) * Shift((0.2, 0.5)) == Shift((0.5, 1.0))
        @test Shift(0.3)^2 == Shift(0.6)
        @test Shift((0.3, 0.5))^2 == Shift((0.6, 1.0))
        @test Shift((0.3, 0.5))^(2, 3) == Shift((0.6, 1.5))
    end

    @testset "Fourier: cₖ = e^{2iπτk} aₖ, the value being measured in periods" begin
        ℱ = Fourier(1, 1.0)
        a = Sequence(ℱ, [0.5, 0.0, 0.5]) # cos(t)
        @test domain(Shift(0.25), ℱ) == codomain(Shift(0.25), ℱ) == ℱ
        expected = Sequence(ℱ, ComplexF64[-0.5im, 0.0, 0.5im])
        test_shift(a, 0.25, expected; cmp = ≈)
        @test shift(a, 0.25) ≈ expected
        @test shift(a, 0.0) == a
        @test_throws ArgumentError shift!(fill(complex(Inf), Fourier(2, 1.0)), a, 0.25)

        # the frequency plays no role
        s = Fourier(3, 2.0)
        b = Sequence(s, ComplexF64[0.1-0.1im, 0.2, 0.3+0.2im, 0.5, 0.3-0.2im, 0.2, 0.1+0.1im])
        @test shift(b, 1.0) ≈ b
        @test shift(b, 1/3) ≈ Sequence(s, [cispi(2k/3) * b[k] for k ∈ indices(s)])

        # Integer and Rational values give exact roots of unity
        @test shift(b, 1//4) == project(Shift(1//4), s, s, ComplexF64)(b) == Sequence(s, [(1.0im)^k * b[k] for k ∈ indices(s)])
        @test shift(b, 1) == (Shift(1//4)^4)(b) == shift(b, 0) == b
        @test shift(b, 1//3) ≈ shift(b, 1/3)

        τ = interval(0.0)
        aᵢ = Sequence(ℱ, Complex{Interval{Float64}}.([0.5, 0.0, 0.5]))
        @test shift(aᵢ, τ) == Shift(τ)(aᵢ) == aᵢ
    end

    @testset "$s: only τ = 0 is supported" for (s, τ) ∈ ((Taylor(2), 0.0), (Chebyshev(2), 0))
        a = Sequence(s, [1.0, 2.0, 3.0])
        @test domain(Shift(τ), s) == codomain(Shift(τ), s) == s
        test_shift(a, τ, copy(a))
        @test shift(a, τ) == a
        @test_throws DomainError shift(a, 1.0)
    end

    @testset "TensorSpace: one value per factor" begin
        # the Taylor factor only admits τ = 0
        s = Taylor(0) ⊗ Fourier(1, 1.0)
        a = Sequence(s, ComplexF64[1.0, 2.0, 3.0])
        @test shift(a, (0.0, 0.25)) ≈ project(Shift((0.0, 0.25)), s, s, ComplexF64)(a) ≈ Sequence(s, ComplexF64[-1.0im, 2.0, 3.0im])

        s2 = Fourier(1, 1.0) ⊗ Fourier(1, 3.0)
        b = Sequence(s2, ComplexF64.(1:9))
        τ = (0.1, 0.2)
        @test shift(b, τ) ≈ project(Shift(τ), s2, s2, ComplexF64)(b) ≈
            Sequence(s2, [b[(k1,k2)]*cispi(2k1*τ[1])*cispi(2k2*τ[2]) for (k1,k2) ∈ indices(s2)])

        τ = (1//2, 1//4)
        @test shift(b, τ) == project(Shift(τ), s2, s2, ComplexF64)(b) ==
            Sequence(s2, [(-1)^k1 * (1.0im)^k2 * b[(k1,k2)] for (k1,k2) ∈ indices(s2)])
    end

    @testset "Symmetric space" begin
        F = Fourier(4, 1.0)
        sE = evensym(F)
        a = Sequence(sE, [1.0, 0.5, 0.2, 0.1, 0.05])
        𝒮 = Shift(1//3)

        # only an Integer or Rational number of periods keeps the symmetry group exact
        @test_throws DomainError Shift(1/3) * a
        @test_throws DomainError domain(Shift(1/3), sE)

        # the shift conjugates the reflection k ↦ -k by the phase e^{4iπk/3}: a_{-k} = a_k turns into
        # c_{-k} = e^{-4iπk/3} c_k since c_k = e^{2iπk/3} a_k
        s3 = SymmetricSpace(F, Group(GroupElement(LatticeAut([-1;;]), Cocycle(0//1, Rational{Int}[4//3]))))
        @test codomain(𝒮, sE) == s3
        @test domain(𝒮, s3) == sE
        @test domain(𝒮, sE) == SymmetricSpace(F, Group(GroupElement(LatticeAut([-1;;]), Cocycle(0//1, Rational{Int}[-4//3]))))
        @test codomain(Shift(1//2), sE) == sE # e^{2iπk} = 1

        c = 𝒮 * a
        @test space(c) == s3
        test_shift(a, 1//3, c; cmp = ≈)

        # the shift is the group element with the identity lattice automorphism and the phase 2/3
        g = GroupElement(LatticeAut([1;;]), Cocycle(0//1, Rational{Int}[2//3]))
        act = [cispi(RadiiPolynomial._angle(g.cocycle, k)) * RadiiPolynomial.getcoefficient(a, (F, g.lattice_aut(k))) for k ∈ indices(F)]
        @test coefficients(project(c, F)) ≈ act
        @test project(c, F) ≈ shift(project(a, F), 1//3)

        # a quarter period is exact and a full period is the identity
        @test project(Shift(1//4) * a, F) == Sequence(F, [(1.0im)^k * RadiiPolynomial.getcoefficient(a, (F, k)) for k ∈ indices(F)])
        @test Shift(1) * a == (Shift(1//3)^3)(a) == a

        # tensor: the phase moves by 2(I - βᵀ)r for every element of the group
        s2 = Fourier(2, 1.0) ⊗ Fourier(2, 1.0)
        sd = d4sym(s2)
        b = Sequence(sd, [1.0 / (1 + sum(abs, k))^2 for k ∈ indices(sd)])
        r = (1//4, 1//4)
        cb = Shift(r) * b
        for g ∈ elements(symmetry(sd))
            β = g.lattice_aut.matrix
            @test GroupElement(g.lattice_aut, Cocycle(0//1, 2 * (I - β') * Rational{Int}[r...])) ∈ elements(symmetry(space(cb)))
        end
        @test domain(Shift(r), space(cb)) == sd
        @test project(cb, s2) ≈ shift(project(b, s2), r)
        @test project(Shift(r), sd, space(cb), ComplexF64)(b) ≈ cb

        # mixed tensor: a Taylor or Chebyshev axis admits only the zero shift and keeps its symmetry
        for sC ∈ (evensym(Chebyshev(2)), oddsym(Taylor(3)))
            m = evensym(F) ⊗ sC
            rm = (1//3, 0)
            am = Sequence(m, [1.0 / (1 + sum(abs, k))^2 for k ∈ indices(m)])
            cm = Shift(rm) * am
            @test space(cm) == s3 ⊗ sC
            @test domain(Shift(rm), space(cm)) == m
            @test project(cm, desymmetrize(m)) ≈ shift(project(am, desymmetrize(m)), rm)
            @test_throws DomainError codomain(Shift((1//3, 1)), m)
            @test_throws DomainError codomain(Shift((1//3, 1//2)), m)
            @test_throws DomainError codomain(Shift((0.5, 0)), m)
        end
        @test codomain(Shift(0), evensym(Chebyshev(2))) == evensym(Chebyshev(2))
        @test_throws DomainError codomain(Shift(1), evensym(Chebyshev(2)))

        # an exact or a thin rational interval number of periods is as good as a rational one
        @test codomain(Shift(exact(1//3)), sE) == codomain(Shift(interval(1//3)), sE) == s3
        @test codomain(Shift(exact(1)), sE) == codomain(Shift(interval(1//1)), sE) == sE
        @test codomain(Shift((exact(1//4), exact(1//4))), sd) == space(cb)
        @test all(in_interval.(coefficients(shift(project(a, F), 1//3)), coefficients(project(Shift(interval(1//3)) * interval(a), F))))
        # a set of shifts has no single group, and floating point endpoints are not exact thirds
        @test_throws DomainError codomain(Shift(interval(1//3, 1//2)), sE)
        @test_throws DomainError codomain(Shift(interval(1.0) / 3), sE)
    end

    @testset "InfiniteSequence" begin
        X = Ell1(GeometricWeight(2.0))
        a = InfiniteSequence(Sequence(Fourier(2, 1.0), ComplexF64[1, 2, 3, 2, 1]), 0.1, 0.2, 0.3, X)
        b = shift(a, 1//3)
        @test sequence(b) == shift(sequence(a), 1//3)
        @test (finite_error(b), tail_error(b), total_error(b)) == (0.1, 0.2, 0.3)
        @test banachspace(b) == X # the shift is an isometry
        @test norm(sequence(b), X) ≈ norm(sequence(a), X)
        @test sequence(Shift(1//3) * a) == sequence(b)
        m = InfiniteSequence(Sequence(Fourier(1, 1.0) ⊗ Chebyshev(1), ComplexF64[1, 2, 3, 4, 5, 6]), 0.1, 0.2, 0.3, Ell1((GeometricWeight(2.0), GeometricWeight(1.5))))
        bm = shift(m, (0.25, 0))
        @test sequence(bm) == shift(sequence(m), (0.25, 0))
        @test banachspace(bm) == banachspace(m)
        @test total_error(bm) == 0.3
    end
end
