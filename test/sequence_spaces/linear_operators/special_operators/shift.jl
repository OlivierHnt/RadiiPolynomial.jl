@testset "Shift" begin
    @testset "constructors and value accessor" begin
        𝒮 = Shift(1.0)
        @test 𝒮 isa Shift{Float64}
        @test value(𝒮) == 1.0

        𝒮₂ = Shift(1.0, 2.0)
        @test 𝒮₂ == Shift((1.0, 2.0))
        @test value(𝒮₂) == (1.0, 2.0)

        @test_throws ArgumentError Shift(())
        @test_throws ArgumentError Shift()
    end

    @testset "arithmetic on Shift operators" begin
        @test Shift(0.3) * Shift(0.7) == Shift(1.0)
        @test Shift((0.3, 0.5)) * Shift((0.2, 0.5)) == Shift((0.5, 1.0))
        @test Shift(0.3)^2 == Shift(0.6)
        @test Shift((0.3, 0.5))^2 == Shift((0.6, 1.0))
        @test Shift((0.3, 0.5))^(2, 3) == Shift((0.6, 1.5))
    end

    @testset "Fourier: the value is measured in periods" begin
        a = Sequence(Fourier(1, 1.0), [0.5, 0.0, 0.5]) # cos(t)
        τ = 0.25 # a quarter period
        𝒮 = Shift(τ)
        @test domain(𝒮, Fourier(1, 1.0)) == Fourier(1, 1.0)
        @test codomain(𝒮, Fourier(1, 1.0)) == Fourier(1, 1.0)
        # cₖ = e^{2iπτk} aₖ: c₋₁ = e^{-iπ/2}(0.5) = -0.5i, c₀ = 0, c₁ = e^{iπ/2}(0.5) = 0.5i
        expected = Sequence(Fourier(1, 1.0), ComplexF64[-0.5im, 0.0, 0.5im])
        out = Sequence(Fourier(1, 1.0), fill(complex(Inf), 3))
        @test shift(a, τ) ≈ 𝒮(a) ≈ project(𝒮, Fourier(1, 1.0), Fourier(1, 1.0), ComplexF64)(a) ≈
            shift!(copy(out), a, τ) ≈ mul!(out, 𝒮, a) ≈ expected

        # τ = 0: identity
        @test shift(a, 0.0) == a

        # shift! requires the destination space to match codomain(Shift(τ), space(a))
        @test_throws ArgumentError shift!(Sequence(Fourier(2, 1.0), fill(complex(Inf), 5)), a, τ)

        # the frequency plays no role: a whole period leaves the sequence unchanged
        s = Fourier(3, 2.0)
        b = Sequence(s, ComplexF64[0.1-0.1im, 0.2, 0.3+0.2im, 0.5, 0.3-0.2im, 0.2, 0.1+0.1im])
        @test shift(b, 1.0) ≈ b
        @test shift(b, 1/3) ≈ Sequence(s, [cispi(2k/3) * b[k] for k ∈ indices(s)])

        # Integer and Rational values give exact roots of unity: a quarter period is the
        # multiplication by iᵏ, a whole period is the identity
        @test shift(b, 1//4) == project(Shift(1//4), s, s, ComplexF64)(b) == Sequence(s, [(1.0im)^k * b[k] for k ∈ indices(s)])
        @test shift(b, 1) == (Shift(1//4)^4)(b) == shift(b, 0) == b
        @test shift(b, 1//3) ≈ shift(b, 1/3)
    end

    @testset "Taylor: only τ = 0 is supported" begin
        a = Sequence(Taylor(2), [1.0, 2.0, 3.0])
        𝒮 = Shift(0.0)
        @test domain(𝒮, Taylor(2)) == Taylor(2)
        @test codomain(𝒮, Taylor(2)) == Taylor(2)
        out = Sequence(Taylor(2), fill(Inf, 3))
        @test shift(a, 0.0) == 𝒮(a) == project(𝒮, Taylor(2), Taylor(2), Float64)(a) ==
            shift!(copy(out), a, 0.0) == mul!(out, 𝒮, a) == a

        @test_throws DomainError shift(a, 1.0)
    end

    @testset "Chebyshev: only τ = 0 is supported" begin
        a = Sequence(Chebyshev(2), [1.0, 2.0, 3.0])
        𝒮 = Shift(0)
        @test domain(𝒮, Chebyshev(2)) == Chebyshev(2)
        @test codomain(𝒮, Chebyshev(2)) == Chebyshev(2)
        out = Sequence(Chebyshev(2), fill(Inf, 3))
        @test shift(a, 0) == 𝒮(a) == project(𝒮, Chebyshev(2), Chebyshev(2), Float64)(a) ==
            shift!(copy(out), a, 0) == mul!(out, 𝒮, a) == a

        @test_throws DomainError shift(a, 1.0)
    end

    @testset "TensorSpace: tuple form per factor" begin
        # Taylor factor left untouched (τ = 0), Fourier factor rotated
        s = Taylor(0) ⊗ Fourier(1, 1.0)
        a = Sequence(s, ComplexF64[1.0, 2.0, 3.0]) # indices (0,-1),(0,0),(0,1)
        τ = (0.0, 0.25)
        c = shift(a, τ)
        # (0,-1): 1*e^{-iπ/2} = -i ; (0,0): 2*1 = 2 ; (0,1): 3*e^{iπ/2} = 3i
        expected = Sequence(s, ComplexF64[-1.0im, 2.0, 3.0im])
        @test c ≈ expected
        @test project(Shift(τ), s, s, ComplexF64)(a) ≈ c

        # both factors rotated
        s2 = Fourier(1, 1.0) ⊗ Fourier(1, 3.0)
        b = Sequence(s2, ComplexF64.(1:9))
        τ2 = (0.1, 0.2)
        cb = shift(b, τ2)
        # c[(k₁,k₂)] = e^{2iπk₁τ₁} e^{2iπk₂τ₂} a[(k₁,k₂)]
        expected2 = Sequence(s2, [b[(k1,k2)]*cispi(2k1*τ2[1])*cispi(2k2*τ2[2]) for (k1,k2) ∈ indices(s2)])
        @test cb ≈ expected2
        @test project(Shift(τ2), s2, s2, ComplexF64)(b) ≈ cb

        # exact values
        τ3 = (1//2, 1//4)
        expected3 = Sequence(s2, [(-1)^k1 * (1.0im)^k2 * b[(k1,k2)] for (k1,k2) ∈ indices(s2)])
        @test shift(b, τ3) == project(Shift(τ3), s2, s2, ComplexF64)(b) == expected3
    end

    @testset "interval τ" begin
        τ = interval(0.0)
        a = Sequence(Fourier(1, 1.0), Complex{Interval{Float64}}.([0.5, 0.0, 0.5]))
        @test shift(a, τ) == Shift(τ)(a) == a
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
        out = Sequence(s3, fill(complex(Inf), dimension(s3)))
        @test c ≈ project(𝒮, sE, s3, ComplexF64)(a) ≈ shift!(copy(out), a, 1//3) ≈ mul!(out, 𝒮, a)

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
            cm = Shift(rm) * Sequence(m, [1.0 / (1 + sum(abs, k))^2 for k ∈ indices(m)])
            @test space(cm) == s3 ⊗ sC
            @test domain(Shift(rm), space(cm)) == m
            @test project(cm, desymmetrize(m)) ≈ shift(project(Sequence(m, [1.0 / (1 + sum(abs, k))^2 for k ∈ indices(m)]), desymmetrize(m)), rm)
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
        @test_throws DomainError codomain(Shift(interval(1//3, 1//2)), sE) # a set of shifts has no single group
        @test_throws DomainError codomain(Shift(interval(1.0) / 3), sE) # floating point endpoints are not exact thirds
    end
end
