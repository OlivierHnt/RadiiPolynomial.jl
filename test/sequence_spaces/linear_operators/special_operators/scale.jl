@testset "Scale" begin
    @testset "constructors and value accessor" begin
        𝒮 = Scale(2.0)
        @test 𝒮 isa Scale{Float64}
        @test value(𝒮) == 2.0

        𝒮₂ = Scale(2.0, 3.0)
        @test 𝒮₂ == Scale((2.0, 3.0))
        @test value(𝒮₂) == (2.0, 3.0)

        @test_throws ArgumentError Scale(())
        @test_throws ArgumentError Scale()
    end

    @testset "arithmetic on Scale operators" begin
        @test Scale(2.0) * Scale(3.0) == Scale(6.0)
        @test Scale((2.0, 3.0)) * Scale((4.0, 5.0)) == Scale((8.0, 15.0))
        @test Scale(2.0)^3 == Scale(8.0)
        @test Scale((2.0, 3.0))^2 == Scale((4.0, 9.0))
        @test Scale((2.0, 3.0))^(2, 3) == Scale((4.0, 27.0))
    end

    @testset "Taylor: coefficient rescaling γᵏ" begin
        a = Sequence(Taylor(3), [1.0, 2.0, 4.0, 8.0])
        γ = 0.5
        𝒮 = Scale(γ)
        @test domain(𝒮, Taylor(3)) == Taylor(3)
        @test codomain(𝒮, Taylor(3)) == Taylor(3)
        # cₖ = aₖ γᵏ = 2ᵏ (1/2)ᵏ = 1
        expected = Sequence(Taylor(3), [1.0, 1.0, 1.0, 1.0])
        out = Sequence(Taylor(3), fill(Inf, 4))
        @test scale(a, γ) == 𝒮(a) == project(𝒮, Taylor(3), Taylor(3), Float64)(a) ==
            scale!(copy(out), a, γ) == mul!(out, 𝒮, a) == expected

        # γ = 1 is the identity
        @test scale(a, 1.0) == a

        # γ > 1: growth
        b = Sequence(Taylor(2), [1.0, 1.0, 1.0])
        @test scale(b, 2.0) == Sequence(Taylor(2), [1.0, 2.0, 4.0])

        # scale! requires the destination space to match codomain(Scale(γ), space(a))
        @test_throws ArgumentError scale!(Sequence(Taylor(2), fill(Inf, 3)), a, γ)
    end

    @testset "Fourier: decimation/expansion by an integer γ" begin
        a = Sequence(Fourier(1, 1.0), [0.5, 1.0, 0.5])
        γ = 2
        𝒮 = Scale(γ)
        @test domain(𝒮, Fourier(4, 1.0)) == Fourier(2, 1.0)
        @test codomain(𝒮, Fourier(2, 1.0)) == Fourier(4, 1.0)
        # cₖ = a_{k÷γ} if γ | k, else 0: c₋₂=a₋₁=0.5, c₋₁=0, c₀=a₀=1.0, c₁=0, c₂=a₁=0.5
        expected = Sequence(Fourier(2, 1.0), [0.5, 0.0, 1.0, 0.0, 0.5])
        out = Sequence(Fourier(2, 1.0), fill(Inf, 5))
        @test scale(a, γ) == 𝒮(a) == project(𝒮, Fourier(1, 1.0), Fourier(2, 1.0), Float64)(a) ==
            scale!(copy(out), a, γ) == mul!(out, 𝒮, a) == expected

        # γ = 1 is the identity
        @test scale(a, 1) == a

        # non-integer γ is explicitly guarded
        @test_throws DomainError domain(Scale(0.5), Fourier(2, 1.0))
        @test_throws DomainError codomain(Scale(0.5), Fourier(2, 1.0))
        @test_throws DomainError scale(a, 0.5)

        # an integer-valued `Float64` γ (e.g. 2.0) behaves identically to the
        # literal `Integer` γ
        @test scale(a, 2.0) == expected

        # negative γ: Scale(γ) means b(t) = a(γt), so b_k = a_{k/γ} with signed
        # division; γ = -1 is a time reversal
        @test scale(Sequence(Fourier(1, 1.0), [10.0, 20.0, 30.0]), -1) ==
            Sequence(Fourier(1, 1.0), [30.0, 20.0, 10.0])

        # γ = -2: only even modes survive (odd modes vanish), and the surviving
        # modes are time-reversed
        expected_neg2 = Sequence(Fourier(2, 1.0), [30.0, 0.0, 20.0, 0.0, 10.0])
        @test scale(Sequence(Fourier(1, 1.0), [10.0, 20.0, 30.0]), -2) == expected_neg2
    end

    @testset "Chebyshev: only γ = 1 is supported" begin
        a = Sequence(Chebyshev(2), [1.0, 2.0, 3.0])
        𝒮 = Scale(1.0)
        @test domain(𝒮, Chebyshev(2)) == Chebyshev(2)
        @test codomain(𝒮, Chebyshev(2)) == Chebyshev(2)
        out = Sequence(Chebyshev(2), fill(Inf, 3))
        @test scale(a, 1.0) == 𝒮(a) == project(𝒮, Chebyshev(2), Chebyshev(2), Float64)(a) ==
            scale!(copy(out), a, 1.0) == mul!(out, 𝒮, a) == a

        @test_throws DomainError scale(a, 2.0)

        # the same guard applies when materializing the operator
        @test_throws DomainError project(Scale(2.0), Chebyshev(2), Chebyshev(2), Float64)
    end

    @testset "TensorSpace: tuple form per factor" begin
        s = Taylor(3) ⊗ Taylor(2)
        𝒮 = Scale((0.5, 0.5))
        a = Sequence(s, [2.0 ^ sum(α) for α ∈ indices(s)])
        # cᵢⱼ = aᵢⱼ (0.5)ⁱ(0.5)ʲ = 2^(i+j) (1/2)^(i+j) = 1
        @test 𝒮(a) == project(𝒮, s, s, Float64)(a) == Sequence(s, ones(12))

        # mixed base spaces: Taylor factor rescaled, Chebyshev factor left as identity (γ=1)
        s2 = Taylor(1) ⊗ Chebyshev(1)
        a2 = Sequence(s2, [1.0, 2.0, 3.0, 4.0]) # indices (0,0),(1,0),(0,1),(1,1)
        𝒮2 = Scale((2.0, 1))
        expected2 = Sequence(s2, [1.0, 4.0, 3.0, 8.0])
        @test 𝒮2(a2) == project(𝒮2, s2, s2, Float64)(a2) == expected2

        # Fourier as the *trailing* tensor factor with a genuine decimation, composed
        # with a leading Taylor factor left as identity (γ=1): indices (0,-1),(1,-1),
        # (0,0),(1,0),(0,1),(1,1) ↦ a Fourier order-2 result, only even k surviving
        s3 = Taylor(1) ⊗ Fourier(1, 1.0)
        a3 = Sequence(s3, collect(1.0:6.0))
        expected3 = Sequence(Taylor(1) ⊗ Fourier(2, 1.0), [1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 5.0, 6.0])
        @test scale(a3, (1, 2)) == expected3

        # Taylor as the *trailing* tensor factor with γ = 1
        s4 = Chebyshev(1) ⊗ Taylor(1)
        a4 = Sequence(s4, [1.0, 2.0, 3.0, 4.0])
        @test scale(a4, (1.0, 1.0)) == a4

        # Fourier as the *trailing* tensor factor with γ = 1, composed with a genuine
        # rescaling of the leading Taylor factor
        s5 = Taylor(1) ⊗ Fourier(1, 1.0)
        a5 = Sequence(s5, collect(1.0:6.0))
        # only the Taylor factor is rescaled (cᵢ = aᵢ 2ⁱ); Fourier (γ=1) is untouched
        expected5 = Sequence(s5, [1.0, 4.0, 3.0, 8.0, 5.0, 12.0])
        @test scale(a5, (2.0, 1)) == expected5

        # Chebyshev is the *trailing* tensor factor and γ ≠ 1: still guarded
        s6 = Taylor(1) ⊗ Chebyshev(1)
        a6 = Sequence(s6, [1.0, 2.0, 3.0, 4.0])
        @test_throws DomainError scale(a6, (1.0, 2.0))

        # Fourier as the *trailing* tensor factor with negative γ (time reversal),
        # composed with a leading Taylor factor left as identity (γ=1)
        s7 = Taylor(0) ⊗ Fourier(1, 1.0)
        a7 = Sequence(s7, [10.0, 20.0, 30.0])
        expected7 = Sequence(s7, [30.0, 20.0, 10.0])
        @test scale(a7, (1.0, -1)) == expected7

        # Fourier as a *non-trailing* tensor factor: its scaling permutes/resizes along
        # its own dimension, so each factor must read from a fresh intermediate array
        s8 = Fourier(1, 1.0) ⊗ Taylor(0)
        a8 = Sequence(s8, [10.0, 20.0, 30.0])
        expected8 = Sequence(s8, [30.0, 20.0, 10.0])
        @test scale(a8, (-1, 1.0)) == expected8

        # leading-Fourier decimation γ = 2: only even modes of the codomain are populated
        s9 = Fourier(1, 1.0) ⊗ Taylor(1)
        a9 = Sequence(s9, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        𝒮9 = Scale((2, 3.0))
        # Fourier factor (fast index): k ↦ 2k, odd modes zero; Taylor factor: degree i scaled by 3ⁱ
        @test scale(a9, (2, 3.0)) == project(𝒮9, s9, codomain(𝒮9, s9), Float64) * a9 ==
            Sequence(Fourier(2, 1.0) ⊗ Taylor(1), [1.0, 0.0, 2.0, 0.0, 3.0, 12.0, 0.0, 15.0, 0.0, 18.0])

        # three factors, Fourier first and negative: cross-checked against the materialized matrix
        s10 = Fourier(1, 1.0) ⊗ Taylor(1) ⊗ Chebyshev(1)
        a10 = Sequence(s10, collect(1.0:dimension(s10)))
        𝒮10 = Scale((-2, 3.0, 1))
        @test scale(a10, (-2, 3.0, 1)) == project(𝒮10, s10, codomain(𝒮10, s10), Float64) * a10
    end

    @testset "composition" begin
        a = Sequence(Taylor(2), [1.0, 1.0, 1.0])
        S1, S2 = Scale(2.0), Scale(3.0)
        @test (S1*S2)(a) == S1(S2(a)) == Sequence(Taylor(2), [1.0, 6.0, 36.0])

        b = Sequence(Fourier(1, 1.0), [0.5, 1.0, 0.5])
        T1, T2 = Scale(2), Scale(3)
        @test (T1*T2)(b) == T1(T2(b))
    end

    @testset "interval γ" begin
        γ = interval(0.5)
        a = Sequence(Taylor(3), interval.([1.0, 2.0, 4.0, 8.0]))
        expected = Sequence(Taylor(3), interval.([1.0, 1.0, 1.0, 1.0]))
        @test scale(a, γ) == Scale(γ)(a) == expected
    end

    @testset "ComplexF64 coefficients" begin
        a = Sequence(Taylor(2), ComplexF64[1.0+1.0im, 2.0, 3.0-1.0im])
        # cₖ = aₖ 2ᵏ
        expected = Sequence(Taylor(2), ComplexF64[1.0+1.0im, 4.0, 12.0-4.0im])
        @test scale(a, 2.0) == expected
    end

    @testset "Symmetric space" begin
        F = Fourier(2, 1.0)
        sE = evensym(F)
        sO = oddsym(F)
        a = Sequence(sE, [1.0, 2.0, 3.0])

        # a reflection k ↦ -k keeps the space and flips the phase, a dilation divides it by γ
        @test codomain(Scale(-1), sE) == sE
        @test codomain(Scale(-1), sO) == sO
        @test codomain(Scale(2), sE) == evensym(Fourier(4, 1.0))
        @test domain(Scale(2), evensym(Fourier(4, 1.0))) == sE
        @test project(scale(a, -1), F) == scale(project(a, F), -1)
        @test project(scale(a, 2), Fourier(4, 1.0)) == scale(project(a, F), 2)
        @test project(Scale(2), sE, evensym(Fourier(4, 1.0)), Float64)(a) == scale(a, 2)
        s3 = SymmetricSpace(F, Group(GroupElement(LatticeAut([-1;;]), Cocycle(0//1, Rational{Int}[4//3]))))
        @test codomain(Scale(2), s3) == SymmetricSpace(Fourier(4, 1.0), Group(GroupElement(LatticeAut([-1;;]), Cocycle(0//1, Rational{Int}[2//3]))))
        @test domain(Scale(2), codomain(Scale(2), s3)) == s3
        @test codomain(Scale(-1), s3) == SymmetricSpace(F, Group(GroupElement(LatticeAut([-1;;]), Cocycle(0//1, Rational{Int}[-4//3]))))
        b = Sequence(s3, ComplexF64[1, 2, 3])
        @test project(scale(b, -1), F) ≈ scale(project(b, F), -1)
        @test_throws DomainError codomain(Scale(1//2), sE)
        @test_throws DomainError codomain(Scale(0.5), sE)

        # mixed tensors: a Chebyshev axis only admits γ = 1, a Taylor axis keeps its cocycle
        m = evensym(Chebyshev(2)) ⊗ sE
        c = Sequence(m, [1.0 / (1 + sum(abs, k))^2 for k ∈ indices(m)])
        @test codomain(Scale((1, -1)), m) == m
        @test project(scale(c, (1, -1)), desymmetrize(m)) == scale(project(c, desymmetrize(m)), (1, -1))
        @test codomain(Scale((1, 2)), m) == evensym(Chebyshev(2)) ⊗ evensym(Fourier(4, 1.0))
        @test project(scale(c, (1, 2)), Chebyshev(2) ⊗ Fourier(4, 1.0)) == scale(project(c, desymmetrize(m)), (1, 2))
        @test_throws DomainError codomain(Scale((2, 1)), m)
        mt = oddsym(Taylor(3)) ⊗ sE
        d = Sequence(mt, [1.0 / (1 + sum(abs, k))^2 for k ∈ indices(mt)])
        @test codomain(Scale((0.5, -1)), mt) == mt
        @test project(scale(d, (0.5, -1)), desymmetrize(mt)) ≈ scale(project(d, desymmetrize(mt)), (0.5, -1))
        @test domain(Scale((0.5, 2)), codomain(Scale((0.5, 2)), mt)) == mt

        # D₄: the quarter turn exchanges the axes, so both must carry the same scale
        t = d4sym(Fourier(2, 1.0) ⊗ Fourier(2, 1.0))
        e = Sequence(t, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        @test codomain(Scale((2, 2)), t) == d4sym(Fourier(4, 1.0) ⊗ Fourier(4, 1.0))
        @test codomain(Scale((-1, -1)), t) == t
        @test project(scale(e, (2, 2)), Fourier(4, 1.0) ⊗ Fourier(4, 1.0)) == scale(project(e, desymmetrize(t)), (2, 2))
        # unequal scales break the quarter turn: only the axis reflections survive, as for partial evaluation
        t21 = codomain(Scale((2, 1)), t)
        @test desymmetrize(t21) == Fourier(4, 1.0) ⊗ Fourier(2, 1.0)
        @test length(elements(symmetry(t21))) == 4
        @test all(g -> g.lattice_aut.matrix[1,2] == g.lattice_aut.matrix[2,1] == 0, elements(symmetry(t21)))
        @test project(scale(e, (2, 1)), desymmetrize(t21)) == scale(project(e, desymmetrize(t)), (2, 1))
        @test domain(Scale((2, 1)), t21) == SymmetricSpace(desymmetrize(t), symmetry(t21))
        # a reflection of one axis conjugates the quarter turn into the other one, so D₄ is preserved
        @test codomain(Scale((-1, 1)), t) == t
        @test codomain(Scale((2, -2)), t) == d4sym(Fourier(4, 1.0) ⊗ Fourier(4, 1.0))
        @test project(scale(e, (2, -2)), Fourier(4, 1.0) ⊗ Fourier(4, 1.0)) == scale(project(e, desymmetrize(t)), (2, -2))
        # the dilation does not see the frequencies, so exchanging axes of different frequencies is fine
        tω = SymmetricSpace(Fourier(2, 1.0) ⊗ Fourier(2, 2.0), Group(GroupElement(LatticeAut([0 1; 1 0]), Cocycle(0//1, Rational{Int}[0, 0]))))
        eω = Sequence(tω, [1.0 / (1 + sum(abs, k))^2 for k ∈ indices(tω)])
        @test codomain(Scale((2, 2)), tω) == SymmetricSpace(Fourier(4, 1.0) ⊗ Fourier(4, 2.0), symmetry(tω))
        @test project(scale(eω, (2, 2)), Fourier(4, 1.0) ⊗ Fourier(4, 2.0)) == scale(project(eω, desymmetrize(tω)), (2, 2))
        @test symmetry(codomain(Scale((2, 1)), tω)) == RadiiPolynomial._trivial_symmetry(Val(2))
        # Taylor axes exchanged by the group must carry the same scale
        tt = SymmetricSpace(Taylor(2) ⊗ Taylor(2), Group(GroupElement(LatticeAut([0 1; 1 0]), Cocycle(0//1, Rational{Int}[0, 0]))))
        @test codomain(Scale((0.5, 0.5)), tt) == tt
        @test symmetry(codomain(Scale((0.5, 2.0)), tt)) == RadiiPolynomial._trivial_symmetry(Val(2))
        et = Sequence(tt, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        @test project(scale(et, (0.5, 0.5)), desymmetrize(tt)) == scale(project(et, desymmetrize(tt)), (0.5, 0.5))
    end

    @testset "InfiniteSequence" begin
        # Fourier: the dilation k ↦ γk pulls the rate back to ν^{1/|γ|}, errors unchanged
        X = Ell1(GeometricWeight(4.0))
        a = InfiniteSequence(Sequence(Fourier(2, 1.0), ComplexF64[1, 2, 3, 2, 1]), 0.1, 0.2, 0.3, X)
        b = scale(a, 2)
        @test sequence(b) == scale(sequence(a), 2)
        @test (finite_error(b), tail_error(b), total_error(b)) == (0.1, 0.2, 0.3)
        @test rate(weight(banachspace(b))) ≈ 2.0
        @test norm(sequence(b), banachspace(b)) ≈ norm(sequence(a), X)
        @test banachspace(scale(a, -1)) == X # a reflection is an isometry
        @test norm(sequence(scale(a, -1)), X) ≈ norm(sequence(a), X)
        @test sequence(Scale(-1) * a) == sequence(scale(a, -1))
        @test banachspace(scale(InfiniteSequence(sequence(a), 0.1, 0.2, 0.3, Ell1()), 2)) == Ell1()
        @test_throws DomainError scale(InfiniteSequence(sequence(a), 0.1, 0.2, 0.3, Ell1(AlgebraicWeight(1.0))), 2)

        # Taylor: γ^k multiplies the weight, so the rate becomes ν/|γ| and must stay at least one
        t = InfiniteSequence(Sequence(Taylor(2), [1.0, 2.0, 3.0]), 0.1, 0.2, 0.3, X)
        tb = scale(t, 0.5)
        @test sequence(tb) == scale(sequence(t), 0.5)
        @test rate(weight(banachspace(tb))) == 8.0
        @test norm(sequence(tb), banachspace(tb)) ≈ norm(sequence(t), X)
        @test rate(weight(banachspace(scale(t, 4.0)))) == 1.0
        @test_throws DomainError scale(t, 5.0)
        @test banachspace(scale(t, -1)) == X

        # tensor: one weight per axis, a Chebyshev axis only admits γ = 1
        m = InfiniteSequence(Sequence(Fourier(1, 1.0) ⊗ Taylor(1) ⊗ Chebyshev(1), ComplexF64.(1:12)), 0.1, 0.2, 0.3, Ell1((GeometricWeight(4.0), GeometricWeight(2.0), GeometricWeight(3.0))))
        mb = scale(m, (2, 0.5, 1))
        @test sequence(mb) == scale(sequence(m), (2, 0.5, 1))
        @test map(rate, weight(banachspace(mb))) == (2.0, 4.0, 3.0)
        @test norm(sequence(mb), banachspace(mb)) ≈ norm(sequence(m), banachspace(m))
        @test total_error(mb) == 0.3
        # a single weight on a tensor space is applied to each axis
        ms = InfiniteSequence(Sequence(Fourier(1, 1.0) ⊗ Fourier(1, 1.0), ComplexF64.(1:9)), 0.1, 0.2, 0.3, Ell1(GeometricWeight(4.0)))
        @test map(rate, weight(banachspace(scale(ms, (2, -1))))) == (2.0, 4.0)

        # interval rates stay rigorous
        ai = InfiniteSequence(interval(sequence(a)), interval(0.1), interval(0.2), interval(0.3), Ell1(GeometricWeight(interval(4.0))))
        @test in_interval(2.0, rate(weight(banachspace(scale(ai, 2)))))
        @test rate(weight(banachspace(scale(ai, -1)))) === interval(4.0)
    end
end
