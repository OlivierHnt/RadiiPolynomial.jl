@testset "Symmetry" begin

    @testset "LatticeAut" begin
        A1 = LatticeAut([1;;])
        B1 = LatticeAut([-1;;])
        @test A1(5) == 5
        @test B1(5) == -5

        # a 90° rotation acts as (k₁, k₂) ↦ (-k₂, k₁)
        R = LatticeAut([0 -1 ; 1 0])
        @test R((1, 0)) == (0, 1)
        @test R((0, 1)) == (-1, 0)
        @test R((2, 3)) == (-3, 2)

        P3 = LatticeAut([0 1 0 ; 1 0 0 ; 0 0 1])
        @test P3((1, 2, 3)) == (2, 1, 3)

        P4 = LatticeAut([0 1 0 0 ; 1 0 0 0 ; 0 0 0 1 ; 0 0 1 0])
        @test P4((1, 2, 3, 4)) == (2, 1, 4, 3)

        @test R * R == LatticeAut([-1 0 ; 0 -1])
        @test (R * R)((1, 0)) == (-1, 0)

        @test A1 == LatticeAut([1;;])
        @test A1 != B1
        @test hash(A1) == hash(LatticeAut([1;;]))
    end

    @testset "Cocycle" begin
        # the phase, in units of π, is reduced mod 2 at construction
        @test Cocycle(1, Rational{Int}[3//1]).phase == [1//1]
        @test Cocycle(1, Rational{Int}[5//2]).phase == [1//2]
        @test_throws MethodError Cocycle(1, [0.5]) # the phase must be rational

        # v(k) = amplitude * cispi(phase ⋅ k)
        v = Cocycle(2, Rational{Int}[1//2])
        @test v(0) == 2.0 + 0.0im         # cispi(0) = 1
        @test v(1) == 0.0 + 2.0im         # cispi(1/2) = i
        @test v(2) == -2.0 + 0.0im        # cispi(1) = -1

        # an interval amplitude gives enclosures of the same values
        w = Cocycle(interval(1.0), Rational{Int}[1//1])
        @test isequal_interval(real(w(0)), interval(1.0))   # cispi(0) = 1
        @test isequal_interval(imag(w(0)), interval(0.0))
        @test isequal_interval(real(w(1)), interval(-1.0))  # cispi(1) = -1
        @test isequal_interval(imag(w(1)), interval(0.0))

        a = Cocycle(2, Rational{Int}[1//2])
        b = Cocycle(3, Rational{Int}[1//2])

        # a cocycle alone does not compose: α_{gh}(k) = α_g(β_h k)·α_h(k) needs the index
        # action of the second factor, so composition lives on `GroupElement`
        @test_throws MethodError a * b

        @test a == Cocycle(2, Rational{Int}[1//2])
        @test a != b
        @test hash(a) == hash(Cocycle(2, Rational{Int}[1//2]))
    end

    @testset "GroupElement" begin
        g1 = GroupElement(LatticeAut([-1;;]), Cocycle(1, Rational{Int}[0//1]))
        g2 = GroupElement(LatticeAut([-1;;]), Cocycle(-1, Rational{Int}[0//1]))

        @test g1 ∘ g1 == GroupElement(LatticeAut([1;;]), Cocycle(1, Rational{Int}[0//1]))

        @test g1 == g1
        @test g1 != g2
        @test hash(g1) == hash(GroupElement(LatticeAut([-1;;]), Cocycle(1, Rational{Int}[0//1])))

        @testset "∘ reproduces the induced action on coefficients" begin
            #= A group element acts by (g·a)_k = α_g(k) a_{β_g(k)} with α_g(k) = ρ_g e^{iπ⟨φ_g,k⟩}.
               Applying g then h,
                   ((a·g)·h)_k = α_h(k) (a·g)_{β_h k} = α_h(k) α_g(β_h k) a_{β_g β_h k},
               so the composite carries β_g β_h and phase β_hᵀφ_g + φ_h — the phase of the first
               factor is transported by the lattice automorphism of the second. Adding the phases is the
               same thing only when β_hᵀφ_g ≡ φ_g (mod 2), which every shipped symmetry happens
               to satisfy; the pair below deliberately does not. =#
            s = Fourier(2, 1.0) ⊗ Fourier(2, 1.0)
            act(g, a) = Sequence(s, [g.cocycle(k) * a[g.lattice_aut(k)] for k ∈ indices(s)])

            g = GroupElement(LatticeAut([1 0 ; 0 1]),  Cocycle(1, Rational{Int}[1//2, 0//1]))
            h = GroupElement(LatticeAut([0 1 ; 1 0]),  Cocycle(1, Rational{Int}[0//1, 0//1]))
            @test mod.(h.lattice_aut.matrix' * g.cocycle.phase, 2) != g.cocycle.phase

            a = Sequence(s, ComplexF64[(1 + 0.4im) / (1 + sum(abs, k))^2 for k ∈ indices(s)])
            @test coefficients(act(h, act(g, a))) ≈ coefficients(act(g ∘ h, a))

            # and the shipped symmetries are unaffected either way
            for gg ∈ elements(d4sym(s).symmetry), hh ∈ elements(d4sym(s).symmetry)
                @test mod.(hh.lattice_aut.matrix' * gg.cocycle.phase, 2) == gg.cocycle.phase
            end
        end
    end

    @testset "Group" begin
        g = GroupElement(LatticeAut([-1;;]), Cocycle(1, Rational{Int}[0//1]))
        G = Group(g)
        @test length(elements(G)) == 2
        @test GroupElement(LatticeAut([1;;]), Cocycle(1, Rational{Int}[0//1])) ∈ elements(G)

        # a 90° rotation generator needs several closure passes to reach {I, R, R², R³}
        R = GroupElement(LatticeAut([0 -1 ; 1 0]), Cocycle(1, Rational{Int}[0//1, 0//1]))
        CG = Group(R)
        @test length(elements(CG)) == 4
        @test Set(h.lattice_aut((1, 0)) for h ∈ elements(CG)) == Set([(1, 0), (0, 1), (-1, 0), (0, -1)])

        g2 = GroupElement(LatticeAut([-1;;]), Cocycle(-1, Rational{Int}[0//1]))
        G2 = Group(g2)
        @test G == Group(g)
        @test hash(G) == hash(Group(g)) # the hash is precomputed at closure, so it must agree across objects
        @test issubset(G, G)
        @test !issubset(G, G2) # g ∉ G2 since the amplitudes differ

        Gu = union(G, G2)
        @test length(elements(Gu)) == 4 # closure adds g ∘ g2: identity index, amplitude 1×(-1) = -1

        Gi = intersect(G, G2)
        @test length(elements(Gi)) == 1
        @test Gi == Group(GroupElement(LatticeAut([1;;]), Cocycle(1, Rational{Int}[0//1])))

        # equal groups short-circuit: the input is returned as-is, skipping the closure
        @test intersect(G, G) === G
        @test union(G, G) === G
        G′ = Group(g) # equal value, distinct object
        @test intersect(G, G′) === G
        @test union(G, G′) === G

        # a group cannot mix group elements acting on indices of different length
        h2 = GroupElement(LatticeAut([1 0 ; 0 1]), Cocycle(1, Rational{Int}[0//1, 0//1]))
        @test_throws MethodError Group(g, h2)
    end

    @testset "symmetry / desymmetrize on non-symmetric spaces" begin
        for s ∈ (Taylor(2), Fourier(2, 1.0), Chebyshev(2), Taylor(1) ⊗ Fourier(1, 1.0))
            G = symmetry(s)
            @test length(elements(G)) == 1
            g = first(elements(G))
            @test desymmetrize(s) == s
        end

        @test desymmetrize(ScalarSpace()) == ScalarSpace()
        @test desymmetrize(CartesianPower(evensym(Fourier(2, 1.0)), 3)) == CartesianPower(Fourier(2, 1.0), 3)
        @test desymmetrize(evensym(Fourier(2, 1.0)) × Taylor(1)) == Fourier(2, 1.0) × Taylor(1)
    end

    @testset "evensym / oddsym — Taylor" begin
        # coefficient-only symmetry: aₖ ↦ (-1)^k aₖ (evensym) or -(-1)^k aₖ (oddsym); index untouched
        @test indices(evensym(Taylor(4))) == 0:2:4
        @test dimension(evensym(Taylor(4))) == 3
        @test indices(oddsym(Taylor(4))) == 1:2:3
        @test dimension(oddsym(Taylor(4))) == 2

        a = Sequence(evensym(Taylor(4)), [1.0, 2.0, 3.0]) # a₀=1, a₂=2, a₄=3
        @test coefficients(project(a, desymmetrize(space(a)))) == [1.0, 0.0, 2.0, 0.0, 3.0]

        b = Sequence(oddsym(Taylor(4)), [7.0, 9.0]) # a₁=7, a₃=9
        @test coefficients(project(b, desymmetrize(space(b)))) == [0.0, 7.0, 0.0, 9.0, 0.0]

        # evensym/oddsym are only defined for a base space, not for a tensor product
        @test_throws MethodError evensym(Taylor(2) ⊗ Fourier(1, 1.0))
        @test_throws MethodError oddsym(Taylor(2) ⊗ Fourier(1, 1.0))
    end

    @testset "evensym / oddsym — Fourier" begin
        # index-flipping symmetry: aₖ ↦ a₋ₖ (evensym) or aₖ ↦ -a₋ₖ (oddsym)
        @test indices(evensym(Fourier(2, 1.0))) == 0:2 # k=0 maps to itself and is self-consistent
        @test dimension(evensym(Fourier(2, 1.0))) == 3
        @test indices(oddsym(Fourier(2, 1.0))) == 1:2  # k=0 excluded since -a₀ = a₀ forces a₀ = 0
        @test dimension(oddsym(Fourier(2, 1.0))) == 2

        a = Sequence(evensym(Fourier(2, 1.0)), [1.0, 2.0, 3.0]) # a₀=1, a₁=2, a₂=3
        @test coefficients(project(a, desymmetrize(space(a)))) == [3.0, 2.0, 1.0, 2.0, 3.0] # a₋₂=a₂, a₋₁=a₁

        b = Sequence(oddsym(Fourier(2, 1.0)), [10.0, 20.0]) # a₁=10, a₂=20
        @test coefficients(project(b, desymmetrize(space(b)))) == [-20.0, -10.0, 0.0, 10.0, 20.0] # a₋ₖ=-aₖ, a₀=0

        c = Sequence(evensym(Fourier(2, 1.0)), ComplexF64[1.0 + 0im, 2.0 + 1im, 3.0 - 2im])
        @test coefficients(project(c, desymmetrize(space(c)))) ==
            ComplexF64[3.0 - 2.0im, 2.0 + 1.0im, 1.0 + 0.0im, 2.0 + 1.0im, 3.0 - 2.0im]

        # with interval coefficients, the real expansion must lie inside the enclosure
        si = interval(evensym(Fourier(2, 1.0)))
        ai = Sequence(si, interval.([1.0, 2.0, 3.0]))
        full_i = project(ai, desymmetrize(si))
        full_r = project(a, desymmetrize(space(a)))
        @test all(in_interval.(coefficients(full_r), coefficients(full_i)))

        # 0 is not a valid orbit representative in oddsym, so indexing it is out of bounds
        @test_throws BoundsError b[0]

        # evensym/oddsym are only defined for a base space, not for a tensor product
        @test_throws MethodError evensym(Fourier(1, 1.0) ⊗ Fourier(1, 1.0))
    end

    @testset "evensym / oddsym — Chebyshev" begin
        # coefficient-only parity rule: Tₖ(-x) = (-1)^k Tₖ(x)
        @test indices(evensym(Chebyshev(4))) == 0:2:4
        @test dimension(evensym(Chebyshev(4))) == 3
        @test indices(oddsym(Chebyshev(4))) == 1:2:3
        @test dimension(oddsym(Chebyshev(4))) == 2

        a = Sequence(oddsym(Chebyshev(4)), [7.0, 9.0]) # a₁=7, a₃=9
        @test coefficients(project(a, desymmetrize(space(a)))) == [0.0, 7.0, 0.0, 9.0, 0.0]

        @test_throws MethodError d4sym(Chebyshev(2))
    end

    @testset "d4sym — TensorSpace{Fourier,Fourier}" begin
        s = d4sym(Fourier(1, 1.0) ⊗ Fourier(1, 1.0))
        G = symmetry(s)
        @test length(elements(G)) == 8 # dihedral group of the square (4 rotations × 2 reflections)

        # 3 orbits on the 3×3 grid {-1,0,1}²: the centre, the 4 edge-midpoints, the 4 corners
        @test indices(s) == [(0, 0), (1, 0), (1, 1)]
        @test dimension(s) == 3

        a = Sequence(s, [1.0, 2.0, 3.0]) # centre = 1, edge-midpoints = 2, corners = 3
        full = project(a, desymmetrize(s))
        @test full[(0, 0)] == 1.0
        @test full[(1, 0)] == 2.0 && full[(-1, 0)] == 2.0 && full[(0, 1)] == 2.0 && full[(0, -1)] == 2.0
        @test full[(1, 1)] == 3.0 && full[(-1, 1)] == 3.0 && full[(-1, -1)] == 3.0 && full[(1, -1)] == 3.0

        # d4sym is only defined for a tensor product of two Fourier spaces
        @test_throws MethodError d4sym(Fourier(1, 1.0))
        @test_throws MethodError d4sym(Taylor(1) ⊗ Taylor(1))
        @test_throws MethodError d4sym(Taylor(1) ⊗ Fourier(1, 1.0))
    end

    @testset "SymmetricSpace construction / equality / subset / intersect / union" begin
        s = Fourier(4, 1.0)

        # default symmetry is the identity group ⟹ the full space is recovered
        @test SymmetricSpace(s) == SymmetricSpace(s, symmetry(s))
        @test indices(SymmetricSpace(s)) == indices(s)

        ss = evensym(s)
        @test SymmetricSpace(ss) === ss

        seven = evensym(s)
        sodd = oddsym(s)
        @test seven == evensym(s)
        @test issubset(seven, SymmetricSpace(s))

        # union of spaces intersects the symmetry groups: even ∪ odd removes every non-trivial constraint
        u = union(seven, sodd)
        @test indices(u) == -4:4
        @test dimension(u) == 9

        # intersect of spaces unions the symmetry groups: even ∩ odd forces a₀ = -a₀ = 0 *and* aₖ = a₋ₖ = -a₋ₖ
        # for every k, i.e. every coefficient must vanish, so no valid orbit representative remains
        i = intersect(seven, sodd)
        @test dimension(i) == 0

        # merging an extra symmetry onto an already-symmetric space unions the two symmetry
        # groups, giving the same group as the intersection of spaces above: closing
        # {id, (k↦-k, amp=1)} ∪ {id, (k↦-k, amp=-1)} under composition adds the element
        # (k↦-k, amp=1) ∘ (k↦-k, amp=-1) = (k↦k, amp=-1), forcing aₖ = -aₖ = 0 for every k
        merged = SymmetricSpace(seven, symmetry(sodd))
        @test merged isa SymmetricSpace
        @test length(elements(symmetry(merged))) == 4
        @test dimension(merged) == 0
        @test merged == i

        # only a sequence space can be symmetrized, not a scalar space
        @test_throws MethodError SymmetricSpace(ScalarSpace(), symmetry(Fourier(1, 1.0)))
    end

    @testset "orbit cache" begin
        # spaces sharing the `(indices, symmetry)` key reuse the memoized orbit data; the
        # frequency does not enter the orbit computation, so it does not split the key
        s = evensym(Fourier(4, 1.0))
        @test s.rep_idx_action === evensym(Fourier(4, 2.0)).rep_idx_action

        t = d4sym(Fourier(2, 1.0) ⊗ Fourier(2, 1.0))
        t′ = d4sym(Fourier(2, 3.0) ⊗ Fourier(2, 3.0))
        @test t.rep_idx_action === t′.rep_idx_action
        @test indices(t) === indices(t′)

        # a different group on the same indices is a distinct entry
        @test s.rep_idx_action !== oddsym(Fourier(4, 1.0)).rep_idx_action

        # the interval symmetry group is a distinct entry: coefficient actions must stay enclosures
        si = interval(s)
        @test si.rep_idx_action !== s.rep_idx_action
        @test last(first(si.rep_idx_action)) isa Complex{<:Interval}
    end

    @testset "order / frequency / dimension / indices" begin
        s = d4sym(Fourier(3, 2.0) ⊗ Fourier(3, 2.0))
        @test order(s) == (3, 3)
        @test frequency(s) == (2.0, 2.0)
        @test dimension(s) == length(indices(s))
    end

    @testset "tensor product of symmetric spaces" begin
        sE = evensym(Fourier(2, 1.0))
        sO = oddsym(Fourier(2, 1.0))

        # ⊗ tensorizes the symmetry groups: block-diagonal lattice automorphisms, amplitudes
        # multiply, phases concatenate
        tEE = sE ⊗ sE
        @test tEE isa SymmetricSpace
        @test desymmetrize(tEE) == Fourier(2, 1.0) ⊗ Fourier(2, 1.0)
        @test length(elements(symmetry(tEE))) == 4
        @test indices(tEE) == [(k, l) for l ∈ 0:2 for k ∈ 0:2] # column-major, as for a plain `TensorSpace`
        a = Sequence(tEE, rand(dimension(tEE)))
        A = Projection(desymmetrize(tEE)) * a
        @test A[(1, 2)] == A[(-1, 2)] == A[(1, -2)] == A[(-1, -2)]

        # odd ⊗ odd: the k = 0 and l = 0 modes are invalid, the pair element (−1)·(−1) has
        # amplitude +1, and the expansion is antisymmetric in each factor separately
        tOO = sO ⊗ sO
        @test indices(tOO) == [(1, 1), (2, 1), (1, 2), (2, 2)]
        b = Sequence(tOO, rand(dimension(tOO)))
        B = Projection(desymmetrize(tOO)) * b
        @test B[(1, 2)] == -B[(-1, 2)]
        @test B[(1, 2)] == B[(-1, -2)]
        @test B[(0, 1)] == 0

        # mixing with a plain space lifts the trivial symmetry group on that factor
        tEF = sE ⊗ Fourier(2, 1.0)
        @test dimension(tEF) == 3 * 5
        c = Sequence(tEF, rand(dimension(tEF)))
        C = Projection(desymmetrize(tEF)) * c
        @test C[(1, 2)] == C[(-1, 2)]
        @test C[(1, -2)] != C[(1, 2)]
        tFE = Fourier(2, 1.0) ⊗ sE
        @test dimension(tFE) == 5 * 3

        # chains flatten through the desymmetrized tensor product
        t3 = sE ⊗ sE ⊗ Fourier(1, 1.0)
        @test desymmetrize(t3) isa TensorSpace{<:NTuple{3,Fourier}}
        @test dimension(t3) == 3 * 3 * 3

        # tensorizing a multivariate symmetric space: d4sym ⊗ evensym gives a 3D space
        # with the direct-product group of order 8 × 2 = 16
        t4 = d4sym(Fourier(1, 1.0) ⊗ Fourier(1, 1.0)) ⊗ sE
        @test length(elements(symmetry(t4))) == 16
        @test dimension(t4) == 3 * 3

        # interval symmetry groups tensorize with enclosure amplitudes
        tI = interval(sE) ⊗ interval(sE)
        @test last(first(tI.rep_idx_action)) isa Complex{<:Interval}
    end

    @testset "_restrict: inverse of lifting through ⊗ with a NoSymSpace" begin
        sE = evensym(Fourier(2, 1.0))
        sO = oddsym(Fourier(2, 1.0))

        # restricting the lifted group recovers the original group
        lifted = symmetry(Chebyshev(2) ⊗ sE) # trivial ⊗ G
        @test RadiiPolynomial._restrict(lifted, Val(1)) == symmetry(sE)
        lifted2 = symmetry(Chebyshev(1) ⊗ (sE ⊗ sO)) # trivial ⊗ (G₁ ⊗ G₂)
        @test RadiiPolynomial._restrict(lifted2, Val(1)) == symmetry(sE ⊗ sO)
        lifted3 = symmetry((Chebyshev(1) ⊗ Chebyshev(1)) ⊗ sE)
        @test RadiiPolynomial._restrict(lifted3, Val(2)) == symmetry(sE)

        # a group mixing the leading and trailing indices cannot be restricted
        d4 = symmetry(d4sym(Fourier(1, 1.0) ⊗ Fourier(1, 1.0))) # swaps the two factors
        @test_throws ArgumentError RadiiPolynomial._restrict(d4, Val(1))
    end

    @testset "_reps_indices" begin
        @test RadiiPolynomial._reps_indices(Int[]) === 1:1:0
        @test RadiiPolynomial._reps_indices([5]) === 5:1:5
        @test RadiiPolynomial._reps_indices([1, 2, 3]) === 1:1:3
        @test RadiiPolynomial._reps_indices([2, 4, 6]) === 2:2:6
        # valid 1D representatives always form an arithmetic progression (an intersection of
        # congruence classes, minus possibly the endpoint 0 fixed by the reflections); anything
        # else means the symmetry group data is inconsistent and must fail loudly
        @test_throws ArgumentError RadiiPolynomial._reps_indices([0, 1, 3])
        @test RadiiPolynomial._reps_indices([(0, 0), (1, 0)]) == [(0, 0), (1, 0)]
    end

    @testset "canonical construction types" begin
        # the runtime type parameters are canonical — determined by the space and the group,
        # never by the data — so repeated constructions always yield the same concrete type;
        # in one variable the representatives are always a `StepRange`
        GE = symmetry(evensym(Fourier(4, 1.0)))
        sE = SymmetricSpace(Fourier(4, 1.0), GE)
        @test indices(sE) isa StepRange{Int,Int}
        @test sE == evensym(Fourier(4, 1.0))
        @test typeof(sE) === typeof(evensym(Fourier(4, 1.0)))

        Gd = symmetry(d4sym(Fourier(1, 1.0) ⊗ Fourier(1, 1.0)))
        t = SymmetricSpace(Fourier(1, 1.0) ⊗ Fourier(1, 1.0), Gd)
        @test indices(t) isa Vector{NTuple{2,Int}}
        @test t == d4sym(Fourier(1, 1.0) ⊗ Fourier(1, 1.0))
        @test typeof(t) === typeof(d4sym(Fourier(1, 1.0) ⊗ Fourier(1, 1.0)))

        si = SymmetricSpace(Fourier(4, interval(1.0)), interval(GE))
        @test last(first(si.rep_idx_action)) isa Complex{<:Interval}
    end

    @testset "_orbit" begin
        G = symmetry(evensym(Fourier(2, 1.0)))
        @test RadiiPolynomial._orbit(G, 1) == Set([1, -1])
        @test RadiiPolynomial._orbit(G, 0) == Set([0])

        Gd = symmetry(d4sym(Fourier(1, 1.0) ⊗ Fourier(1, 1.0)))
        @test RadiiPolynomial._orbit(Gd, (1, 0)) == Set([(1, 0), (0, 1), (-1, 0), (0, -1)])
        @test RadiiPolynomial._orbit(Gd, (0, 0)) == Set([(0, 0)])
    end

    @testset "_findindex_constant / _iscompatible" begin
        @test RadiiPolynomial._findindex_constant(evensym(Fourier(4, 1.0))) == 0
        @test RadiiPolynomial._findindex_constant(oddsym(Fourier(4, 1.0))) === nothing # 0 is not a valid index

        # compatibility ignores the symmetry group entirely, only the underlying space matters
        @test RadiiPolynomial._iscompatible(evensym(Fourier(4, 1.0)), Fourier(4, 1.0))
        @test RadiiPolynomial._iscompatible(Fourier(4, 1.0), evensym(Fourier(4, 1.0)))
        @test RadiiPolynomial._iscompatible(evensym(Fourier(4, 1.0)), oddsym(Fourier(4, 1.0)))
    end

    @testset "interval" begin
        s = evensym(Fourier(2, 1.0))
        si = interval(s)
        @test desymmetrize(si) == Fourier(2, interval(1.0))
        @test length(elements(symmetry(si))) == length(elements(symmetry(s)))
        g = first(elements(symmetry(si)))
        @test isequal_interval(g.cocycle.amplitude, interval(1.0))

        siT = interval(Float64, s)
        @test siT isa SymmetricSpace
        @test desymmetrize(siT) == Fourier(2, interval(1.0))
        @test siT == si
    end

end
