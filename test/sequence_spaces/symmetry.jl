@testset "Symmetry" begin

    @testset "IndexAction" begin
        # N = 1: multiplication by a 1×1 matrix
        A1 = IndexAction([1;;])
        B1 = IndexAction([-1;;])
        @test A1(5) == 5
        @test B1(5) == -5

        # N = 2: 90° rotation matrix [0 -1; 1 0], (k₁,k₂) ↦ (M₁₁k₁+M₁₂k₂, M₂₁k₁+M₂₂k₂)
        R = IndexAction([0 -1 ; 1 0])
        @test R((1, 0)) == (0, 1)
        @test R((0, 1)) == (-1, 0)
        @test R((2, 3)) == (-3, 2)

        # N = 3: permutation matrix swapping the first two coordinates (dedicated method)
        P3 = IndexAction([0 1 0 ; 1 0 0 ; 0 0 1])
        @test P3((1, 2, 3)) == (2, 1, 3)

        # N = 4: generic fallback method (only N = 1, 2, 3 have dedicated methods)
        P4 = IndexAction([0 1 0 0 ; 1 0 0 0 ; 0 0 0 1 ; 0 0 1 0])
        @test P4((1, 2, 3, 4)) == (2, 1, 4, 3)

        # composition is matrix multiplication; R ∘ R = -I (180° rotation)
        @test R * R == IndexAction([-1 0 ; 0 -1])
        @test (R * R)((1, 0)) == (-1, 0)

        # equality / hash
        @test A1 == IndexAction([1;;])
        @test A1 != B1
        @test hash(A1) == hash(IndexAction([1;;]))
    end

    @testset "CoefAction" begin
        # construction reduces the phase mod 2 (a factor of π)
        @test CoefAction(1, Rational{Int}[3//1]).phase == [1//1]  # 3 mod 2 = 1
        @test CoefAction(1, Rational{Int}[5//2]).phase == [1//2]  # 5/2 mod 2 = 1/2
        @test_throws MethodError CoefAction(1, [0.5])  # phase must be a Rational{Int} vector

        # generic (non-Interval) application: v(k) = amplitude * cispi(phase ⋅ k)
        v = CoefAction(2, Rational{Int}[1//2])
        @test v(0) == 2.0 + 0.0im         # cispi(0) = 1
        @test v(1) == 0.0 + 2.0im         # cispi(1/2) = i
        @test v(2) == -2.0 + 0.0im        # cispi(1) = -1

        # Interval-valued amplitude uses the dedicated cispi(interval(...)) branch
        w = CoefAction(interval(1.0), Rational{Int}[1//1])
        @test isequal_interval(real(w(0)), interval(1.0))   # cispi(0) = 1
        @test isequal_interval(imag(w(0)), interval(0.0))
        @test isequal_interval(real(w(1)), interval(-1.0))  # cispi(1) = -1
        @test isequal_interval(imag(w(1)), interval(0.0))

        # composition multiplies amplitudes and adds phases (mod 2)
        a = CoefAction(2, Rational{Int}[1//2])
        b = CoefAction(3, Rational{Int}[1//2])
        ab = a * b
        @test ab.amplitude == 6
        @test ab.phase == [1//1]  # 1/2 + 1/2 = 1

        # equality / hash
        @test a == CoefAction(2, Rational{Int}[1//2])
        @test a != b
        @test hash(a) == hash(CoefAction(2, Rational{Int}[1//2]))
    end

    @testset "GroupElement" begin
        g1 = GroupElement(IndexAction([-1;;]), CoefAction(1, Rational{Int}[0//1]))
        g2 = GroupElement(IndexAction([-1;;]), CoefAction(-1, Rational{Int}[0//1]))

        # composition combines the index and coefficient actions independently; g1 is an involution
        @test g1 ∘ g1 == GroupElement(IndexAction([1;;]), CoefAction(1, Rational{Int}[0//1]))

        # equality / hash
        @test g1 == g1
        @test g1 != g2
        @test hash(g1) == hash(GroupElement(IndexAction([-1;;]), CoefAction(1, Rational{Int}[0//1])))
    end

    @testset "Group" begin
        # a self-inverse generator (reflection) closes into an order-2 group
        g = GroupElement(IndexAction([-1;;]), CoefAction(1, Rational{Int}[0//1]))
        G = Group(g)
        @test length(elements(G)) == 2
        @test GroupElement(IndexAction([1;;]), CoefAction(1, Rational{Int}[0//1])) ∈ elements(G) # identity appears

        # a 90° rotation generator needs several closure passes to reach the cyclic group of order 4: {I,R,R²,R³}
        R = GroupElement(IndexAction([0 -1 ; 1 0]), CoefAction(1, Rational{Int}[0//1, 0//1]))
        CG = Group(R)
        @test length(elements(CG)) == 4
        @test Set(h.index_action((1, 0)) for h ∈ elements(CG)) == Set([(1, 0), (0, 1), (-1, 0), (0, -1)])

        # == / issubset
        g2 = GroupElement(IndexAction([-1;;]), CoefAction(-1, Rational{Int}[0//1]))
        G2 = Group(g2)
        @test G == Group(g) # rebuilding from the same generator gives the same set of elements
        @test hash(G) == hash(Group(g)) # the hash is precomputed at closure, so it must agree across objects
        @test issubset(G, G)
        @test !issubset(G, G2) # g ∉ G2 since the amplitudes differ

        # union of two order-2 groups differing only in the sign of the amplitude of the reflection
        Gu = union(G, G2)
        @test length(elements(Gu)) == 4 # closure adds (g ∘ g2): identity index, amplitude 1×(-1) = -1

        # intersect keeps only the common (identity) element
        Gi = intersect(G, G2)
        @test length(elements(Gi)) == 1
        @test Gi == Group(GroupElement(IndexAction([1;;]), CoefAction(1, Rational{Int}[0//1])))

        # equal groups short-circuit: the input is returned as-is, skipping the closure
        # (safe since groups are never mutated after construction)
        @test intersect(G, G) === G
        @test union(G, G) === G
        G′ = Group(g) # equal value, distinct object
        @test intersect(G, G′) === G
        @test union(G, G′) === G

        # a group cannot mix `GroupElement`s of different dimension N
        h2 = GroupElement(IndexAction([1 0 ; 0 1]), CoefAction(1, Rational{Int}[0//1, 0//1]))
        @test_throws MethodError Group(g, h2)
    end

    @testset "symmetry / desymmetrize on non-symmetric spaces" begin
        # every BaseSpace / TensorSpace carries the trivial (identity) symmetry group
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
        @test indices(evensym(Taylor(4))) == 0:2:4 # even powers only
        @test dimension(evensym(Taylor(4))) == 3
        @test indices(oddsym(Taylor(4))) == 1:2:3  # odd powers only
        @test dimension(oddsym(Taylor(4))) == 2

        a = Sequence(evensym(Taylor(4)), [1.0, 2.0, 3.0]) # a₀=1, a₂=2, a₄=3
        @test coefficients(project(a, desymmetrize(space(a)))) == [1.0, 0.0, 2.0, 0.0, 3.0]

        b = Sequence(oddsym(Taylor(4)), [7.0, 9.0]) # a₁=7, a₃=9
        @test coefficients(project(b, desymmetrize(space(b)))) == [0.0, 7.0, 0.0, 9.0, 0.0]

        # guard rail: evensym/oddsym are only defined for Taylor, not for a TensorSpace
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

        # ComplexF64 coefficients expand the same way, entry by entry
        c = Sequence(evensym(Fourier(2, 1.0)), ComplexF64[1.0 + 0im, 2.0 + 1im, 3.0 - 2im])
        @test coefficients(project(c, desymmetrize(space(c)))) ==
            ComplexF64[3.0 - 2.0im, 2.0 + 1.0im, 1.0 + 0.0im, 2.0 + 1.0im, 3.0 - 2.0im]

        # Interval{Float64} coefficients: the known real expansion must lie inside the enclosure
        si = interval(evensym(Fourier(2, 1.0)))
        ai = Sequence(si, interval.([1.0, 2.0, 3.0]))
        full_i = project(ai, desymmetrize(si))
        full_r = project(a, desymmetrize(space(a)))
        @test all(in_interval.(coefficients(full_r), coefficients(full_i)))

        # 0 is not a valid orbit representative in oddsym, so indexing it is out of bounds
        @test_throws BoundsError b[0]

        # guard rail: evensym/oddsym are only defined for Fourier, not for a bare TensorSpace
        @test_throws MethodError evensym(Fourier(1, 1.0) ⊗ Fourier(1, 1.0))
    end

    @testset "evensym / oddsym — Chebyshev" begin
        # same coefficient-only parity rule as Taylor: Tₖ(-x) = (-1)^k Tₖ(x)
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

        # guard rails: d4sym is only defined for TensorSpace{<:Tuple{Fourier,Fourier}}
        @test_throws MethodError d4sym(Fourier(1, 1.0))
        @test_throws MethodError d4sym(Taylor(1) ⊗ Taylor(1))
        @test_throws MethodError d4sym(Taylor(1) ⊗ Fourier(1, 1.0))
    end

    @testset "SymmetricSpace construction / equality / subset / intersect / union" begin
        s = Fourier(4, 1.0)

        # default symmetry is the identity group ⟹ the full space is recovered
        @test SymmetricSpace(s) == SymmetricSpace(s, symmetry(s))
        @test indices(SymmetricSpace(s)) == indices(s)

        # idempotent
        ss = evensym(s)
        @test SymmetricSpace(ss) === ss

        seven = evensym(s)
        sodd = oddsym(s)
        @test seven == evensym(s)
        @test issubset(seven, SymmetricSpace(s)) # every symmetric subspace embeds in the unrestricted one

        # union of spaces intersects the symmetry groups: even ∪ odd removes every non-trivial constraint
        u = union(seven, sodd)
        @test indices(u) == -4:4
        @test dimension(u) == 9

        # intersect of spaces unions the symmetry groups: even ∩ odd forces a₀ = -a₀ = 0 *and* aₖ = a₋ₖ = -a₋ₖ
        # for every k, i.e. every coefficient must vanish, so no valid orbit representative remains
        i = intersect(seven, sodd)
        @test dimension(i) == 0

        # merging an extra symmetry onto an already-symmetric space unions the two symmetry groups:
        # `SymmetricSpace(space::SymmetricSpace, sym::Group) = SymmetricSpace(desymmetrize(space), symmetry(space) ∪ sym)`.
        # Here that union is exactly `union(symmetry(seven), symmetry(sodd))`, the same group `intersect(seven, sodd)`
        # builds above: closing {id, (k↦-k, amp=1)} ∪ {id, (k↦-k, amp=-1)} under composition adds the identity-index,
        # amplitude-(-1) element (k↦-k,amp=1) ∘ (k↦-k,amp=-1) = (k↦k,amp=-1), which forces aₖ = -aₖ = 0 for every k
        merged = SymmetricSpace(seven, symmetry(sodd))
        @test merged isa SymmetricSpace
        @test length(elements(symmetry(merged))) == 4
        @test dimension(merged) == 0
        @test merged == i

        # guard rail: SymmetricSpace can only wrap a BaseSpace or a TensorSpace, not a ScalarSpace
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

        # ⊗ tensorizes the symmetry groups: block-diagonal index actions, amplitudes
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
        # restricting past two leading factors
        lifted3 = symmetry((Chebyshev(1) ⊗ Chebyshev(1)) ⊗ sE)
        @test RadiiPolynomial._restrict(lifted3, Val(2)) == symmetry(sE)

        # a group mixing the leading and trailing indices cannot be restricted
        d4 = symmetry(d4sym(Fourier(1, 1.0) ⊗ Fourier(1, 1.0))) # swaps the two factors
        @test_throws ArgumentError RadiiPolynomial._restrict(d4, Val(1))
    end

    @testset "_reps_indices" begin
        @test RadiiPolynomial._reps_indices(Int[]) === 1:1:0 # empty progression
        @test RadiiPolynomial._reps_indices([5]) === 5:1:5
        @test RadiiPolynomial._reps_indices([1, 2, 3]) === 1:1:3
        @test RadiiPolynomial._reps_indices([2, 4, 6]) === 2:2:6
        # valid 1D representatives always form an arithmetic progression (an intersection of
        # congruence classes, minus possibly the endpoint 0 fixed by the reflections); anything
        # else means the symmetry group data is inconsistent and must fail loudly
        @test_throws ArgumentError RadiiPolynomial._reps_indices([0, 1, 3])
        # tuple representatives (tensor spaces) stay as a plain vector
        @test RadiiPolynomial._reps_indices([(0, 0), (1, 0)]) == [(0, 0), (1, 0)]
    end

    @testset "canonical construction types" begin
        # the runtime type parameters are canonical — determined by `(S, G)`, never by the
        # data — so repeated constructions always yield the same concrete type and dynamic
        # dispatch downstream stays warm; in 1D the representatives are always a `StepRange`
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

        # interval symmetry groups carry enclosure-valued actions
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
        @test isequal_interval(g.coef_action.amplitude, interval(1.0))

        siT = interval(Float64, s)
        @test siT isa SymmetricSpace
        @test desymmetrize(siT) == Fourier(2, interval(1.0))
        @test siT == si
    end

end
