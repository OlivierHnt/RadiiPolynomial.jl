@testset "Vector spaces" begin

    @testset "VectorSpace (generic fallbacks)" begin
        𝒫 = ScalarSpace()
        𝒯 = Taylor(0)
        𝒫² = CartesianPower(𝒫, 2)
        @test !(𝒫 == 𝒯) && !(𝒯 == 𝒫)
        @test !(𝒫 ⊆ 𝒯) && !(𝒯 ⊆ 𝒫)
        for s ∈ (𝒫², UndefSpace())
            @test_throws MethodError intersect(s, 𝒯)
            @test_throws MethodError union(s, 𝒯)
        end
        @test dimension(𝒫) == length(indices(𝒫)) == 1
        @test RadiiPolynomial._firstindex(𝒯) == first(indices(𝒯)) == 0
        @test RadiiPolynomial._lastindex(𝒯) == last(indices(𝒯)) == 0
        @test RadiiPolynomial._checkbounds_indices(0, Taylor(2))
        @test !RadiiPolynomial._checkbounds_indices(3, Taylor(2))
        @test RadiiPolynomial._checkbounds_indices(0:2, Taylor(2))
        @test !RadiiPolynomial._checkbounds_indices(0:3, Taylor(2))
        @test RadiiPolynomial._checkbounds_indices(:, Taylor(2))
        @test RadiiPolynomial._checkbounds_indices(Taylor(1), Taylor(2)) == issubset(Taylor(1), Taylor(2)) == true
    end

    @testset "UndefSpace" begin
        s_undef = UndefSpace()
        @test s_undef == UndefSpace()
        @test s_undef ⊆ s_undef
        @test s_undef ∩ s_undef == s_undef ∪ s_undef == s_undef
        @test dimension(s_undef) == 0
        @test indices(s_undef) == Base.OneTo(0)
        @test RadiiPolynomial._findposition(1, s_undef) == 1
        @test RadiiPolynomial._findposition(s_undef, s_undef) == Base.OneTo(0)
    end

    @testset "ScalarSpace" begin
        𝒫 = ScalarSpace()
        @test 𝒫 ⊆ 𝒫
        @test 𝒫 ∩ 𝒫 == 𝒫 ∪ 𝒫 == 𝒫
        @test dimension(𝒫) == 1
        @test RadiiPolynomial._firstindex(𝒫) == 1
        @test RadiiPolynomial._lastindex(𝒫) == 1
        @test indices(𝒫) == Base.OneTo(1)
        @test RadiiPolynomial._findposition(1, 𝒫) == 1
        @test RadiiPolynomial._findposition(:, 𝒫) == Colon()
        @test RadiiPolynomial._iscompatible(𝒫, 𝒫)
        @test interval(Float64, 𝒫) == interval(𝒫) == 𝒫
        @test RadiiPolynomial._prettystring(𝒫, true) == "𝕂"
        @test RadiiPolynomial._prettystring(𝒫, false) == "ScalarSpace()"
    end

    @testset "SequenceSpace ↔ ScalarSpace (mix)" begin
        𝒫 = ScalarSpace()
        𝑇 = Taylor(2) ⊗ Fourier(1, 1.0)
        # the intersection keeps the constant mode only, in every factor of a `TensorSpace`
        for (s, s₀) ∈ ((Taylor(3), Taylor(0)), (Fourier(3, 1.0), Fourier(0, 1.0)), (Chebyshev(3), Chebyshev(0)),
                       (𝑇, Taylor(0) ⊗ Fourier(0, 1.0)))
            @test intersect(s, 𝒫) == intersect(𝒫, s) == s₀
            @test union(s, 𝒫) == union(𝒫, s) == s
        end
    end

    @testset "BaseSpace: $(nameof(typeof(s₂)))" for (s₂, s₃, idx) ∈ ((Taylor(2), Taylor(3), 0:2),
            (Fourier(2, 1.0), Fourier(3, 1.0), -2:2), (Chebyshev(2), Chebyshev(3), 0:2))
        @test !(s₂ == s₃)
        @test s₂ ⊆ s₃
        @test !(s₃ ⊆ s₂)
        @test s₂ ∩ s₃ == s₂
        @test s₂ ∪ s₃ == s₃
        @test order(s₂) == 2
        @test dimension(s₂) == length(idx)
        @test RadiiPolynomial._firstindex(s₂) == first(idx)
        @test RadiiPolynomial._lastindex(s₂) == last(idx)
        @test indices(s₂) == idx
        @test RadiiPolynomial._findindex_constant(s₂) == 0
        @test RadiiPolynomial._findposition(idx[2], s₂) == 2
        @test RadiiPolynomial._findposition(idx[2:3], s₂) == 2:3
        @test RadiiPolynomial._findposition([first(idx), last(idx)], s₂) == [1, length(idx)]
        @test RadiiPolynomial._findposition(:, s₂) == Colon()
        @test RadiiPolynomial._iscompatible(s₂, s₃)
    end

    @testset "Taylor and Chebyshev" begin
        @test_throws DomainError Taylor(-1)
        @test_throws DomainError Chebyshev(-1)
        @test RadiiPolynomial._compatible_space_with_constant_index(Taylor(2)) == Taylor(2)
        @test RadiiPolynomial._findposition(Taylor(1), Taylor(2)) == 1:2
        for s ∈ (Taylor(2), Chebyshev(2))
            @test interval(Float64, s) == interval(s) == s
        end
    end

    @testset "Fourier" begin
        ℱ₂ = Fourier(2, 1.0)
        @test_throws DomainError Fourier(-1, 1.0)
        @test_throws DomainError Fourier(1, -1.0)
        @test_throws DomainError Fourier(1, interval(-1.0, 1.0))
        @test in_interval(1.5, frequency(Fourier(2, interval(1.0, 2.0)))) # a non-thin nonnegative frequency is accepted
        @test frequency(ℱ₂) == 1.0
        @test_throws ArgumentError intersect(ℱ₂, Fourier(2, 2.0))
        @test_throws ArgumentError union(ℱ₂, Fourier(2, 2.0))
        @test !RadiiPolynomial._iscompatible(ℱ₂, Fourier(2, 2.0))
        @test convert(Fourier{Float64}, ℱ₂) == ℱ₂
        @test convert(Fourier{Int}, ℱ₂) == Fourier(2, 1)
        @test promote_type(Fourier{Float64}, Fourier{Float64}) == Fourier{Float64}
        @test promote_type(Fourier{Float64}, Fourier{Int}) == Fourier{Float64}
        ℱ₂I = interval(ℱ₂)
        @test order(ℱ₂I) == 2
        @test frequency(ℱ₂I) isa Interval
        @test interval(Float64, ℱ₂) == ℱ₂I == Fourier(2, interval(1.0))
    end

    @testset "TensorSpace" begin
        𝒯₂, ℱ₂, 𝒞₂ = Taylor(2), Fourier(2, 1.0), Chebyshev(2)
        @test_throws ArgumentError TensorSpace()
        𝑇 = TensorSpace((𝒯₂, ℱ₂, 𝒞₂))
        @test spaces(𝑇) == (𝒯₂, ℱ₂, 𝒞₂)
        @test nspaces(𝑇) == 3
        @test 𝒯₂ ⊗ ℱ₂ ⊗ 𝒞₂ == (𝒯₂ ⊗ ℱ₂) ⊗ 𝒞₂ == 𝒯₂ ⊗ (ℱ₂ ⊗ 𝒞₂) == TensorSpace(𝒯₂, ℱ₂, 𝒞₂) == 𝑇
        @test 𝑇 ⊗ 𝑇 == TensorSpace((𝒯₂, ℱ₂, 𝒞₂, 𝒯₂, ℱ₂, 𝒞₂))
        @test 𝑇[1] == 𝒯₂
        @test 𝑇[1:3] == 𝑇[[1, 2, 3]] == 𝑇[:] == 𝑇
        @test Base.tail(𝑇) == TensorSpace((ℱ₂, 𝒞₂))
        @test Base.front(𝑇) == TensorSpace((𝒯₂, ℱ₂))
        @test 𝑇 ⊆ Taylor(3) ⊗ ℱ₂ ⊗ 𝒞₂
        @test !(Taylor(3) ⊗ ℱ₂ ⊗ 𝒞₂ ⊆ 𝑇)
        @test 𝑇 ∩ (Taylor(3) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(3)) == 𝒯₂ ⊗ Fourier(1, 1.0) ⊗ 𝒞₂
        @test 𝑇 ∪ (Taylor(3) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(3)) == Taylor(3) ⊗ ℱ₂ ⊗ Chebyshev(3)
        @test !(𝑇 == TensorSpace((𝒯₂, ℱ₂)))
        @test !(TensorSpace((𝒯₂, ℱ₂)) ⊆ 𝑇)
        @test dimension(𝑇) == 3 * 5 * 3
        @test dimension(𝑇, 1) == 3
        @test dimensions(𝑇) == (3, 5, 3)
        @test RadiiPolynomial._firstindex(𝑇) == (0, -2, 0)
        @test RadiiPolynomial._lastindex(𝑇) == (2, 2, 2)
        @test indices(𝑇) == TensorIndices((0:2, -2:2, 0:2))
        @test RadiiPolynomial._findindex_constant(𝑇) == (0, 0, 0)
        @test RadiiPolynomial._findposition((1, 2, 0), 𝑇) == 14
        @test RadiiPolynomial._findposition((:, -2:2, :), 𝑇) == collect(1:45)
        @test RadiiPolynomial._findposition([(1, 2, 0), (2, 2, 2)], 𝑇) == [14, 45]
        @test RadiiPolynomial._findposition((:, :, :), 𝑇) == Colon()
        @test RadiiPolynomial._findposition(:, 𝑇) == Colon()
        @test order(𝑇) == (2, 2, 2)
        @test order(𝑇, 1) == 2
        @test_throws MethodError frequency(𝑇) # only the Fourier factor carries a frequency
        @test frequency(𝑇, 2) == 1.0
        @test RadiiPolynomial._iscompatible(𝑇, Taylor(5) ⊗ Fourier(2, 1.0) ⊗ Chebyshev(5))
        @test !RadiiPolynomial._iscompatible(𝑇, Taylor(5) ⊗ Fourier(2, 2.0) ⊗ Chebyshev(5))
        @test convert(TensorSpace{Tuple{Taylor,Fourier{Float64},Chebyshev}}, 𝑇) == 𝑇
        @test convert(TensorSpace{Tuple{Taylor,Fourier{Int},Chebyshev}}, 𝑇) == Taylor(2) ⊗ Fourier(2, 1) ⊗ Chebyshev(2)
        @test promote_type(TensorSpace{Tuple{Taylor,Fourier{Float64}}}, TensorSpace{Tuple{Taylor,Fourier{Int}}}) ==
            TensorSpace{Tuple{Taylor,Fourier}} # promotion of the underlying `Tuple` type is not elementwise
        @test interval(Float64, 𝑇) == interval(𝑇) == Taylor(2) ⊗ interval(ℱ₂) ⊗ Chebyshev(2)
        @test RadiiPolynomial._checkbounds_indices(indices(𝑇), 𝑇)
        @test !RadiiPolynomial._checkbounds_indices(TensorIndices((0:3, -2:2, 0:2)), 𝑇)
        @test RadiiPolynomial._prettystring(𝑇, true) == "Taylor(2) ⊗ Fourier(2, 1.0) ⊗ Chebyshev(2)"
    end

    @testset "TensorIndices" begin
        ti = TensorIndices((0:2, -1:1))
        @test eltype(ti) == Tuple{Int,Int}
        @test length(ti) == 9
        # the first factor varies fastest
        cti = [(0, -1), (1, -1), (2, -1), (0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
        @test collect(ti) == cti == [ti[k] for k ∈ 1:9]
        @test ti[CartesianIndex(2, 3)] == (1, 1)
        @test indices(Taylor(2) ⊗ Fourier(1, 1.0)) == ti
        @test issubset(TensorIndices((0:1, 0:1)), TensorIndices((0:2, -1:2)))
        @test intersect(TensorIndices((0:2, -1:2)), TensorIndices((1:3, -2:1))) == TensorIndices((1:2, -1:1))
        # `union` yields vectors, which `==` on `TensorIndices` compares by identity
        @test collect(union(TensorIndices((0:1, 0:1)), TensorIndices((1:2, 1:2)))) == collect(TensorIndices(([0, 1, 2], [0, 1, 2])))
    end

    @testset "CartesianPower" begin
        @test_throws DomainError CartesianPower(Taylor(1), -1)
        𝒯₁¹ = CartesianPower(Taylor(1), 1)
        𝒯₁³ = Taylor(1)^3
        @test 𝒯₁³ == CartesianPower(Taylor(1), 3)
        @test space(𝒯₁³) == Taylor(1)
        @test spaces(𝒯₁³) == [Taylor(1), Taylor(1), Taylor(1)]
        @test nspaces(𝒯₁³) == 3
        @test 𝒯₁¹^1 == CartesianPower(𝒯₁¹, 1)
        @test 𝒯₁³[1] == Taylor(1)
        @test 𝒯₁³[1:2] == 𝒯₁³[[1, 2]] == CartesianPower(Taylor(1), 2)
        @test 𝒯₁³[:] == 𝒯₁³
        @test_throws BoundsError 𝒯₁³[4]
        @test_throws BoundsError 𝒯₁³[1:4]
        @test_throws BoundsError 𝒯₁³[[1, 4]]
        @test 𝒯₁³ ⊆ Taylor(2)^3
        @test !(𝒯₁³ ⊆ Taylor(2)^2)
        @test 𝒯₁³ ∩ Taylor(2)^3 == 𝒯₁³
        @test 𝒯₁³ ∪ Taylor(2)^3 == Taylor(2)^3
        @test_throws ArgumentError intersect(𝒯₁³, Taylor(1)^2)
        @test_throws ArgumentError union(𝒯₁³, Taylor(1)^2)
        @test dimension(𝒯₁³) == 6
        @test dimension(𝒯₁³, 2) == 2
        @test_throws BoundsError dimension(𝒯₁³, 0)
        @test_throws BoundsError dimension(𝒯₁³, 4)
        @test dimensions(𝒯₁³) == [2, 2, 2]
        @test indices(𝒯₁³) == Base.OneTo(6)
        @test order(𝒯₁³) == [1, 1, 1]
        @test order(𝒯₁³, 2) == 1
        @test_throws BoundsError order(𝒯₁³, 4)
        ℱ₃³ = Fourier(3, 1.0)^3
        @test frequency(ℱ₃³) == [1.0, 1.0, 1.0]
        @test frequency(ℱ₃³, 1) == 1.0
        @test_throws BoundsError frequency(ℱ₃³, 4)
        @test RadiiPolynomial._component_findposition(1, 𝒯₁³) == RadiiPolynomial._component_findposition(1:1, 𝒯₁³) == 1:2
        @test RadiiPolynomial._component_findposition(1:2, 𝒯₁³) == 1:4
        @test RadiiPolynomial._component_findposition(:, 𝒯₁³) == Colon()
        @test RadiiPolynomial._component_findposition([1, 3], 𝒯₁³) == [1, 2, 5, 6]
        @test RadiiPolynomial._iscompatible(𝒯₁³, Taylor(9)^3)
        @test !RadiiPolynomial._iscompatible(𝒯₁³, Taylor(9)^2)
        @test convert(CartesianPower{Taylor}, 𝒯₁³) == 𝒯₁³
        @test promote_type(CartesianPower{Taylor}, CartesianPower{Chebyshev}) == CartesianPower{BaseSpace}
        ℱ₂² = Fourier(2, 1.0)^2
        @test interval(Float64, ℱ₂²) == interval(ℱ₂²) == Fourier(2, interval(1.0))^2
        @test RadiiPolynomial._prettystring((Taylor(1) ⊗ Fourier(1, 1.0))^2, true) == "(Taylor(1) ⊗ Fourier(1, 1.0))²"
    end

    @testset "CartesianProduct" begin
        @test_throws ArgumentError CartesianProduct()
        𝒯₁¹ = CartesianPower(Taylor(1), 1)
        𝒞₂² = CartesianPower(Chebyshev(2), 2)
        ℱ₃³ = CartesianPower(Fourier(3, 1.0), 3)
        𝑃 = CartesianProduct((𝒯₁¹, 𝒞₂², ℱ₃³))
        𝑃′ = CartesianProduct((CartesianPower(Taylor(2), 1), CartesianPower(Chebyshev(3), 2), ℱ₃³))
        @test spaces(𝒯₁¹) == [Taylor(1)]
        @test spaces(𝑃) == (𝒯₁¹, 𝒞₂², ℱ₃³)
        @test nspaces(ℱ₃³) == nspaces(𝑃) == 3
        @test 𝒯₁¹ × 𝒞₂² × ℱ₃³ == 𝒯₁¹ × (𝒞₂² × ℱ₃³) == (𝒯₁¹ × 𝒞₂²) × ℱ₃³ == (𝒯₁¹ × 𝒞₂²) × CartesianProduct(tuple(ℱ₃³)) == 𝑃
        @test Base.front(𝑃) == 𝒯₁¹ × 𝒞₂²
        @test Base.tail(𝑃) == 𝒞₂² × ℱ₃³
        @test 𝒞₂²[1] == Chebyshev(2)
        @test 𝒞₂²[1:2] == 𝒞₂²[[1, 2]] == 𝒞₂²[:] == 𝒞₂²
        @test 𝑃[1] == 𝒯₁¹
        @test 𝑃[1:3] == 𝑃[[1, 2, 3]] == 𝑃[:] == 𝑃
        @test 𝑃 ⊆ 𝑃′
        @test !(𝑃′ ⊆ 𝑃)
        @test 𝑃 ∩ 𝑃′ == 𝑃
        @test 𝑃 ∪ 𝑃′ == 𝑃′
        @test !(𝑃 == (𝒯₁¹ × 𝒞₂²))
        @test dimension(𝑃) == 2 + 6 + 21
        @test dimension(𝑃, 1) == dimension(𝒯₁¹, 1) == 2
        @test dimensions(𝒯₁¹) == [2]
        @test dimensions(𝑃) == (2, 6, 21)
        @test RadiiPolynomial._firstindex(𝑃) == RadiiPolynomial._firstindex(𝒞₂²) == 1
        @test RadiiPolynomial._lastindex(𝒞₂²) == 6
        @test RadiiPolynomial._lastindex(𝑃) == 29
        @test indices(𝒞₂²) == Base.OneTo(6)
        @test indices(𝑃) == Base.OneTo(29)
        @test order(𝑃) == ([1], [2, 2], [3, 3, 3])
        @test order(𝑃, 1) == [1]
        @test order(𝒯₁¹, 1) == 1
        @test_throws MethodError frequency(𝑃) # only the Fourier block carries a frequency
        @test frequency(𝑃, 3) == frequency(ℱ₃³) == [1.0, 1.0, 1.0]
        @test RadiiPolynomial._component_findposition(1, 𝒯₁¹) == RadiiPolynomial._component_findposition(1:1, 𝒯₁¹) == 1:2
        @test RadiiPolynomial._component_findposition(2, 𝒞₂²) == RadiiPolynomial._component_findposition(2:2, 𝒞₂²) == 4:6
        @test RadiiPolynomial._component_findposition(1, 𝑃) == RadiiPolynomial._component_findposition(1:1, 𝑃) == 1:2
        @test RadiiPolynomial._component_findposition(2, 𝑃) == RadiiPolynomial._component_findposition(2:2, 𝑃) == 3:8
        @test RadiiPolynomial._component_findposition(1:2, 𝑃) == 1:8
        @test RadiiPolynomial._component_findposition([1, 3], 𝑃) == vcat(1:2, 9:29)
        @test RadiiPolynomial._iscompatible(𝑃, 𝑃′)
        @test convert(CartesianProduct{Tuple{CartesianPower{Taylor}, CartesianPower{Chebyshev}, CartesianPower{Fourier{Float64}}}}, 𝑃) == 𝑃
        @test convert(CartesianProduct{Tuple{CartesianPower{Taylor}, CartesianPower{Chebyshev}, CartesianPower{Fourier{Int}}}}, 𝑃) ==
            𝒯₁¹ × 𝒞₂² × Fourier(3, 1)^3
        @test interval(Float64, 𝑃) == interval(𝑃) == 𝒯₁¹ × 𝒞₂² × CartesianPower(interval(Fourier(3, 1.0)), 3)
        @test RadiiPolynomial._prettystring(Taylor(1) × Fourier(1, 1.0) × Chebyshev(1), true) == "Taylor(1) × Fourier(1, 1.0) × Chebyshev(1)"
    end

    @testset "_supscript" begin
        glyphs = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹']
        for i ∈ 0:9
            @test RadiiPolynomial._supscript_digit(i) == RadiiPolynomial._supscript(i) == glyphs[i+1]
        end
        @test RadiiPolynomial._supscript(12) == "¹²"
        @test RadiiPolynomial._supscript(100) == "¹⁰⁰"
    end

    @testset "CartesianPower ↔ CartesianProduct (mix)" begin
        pow = Taylor(1)^3
        prod_same = CartesianProduct((Taylor(1), Taylor(1), Taylor(1)))
        prod_larger = CartesianProduct((Taylor(2), Taylor(2), Taylor(2)))
        prod_mismatch_n = CartesianProduct((Taylor(1), Taylor(1)))
        @test pow == prod_same && prod_same == pow
        @test pow ⊆ prod_larger
        @test prod_same ⊆ pow
        @test !(pow == prod_mismatch_n)
        @test !(pow ⊆ prod_mismatch_n)
        @test intersect(pow, prod_larger) == intersect(prod_larger, pow) == prod_same
        @test union(pow, prod_larger) == union(prod_larger, pow) == prod_larger
        for op ∈ (intersect, union)
            @test_throws ArgumentError op(pow, prod_mismatch_n)
            @test_throws ArgumentError op(prod_mismatch_n, pow)
        end
        @test RadiiPolynomial._iscompatible(pow, prod_larger)
        @test RadiiPolynomial._iscompatible(prod_larger, pow)
        @test !RadiiPolynomial._iscompatible(pow, prod_mismatch_n)
    end

    @testset "indexing a CartesianSpace by a smaller one" begin
        # the leading coefficients of every block are selected. The third and fourth cases nest a
        # power inside a product and a product inside a power, where each slot must resolve to its
        # own leaf space
        for (s, α, pos) ∈ ((Taylor(2)^3, Taylor(1)^3, [1, 2, 4, 5, 7, 8]),
                           (Taylor(1) × Fourier(1, 1.0) × Chebyshev(1), Taylor(0) × Fourier(0, 1.0) × Chebyshev(0), [1, 4, 6]),
                           (Taylor(1)^2 × Chebyshev(1), Taylor(0)^2 × Chebyshev(0), [1, 3, 5]),
                           ((Fourier(1, 1.0) × Taylor(1))^2, (Fourier(0, 1.0) × Taylor(0))^2, [2, 4, 7, 9]))
            @test RadiiPolynomial._findposition(α, s) == pos
            @test coefficients(Sequence(s, collect(1.0:dimension(s)))[α]) == pos
        end
        a = Sequence((Fourier(1, 1.0) × Taylor(1))^2, collect(1.0:10.0))
        β = Fourier(0, 1.0) × Taylor(0)
        @test coefficients(component(a, 1)[β]) == [2.0, 4.0]
        @test coefficients(component(a, 2)[β]) == [7.0, 9.0]
    end

end
