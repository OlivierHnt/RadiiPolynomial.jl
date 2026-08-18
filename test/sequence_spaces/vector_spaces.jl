@testset "Vector spaces" begin

    @testset "VectorSpace (generic fallbacks)" begin
        𝒫 = ScalarSpace()
        𝒯 = Taylor(0)
        𝒫² = CartesianPower(𝒫, 2)
        @test !(𝒫 == 𝒯)
        @test !(𝒯 == 𝒫)
        @test !(𝒫 ⊆ 𝒯)
        @test !(𝒯 ⊆ 𝒫)
        @test_throws MethodError intersect(𝒫², 𝒯)
        @test_throws MethodError union(𝒫², 𝒯)
        @test_throws MethodError intersect(UndefSpace(), 𝒯)
        @test_throws MethodError union(UndefSpace(), 𝒯)
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
        @test intersect(Taylor(3), 𝒫) == intersect(𝒫, Taylor(3)) == Taylor(0)
        @test union(Taylor(3), 𝒫) == union(𝒫, Taylor(3)) == Taylor(3)
        @test intersect(Fourier(3, 1.0), 𝒫) == intersect(𝒫, Fourier(3, 1.0)) == Fourier(0, 1.0)
        @test union(Fourier(3, 1.0), 𝒫) == union(𝒫, Fourier(3, 1.0)) == Fourier(3, 1.0)
        @test intersect(Chebyshev(3), 𝒫) == intersect(𝒫, Chebyshev(3)) == Chebyshev(0)
        @test union(Chebyshev(3), 𝒫) == union(𝒫, Chebyshev(3)) == Chebyshev(3)
        𝑇 = Taylor(2) ⊗ Fourier(1, 1.0)
        @test union(𝑇, 𝒫) == union(𝒫, 𝑇) == 𝑇
        # intersecting a `TensorSpace` collapses every factor to order zero
        @test intersect(𝑇, 𝒫) == Taylor(0) ⊗ Fourier(0, 1.0)
        @test intersect(𝒫, 𝑇) == Taylor(0) ⊗ Fourier(0, 1.0)
    end

    @testset "Taylor" begin
        𝒯₂, 𝒯₃ = Taylor(2), Taylor(3)
        @test_throws DomainError Taylor(-1)
        @test !(𝒯₂ == 𝒯₃)
        @test 𝒯₂ ⊆ 𝒯₃
        @test !(𝒯₃ ⊆ 𝒯₂)
        @test 𝒯₂ ∩ 𝒯₃ == 𝒯₂
        @test 𝒯₂ ∪ 𝒯₃ == 𝒯₃
        @test order(𝒯₂) == 2
        @test dimension(𝒯₂) == 3
        @test RadiiPolynomial._firstindex(𝒯₂) == 0
        @test RadiiPolynomial._lastindex(𝒯₂) == 2
        @test indices(𝒯₂) == 0:2
        @test collect(indices(𝒯₂)) == [0, 1, 2]
        @test RadiiPolynomial._compatible_space_with_constant_index(𝒯₂) == 𝒯₂
        @test RadiiPolynomial._findindex_constant(𝒯₂) == 0
        @test RadiiPolynomial._findposition(1, 𝒯₂) == 2
        @test RadiiPolynomial._findposition(0:2, 𝒯₂) == 1:3
        @test RadiiPolynomial._findposition([0, 2], 𝒯₂) == [1, 3]
        @test RadiiPolynomial._findposition(:, 𝒯₂) == Colon()
        @test RadiiPolynomial._findposition(Taylor(1), 𝒯₂) == 1:2
        @test RadiiPolynomial._iscompatible(𝒯₂, 𝒯₃)
        @test interval(Float64, 𝒯₂) == interval(𝒯₂) == 𝒯₂
    end

    @testset "Fourier" begin
        ℱ₂, ℱ₃ = Fourier(2, 1.0), Fourier(3, 1.0)
        @test_throws DomainError Fourier(-1, 1.0)
        @test_throws DomainError Fourier(1, -1.0)
        @test_throws DomainError Fourier(1, interval(-1.0, 1.0)) # an interval frequency must not contain negative values
        @test !(ℱ₂ == ℱ₃)
        @test ℱ₂ ⊆ ℱ₃
        @test !(ℱ₃ ⊆ ℱ₂)
        @test ℱ₂ ∩ ℱ₃ == ℱ₂
        @test ℱ₂ ∪ ℱ₃ == ℱ₃
        @test_throws ArgumentError intersect(ℱ₂, Fourier(2, 2.0))
        @test_throws ArgumentError union(ℱ₂, Fourier(2, 2.0))
        @test order(ℱ₂) == 2
        @test frequency(ℱ₂) == 1.0
        @test dimension(ℱ₂) == 5
        @test RadiiPolynomial._firstindex(ℱ₂) == -2
        @test RadiiPolynomial._lastindex(ℱ₂) == 2
        @test indices(ℱ₂) == -2:2
        @test collect(indices(ℱ₂)) == [-2, -1, 0, 1, 2]
        @test RadiiPolynomial._findindex_constant(ℱ₂) == 0
        @test RadiiPolynomial._findposition(-1, ℱ₂) == 2
        @test RadiiPolynomial._findposition(0:1, ℱ₂) == 3:4
        @test RadiiPolynomial._findposition([-2, 2], ℱ₂) == [1, 5]
        @test RadiiPolynomial._findposition(:, ℱ₂) == Colon()
        @test RadiiPolynomial._iscompatible(ℱ₂, ℱ₃)
        @test !RadiiPolynomial._iscompatible(ℱ₂, Fourier(2, 2.0))
        @test convert(Fourier{Float64}, ℱ₂) == ℱ₂
        @test convert(Fourier{Int}, ℱ₂) == Fourier(2, 1)
        @test promote_type(Fourier{Float64}, Fourier{Float64}) == Fourier{Float64}
        @test promote_type(Fourier{Float64}, Fourier{Int}) == Fourier{Float64}
        ℱ₂I = interval(ℱ₂)
        @test order(ℱ₂I) == 2
        @test frequency(ℱ₂I) isa Interval
        @test ℱ₂I == Fourier(2, interval(1.0))
        @test interval(Float64, ℱ₂) == ℱ₂I
        # a non-thin frequency is accepted as long as it stays nonnegative
        ℱI = Fourier(2, interval(1.0, 2.0))
        @test in_interval(1.5, frequency(ℱI))
    end

    @testset "Chebyshev" begin
        𝒞₂, 𝒞₃ = Chebyshev(2), Chebyshev(3)
        @test_throws DomainError Chebyshev(-1)
        @test !(𝒞₂ == 𝒞₃)
        @test 𝒞₂ ⊆ 𝒞₃
        @test 𝒞₂ ∩ 𝒞₃ == 𝒞₂
        @test 𝒞₂ ∪ 𝒞₃ == 𝒞₃
        @test order(𝒞₂) == 2
        @test dimension(𝒞₂) == 3
        @test RadiiPolynomial._firstindex(𝒞₂) == 0
        @test RadiiPolynomial._lastindex(𝒞₂) == 2
        @test indices(𝒞₂) == 0:2
        @test collect(indices(𝒞₂)) == [0, 1, 2]
        @test RadiiPolynomial._findindex_constant(𝒞₂) == 0
        @test RadiiPolynomial._findposition(1, 𝒞₂) == 2
        @test RadiiPolynomial._findposition(0:2, 𝒞₂) == 1:3
        @test RadiiPolynomial._findposition([0, 2], 𝒞₂) == [1, 3]
        @test RadiiPolynomial._findposition(:, 𝒞₂) == Colon()
        @test RadiiPolynomial._iscompatible(𝒞₂, 𝒞₃)
        @test interval(Float64, 𝒞₂) == interval(𝒞₂) == 𝒞₂
    end

    @testset "TensorSpace" begin
        𝒯₂, ℱ₂, 𝒞₂ = Taylor(2), Fourier(2, 1.0), Chebyshev(2)
        @test_throws ArgumentError TensorSpace()
        𝑇 = TensorSpace((𝒯₂, ℱ₂, 𝒞₂))
        @test spaces(𝑇) == (𝒯₂, ℱ₂, 𝒞₂)
        @test nspaces(𝑇) == 3
        @test 𝒯₂ ⊗ ℱ₂ ⊗ 𝒞₂ == (𝒯₂ ⊗ ℱ₂) ⊗ 𝒞₂ == 𝒯₂ ⊗ (ℱ₂ ⊗ 𝒞₂) == TensorSpace(𝒯₂, ℱ₂, 𝒞₂) == 𝑇
        @test 𝑇 ⊗ 𝑇 == TensorSpace((𝒯₂, ℱ₂, 𝒞₂, 𝒯₂, ℱ₂, 𝒞₂))
        @test (𝒯₂ ⊗ ℱ₂) ⊗ 𝒞₂ == TensorSpace((𝒯₂, ℱ₂)) ⊗ 𝒞₂
        @test 𝒯₂ ⊗ (ℱ₂ ⊗ 𝒞₂) == 𝒯₂ ⊗ TensorSpace((ℱ₂, 𝒞₂))
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
        @test dimension(𝑇) == 3 * 5 * 3 == 45
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
        @test length(ti) == 3 * 3 == 9
        # first factor varies fastest (standard product-iterator convention)
        @test collect(ti) == [(0, -1), (1, -1), (2, -1), (0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
        @test indices(Taylor(2) ⊗ Fourier(1, 1.0)) == TensorIndices((0:2, -1:1))
        @test issubset(TensorIndices((0:1, 0:1)), TensorIndices((0:2, -1:2)))
        @test intersect(TensorIndices((0:2, -1:2)), TensorIndices((1:3, -2:1))) == TensorIndices((1:2, -1:1))
        # `union` yields vectors rather than ranges, and the default structural
        # equality compares those by identity, so compare the iterations instead
        @test collect(union(TensorIndices((0:1, 0:1)), TensorIndices((1:2, 1:2)))) == collect(TensorIndices(([0, 1, 2], [0, 1, 2])))
        cti = collect(ti)
        for k in 1:length(ti)
            @test ti[k] == cti[k]
        end
        @test ti[1] == (0, -1)
        @test ti[9] == (2, 1)
        @test ti[CartesianIndex(2, 3)] == cti[8] == (1, 1)
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
        @test dimension(𝒯₁³) == 2 * 3 == 6
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
        # indexing by a smaller power selects the leading coefficients of every block
        s = Taylor(2)^3
        α = Taylor(1)^3
        @test RadiiPolynomial._findposition(α, s) == [1, 2, 4, 5, 7, 8]
        a = Sequence(s, collect(1.0:9.0))
        @test coefficients(a[α]) == [1.0, 2.0, 4.0, 5.0, 7.0, 8.0]
        @test convert(CartesianPower{Taylor}, 𝒯₁³) == 𝒯₁³
        @test promote_type(CartesianPower{Taylor}, CartesianPower{Chebyshev}) == CartesianPower{BaseSpace}
        ℱ₂² = Fourier(2, 1.0)^2
        @test interval(Float64, ℱ₂²) == interval(ℱ₂²) == Fourier(2, interval(1.0))^2
        𝒯ℱ = Taylor(1) ⊗ Fourier(1, 1.0)
        @test RadiiPolynomial._prettystring(𝒯ℱ^2, true) == "(Taylor(1) ⊗ Fourier(1, 1.0))²"
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
        @test dimension(𝑃) == 2 + 6 + 21 == 29
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
        @test frequency(ℱ₃³, 1) == 1.0
        @test RadiiPolynomial._component_findposition(1, 𝒯₁¹) == RadiiPolynomial._component_findposition(1:1, 𝒯₁¹) == 1:2
        @test RadiiPolynomial._component_findposition(2, 𝒞₂²) == RadiiPolynomial._component_findposition(2:2, 𝒞₂²) == 4:6
        @test RadiiPolynomial._component_findposition(1, 𝑃) == RadiiPolynomial._component_findposition(1:1, 𝑃) == 1:2
        @test RadiiPolynomial._component_findposition(2, 𝑃) == RadiiPolynomial._component_findposition(2:2, 𝑃) == 3:8
        @test RadiiPolynomial._component_findposition(1:2, 𝑃) == 1:8
        @test RadiiPolynomial._component_findposition([1, 3], 𝑃) == vcat(1:2, 9:29)
        @test RadiiPolynomial._iscompatible(𝑃, 𝑃′)
        # indexing by a smaller product picks matching positions blockwise
        s2 = Taylor(1) × Fourier(1, 1.0) × Chebyshev(1)
        α2 = Taylor(0) × Fourier(0, 1.0) × Chebyshev(0)
        @test RadiiPolynomial._findposition(α2, s2) == [1, 4, 6]
        a2 = Sequence(s2, collect(1.0:7.0))
        @test coefficients(a2[α2]) == [1.0, 4.0, 6.0]
        @test convert(CartesianProduct{Tuple{CartesianPower{Taylor}, CartesianPower{Chebyshev}, CartesianPower{Fourier{Float64}}}}, 𝑃) == 𝑃
        @test convert(CartesianProduct{Tuple{CartesianPower{Taylor}, CartesianPower{Chebyshev}, CartesianPower{Fourier{Int}}}}, 𝑃) ==
            𝒯₁¹ × 𝒞₂² × Fourier(3, 1)^3
        @test interval(Float64, 𝑃) == interval(𝑃) == 𝒯₁¹ × 𝒞₂² × CartesianPower(interval(Fourier(3, 1.0)), 3)
        𝒯ℱ𝒞 = Taylor(1) × Fourier(1, 1.0) × Chebyshev(1)
        @test RadiiPolynomial._prettystring(𝒯ℱ𝒞, true) == "Taylor(1) × Fourier(1, 1.0) × Chebyshev(1)"
    end

    @testset "_supscript / _supscript_digit" begin
        glyphs = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹']
        for i in 0:9
            @test RadiiPolynomial._supscript_digit(i) == glyphs[i+1]
            @test RadiiPolynomial._supscript(i) == glyphs[i+1]
        end
        @test RadiiPolynomial._supscript(12) == "¹²"
        @test RadiiPolynomial._supscript(100) == "¹⁰⁰" # interior zero digit
    end

    @testset "CartesianPower ↔ CartesianProduct (mix)" begin
        pow = Taylor(1)^3
        prod_same = CartesianProduct((Taylor(1), Taylor(1), Taylor(1)))
        prod_larger = CartesianProduct((Taylor(2), Taylor(2), Taylor(2)))
        prod_mismatch_n = CartesianProduct((Taylor(1), Taylor(1)))
        @test pow == prod_same
        @test prod_same == pow
        @test pow ⊆ prod_larger
        @test prod_same ⊆ pow
        @test !(pow == prod_mismatch_n)
        @test !(pow ⊆ prod_mismatch_n)
        @test intersect(pow, prod_larger) == CartesianProduct((Taylor(1), Taylor(1), Taylor(1)))
        @test union(pow, prod_larger) == CartesianProduct((Taylor(2), Taylor(2), Taylor(2)))
        @test intersect(prod_larger, pow) == CartesianProduct((Taylor(1), Taylor(1), Taylor(1)))
        @test union(prod_larger, pow) == CartesianProduct((Taylor(2), Taylor(2), Taylor(2)))
        @test_throws ArgumentError intersect(pow, prod_mismatch_n)
        @test_throws ArgumentError union(pow, prod_mismatch_n)
        @test_throws ArgumentError intersect(prod_mismatch_n, pow)
        @test_throws ArgumentError union(prod_mismatch_n, pow)
        @test RadiiPolynomial._iscompatible(pow, prod_larger)
        @test RadiiPolynomial._iscompatible(prod_larger, pow)
        @test !RadiiPolynomial._iscompatible(pow, prod_mismatch_n)
    end

    @testset "Nested CartesianSpace _findposition" begin
        # a power of a base space is uniform: every copy inside it is the same leaf space
        s3 = Taylor(1)^2 × Chebyshev(1)
        α3 = Taylor(0)^2 × Chebyshev(0)
        @test RadiiPolynomial._findposition(α3, s3) == [1, 3, 5]
        a3 = Sequence(s3, collect(1.0:6.0))
        @test coefficients(a3[α3]) == [1.0, 3.0, 5.0]

        # a power of a product of distinct spaces repeats its factor pattern, so slots
        # beyond the first copy must still resolve to the right leaf space
        s4 = (Fourier(1, 1.0) × Taylor(1))^2
        α4 = (Fourier(0, 1.0) × Taylor(0))^2
        a4 = Sequence(s4, collect(1.0:10.0))
        # copy 1 is [1,2,3,4,5] (Fourier zero mode at 2, Taylor zero mode at 4), copy 2 is
        # [6,7,8,9,10] (Fourier zero mode at 7, Taylor zero mode at 9)
        @test RadiiPolynomial._findposition(α4, s4) == [2, 4, 7, 9]
        @test coefficients(a4[α4]) == [2.0, 4.0, 7.0, 9.0]
        # each copy's zero-mode positions match its own block indexed on its own
        β4 = Fourier(0, 1.0) × Taylor(0)
        @test coefficients(component(a4, 1)[β4]) == [2.0, 4.0]
        @test coefficients(component(a4, 2)[β4]) == [7.0, 9.0]
    end

    @testset "_iscompatible" begin
        @test RadiiPolynomial._iscompatible(ScalarSpace(), ScalarSpace())
        @test RadiiPolynomial._iscompatible(Taylor(1), Taylor(5))
        @test RadiiPolynomial._iscompatible(Chebyshev(1), Chebyshev(9))
        @test RadiiPolynomial._iscompatible(Fourier(1, 1.0), Fourier(5, 1.0))
        @test !RadiiPolynomial._iscompatible(Fourier(1, 1.0), Fourier(1, 2.0))
        𝑇 = Taylor(1) ⊗ Fourier(1, 1.0)
        @test RadiiPolynomial._iscompatible(𝑇, Taylor(5) ⊗ Fourier(5, 1.0))
        @test !RadiiPolynomial._iscompatible(𝑇, Taylor(5) ⊗ Fourier(5, 2.0))
    end

end
