@testset "Vector spaces" begin

    @testset "VectorSpace (generic fallbacks)" begin
        # unrelated concrete subtypes: `==`, `issubset` fall back to `false`,
        # `intersect`/`union` fall back to a `MethodError` (cf. lines 10-13)
        𝒫 = ScalarSpace()
        𝒯 = Taylor(0)
        𝒫² = CartesianPower(𝒫, 2)
        @test !(𝒫 == 𝒯)
        @test !(𝒯 == 𝒫)
        @test !(𝒫 ⊆ 𝒯)
        @test !(𝒯 ⊆ 𝒫)
        @test_throws MethodError intersect(𝒫², 𝒯) # CartesianSpace vs SequenceSpace: no specific method
        @test_throws MethodError union(𝒫², 𝒯)
        @test_throws MethodError intersect(EmptySpace(), 𝒯)
        @test_throws MethodError union(EmptySpace(), 𝒯)
        # generic `dimension`/`_firstindex`/`_lastindex` derived from `indices`
        @test dimension(𝒫) == length(indices(𝒫)) == 1
        @test RadiiPolynomial._firstindex(𝒯) == first(indices(𝒯)) == 0
        @test RadiiPolynomial._lastindex(𝒯) == last(indices(𝒯)) == 0
        # `_checkbounds_indices` generic behaviour
        @test RadiiPolynomial._checkbounds_indices(0, Taylor(2))
        @test !RadiiPolynomial._checkbounds_indices(3, Taylor(2))
        @test RadiiPolynomial._checkbounds_indices(0:2, Taylor(2))
        @test !RadiiPolynomial._checkbounds_indices(0:3, Taylor(2))
        @test RadiiPolynomial._checkbounds_indices(:, Taylor(2))
        @test RadiiPolynomial._checkbounds_indices(Taylor(1), Taylor(2)) == issubset(Taylor(1), Taylor(2)) == true
    end

    @testset "EmptySpace" begin
        ∅ = EmptySpace()
        @test ∅ == EmptySpace()
        @test ∅ ⊆ ∅
        @test ∅ ∩ ∅ == ∅ ∪ ∅ == ∅
        @test dimension(∅) == 0
        @test indices(∅) == Base.OneTo(0)
        @test RadiiPolynomial._findposition(1, ∅) == 1 # identity map
        @test RadiiPolynomial._findposition(∅, ∅) == Base.OneTo(0)
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
        # interval overload: a `ScalarSpace` is its own interval counterpart
        @test interval(Float64, 𝒫) == interval(𝒫) == 𝒫
        # `_prettystring` both branches of the `ifelse` (line 98)
        @test RadiiPolynomial._prettystring(𝒫, true) == "𝕂"
        @test RadiiPolynomial._prettystring(𝒫, false) == "ScalarSpace()"
    end

    @testset "SequenceSpace ↔ ScalarSpace (mix)" begin
        # `intersect`/`union` between a `SequenceSpace` and a `ScalarSpace` are
        # defined regardless of argument order (lines 109-112)
        𝒫 = ScalarSpace()
        @test intersect(Taylor(3), 𝒫) == intersect(𝒫, Taylor(3)) == Taylor(0) # order collapses to 0
        @test union(Taylor(3), 𝒫) == union(𝒫, Taylor(3)) == Taylor(3) # union is a no-op
        @test intersect(Fourier(3, 1.0), 𝒫) == intersect(𝒫, Fourier(3, 1.0)) == Fourier(0, 1.0)
        @test union(Fourier(3, 1.0), 𝒫) == union(𝒫, Fourier(3, 1.0)) == Fourier(3, 1.0)
        @test intersect(Chebyshev(3), 𝒫) == intersect(𝒫, Chebyshev(3)) == Chebyshev(0)
        @test union(Chebyshev(3), 𝒫) == union(𝒫, Chebyshev(3)) == Chebyshev(3)
        # `union` with a `TensorSpace` is a no-op (goes through the generic `s` branch)
        𝑇 = Taylor(2) ⊗ Fourier(1, 1.0)
        @test union(𝑇, 𝒫) == union(𝒫, 𝑇) == 𝑇
        # `intersect` with a `TensorSpace` calls `_zero_space(s::TensorSpace)` (defined in
        # linear_operators/projection.jl as `TensorSpace(map(_zero_space, spaces(s)))`)
        @test intersect(𝑇, 𝒫) == Taylor(0) ⊗ Fourier(0, 1.0)
        @test intersect(𝒫, 𝑇) == Taylor(0) ⊗ Fourier(0, 1.0)
    end

    @testset "Taylor" begin
        𝒯₂, 𝒯₃ = Taylor(2), Taylor(3)
        @test_throws DomainError Taylor(-1)
        @test !(𝒯₂ == 𝒯₃)
        @test 𝒯₂ ⊆ 𝒯₃
        @test !(𝒯₃ ⊆ 𝒯₂)
        @test 𝒯₂ ∩ 𝒯₃ == 𝒯₂ # intersection keeps the smaller order
        @test 𝒯₂ ∪ 𝒯₃ == 𝒯₃ # union keeps the larger order
        @test order(𝒯₂) == 2
        @test dimension(𝒯₂) == 3 # orders 0,1,2
        @test RadiiPolynomial._firstindex(𝒯₂) == 0
        @test RadiiPolynomial._lastindex(𝒯₂) == 2
        @test indices(𝒯₂) == 0:2
        @test collect(indices(𝒯₂)) == [0, 1, 2] # ascending canonical order
        @test RadiiPolynomial._compatible_space_with_constant_index(𝒯₂) == 𝒯₂
        @test RadiiPolynomial._findindex_constant(𝒯₂) == 0
        @test RadiiPolynomial._findposition(1, 𝒯₂) == 2 # order 1 sits at position 2 (1-indexed)
        @test RadiiPolynomial._findposition(0:2, 𝒯₂) == 1:3
        @test RadiiPolynomial._findposition([0, 2], 𝒯₂) == [1, 3]
        @test RadiiPolynomial._findposition(:, 𝒯₂) == Colon()
        @test RadiiPolynomial._findposition(Taylor(1), 𝒯₂) == 1:2
        @test RadiiPolynomial._iscompatible(𝒯₂, 𝒯₃) # compatible regardless of order
        @test interval(Float64, 𝒯₂) == interval(𝒯₂) == 𝒯₂ # `Taylor` is its own interval counterpart
    end

    @testset "Fourier" begin
        ℱ₂, ℱ₃ = Fourier(2, 1.0), Fourier(3, 1.0)
        @test_throws DomainError Fourier(-1, 1.0)
        @test_throws DomainError Fourier(1, -1.0)
        @test_throws DomainError Fourier(1, interval(-1.0, 1.0)) # `inf(frequency) < 0`
        @test !(ℱ₂ == ℱ₃)
        @test ℱ₂ ⊆ ℱ₃
        @test !(ℱ₃ ⊆ ℱ₂)
        @test ℱ₂ ∩ ℱ₃ == ℱ₂
        @test ℱ₂ ∪ ℱ₃ == ℱ₃
        @test_throws ArgumentError intersect(ℱ₂, Fourier(2, 2.0)) # frequencies must match
        @test_throws ArgumentError union(ℱ₂, Fourier(2, 2.0))
        @test order(ℱ₂) == 2
        @test frequency(ℱ₂) == 1.0
        @test dimension(ℱ₂) == 5 # orders -2,...,2
        @test RadiiPolynomial._firstindex(ℱ₂) == -2
        @test RadiiPolynomial._lastindex(ℱ₂) == 2
        @test indices(ℱ₂) == -2:2
        @test collect(indices(ℱ₂)) == [-2, -1, 0, 1, 2] # ascending canonical order
        @test RadiiPolynomial._findindex_constant(ℱ₂) == 0
        @test RadiiPolynomial._findposition(-1, ℱ₂) == 2
        @test RadiiPolynomial._findposition(0:1, ℱ₂) == 3:4
        @test RadiiPolynomial._findposition([-2, 2], ℱ₂) == [1, 5]
        @test RadiiPolynomial._findposition(:, ℱ₂) == Colon()
        @test RadiiPolynomial._iscompatible(ℱ₂, ℱ₃) # same frequency
        @test !RadiiPolynomial._iscompatible(ℱ₂, Fourier(2, 2.0)) # different frequency
        # promotion
        @test convert(Fourier{Float64}, ℱ₂) == ℱ₂
        @test convert(Fourier{Int}, ℱ₂) == Fourier(2, 1)
        @test promote_type(Fourier{Float64}, Fourier{Float64}) == Fourier{Float64}
        @test promote_type(Fourier{Float64}, Fourier{Int}) == Fourier{Float64}
        # interval overload: frequency becomes an `Interval`
        ℱ₂I = interval(ℱ₂)
        @test order(ℱ₂I) == 2
        @test frequency(ℱ₂I) isa Interval
        @test ℱ₂I == Fourier(2, interval(1.0)) # `==` uses `isequal_interval` internally (safe)
        @test interval(Float64, ℱ₂) == ℱ₂I
        # a Fourier space built with an interval frequency that does not straddle zero is valid
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
        @test_throws ArgumentError TensorSpace() # at least one `BaseSpace` is required
        𝑇 = TensorSpace((𝒯₂, ℱ₂, 𝒞₂))
        @test spaces(𝑇) == (𝒯₂, ℱ₂, 𝒞₂)
        @test nspaces(𝑇) == 3
        # flattening/associativity of `⊗`
        @test 𝒯₂ ⊗ ℱ₂ ⊗ 𝒞₂ == (𝒯₂ ⊗ ℱ₂) ⊗ 𝒞₂ == 𝒯₂ ⊗ (ℱ₂ ⊗ 𝒞₂) == TensorSpace(𝒯₂, ℱ₂, 𝒞₂) == 𝑇
        @test 𝑇 ⊗ 𝑇 == TensorSpace((𝒯₂, ℱ₂, 𝒞₂, 𝒯₂, ℱ₂, 𝒞₂)) # TensorSpace ⊗ TensorSpace flattens both
        @test (𝒯₂ ⊗ ℱ₂) ⊗ 𝒞₂ == TensorSpace((𝒯₂, ℱ₂)) ⊗ 𝒞₂ # TensorSpace ⊗ BaseSpace flattens the left
        @test 𝒯₂ ⊗ (ℱ₂ ⊗ 𝒞₂) == 𝒯₂ ⊗ TensorSpace((ℱ₂, 𝒞₂)) # BaseSpace ⊗ TensorSpace flattens the right
        @test 𝑇[1] == 𝒯₂
        @test 𝑇[1:3] == 𝑇[[1, 2, 3]] == 𝑇[:] == 𝑇
        @test Base.tail(𝑇) == TensorSpace((ℱ₂, 𝒞₂))
        @test Base.front(𝑇) == TensorSpace((𝒯₂, ℱ₂))
        @test 𝑇 ⊆ Taylor(3) ⊗ ℱ₂ ⊗ 𝒞₂
        @test !(Taylor(3) ⊗ ℱ₂ ⊗ 𝒞₂ ⊆ 𝑇)
        @test 𝑇 ∩ (Taylor(3) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(3)) == 𝒯₂ ⊗ Fourier(1, 1.0) ⊗ 𝒞₂ # componentwise ∩
        @test 𝑇 ∪ (Taylor(3) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(3)) == Taylor(3) ⊗ ℱ₂ ⊗ Chebyshev(3) # componentwise ∪
        @test !(𝑇 == TensorSpace((𝒯₂, ℱ₂))) # mismatched number of factors falls back to `false`
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
        @test_throws MethodError frequency(𝑇) # `Taylor`/`Chebyshev` have no frequency
        @test frequency(𝑇, 2) == 1.0
        @test RadiiPolynomial._iscompatible(𝑇, Taylor(5) ⊗ Fourier(2, 1.0) ⊗ Chebyshev(5))
        @test !RadiiPolynomial._iscompatible(𝑇, Taylor(5) ⊗ Fourier(2, 2.0) ⊗ Chebyshev(5)) # frequency mismatch
        # promotion
        @test convert(TensorSpace{Tuple{Taylor,Fourier{Float64},Chebyshev}}, 𝑇) == 𝑇
        @test convert(TensorSpace{Tuple{Taylor,Fourier{Int},Chebyshev}}, 𝑇) == Taylor(2) ⊗ Fourier(2, 1) ⊗ Chebyshev(2)
        @test promote_type(TensorSpace{Tuple{Taylor,Fourier{Float64}}}, TensorSpace{Tuple{Taylor,Fourier{Int}}}) ==
            TensorSpace{Tuple{Taylor,Fourier}} # `promote_type` on the underlying `Tuple` types is not elementwise
        # interval overload
        @test interval(Float64, 𝑇) == interval(𝑇) == Taylor(2) ⊗ interval(ℱ₂) ⊗ Chebyshev(2)
        # `_checkbounds_indices` on a `TensorIndices` delegates to the underlying tuple of ranges (line 266)
        @test RadiiPolynomial._checkbounds_indices(indices(𝑇), 𝑇)
        @test !RadiiPolynomial._checkbounds_indices(TensorIndices((0:3, -2:2, 0:2)), 𝑇) # order 3 exceeds Taylor(2)'s order 2
        # `_prettystring` for a 3-factor `TensorSpace` recurses via the generic method (line 318)
        # onto the 2-factor method (line 319) for the tail
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
        # `TensorIndices` has no custom `==`; the derived default only works out-of-the-box for
        # `isbits` contents (ranges), so compare the materialized iteration instead
        @test collect(union(TensorIndices((0:1, 0:1)), TensorIndices((1:2, 1:2)))) == collect(TensorIndices(([0, 1, 2], [0, 1, 2])))
        # `getindex` goes through `CartesianIndices(map(length, ti.indices))`, so both a
        # linear index and a `CartesianIndex` agree with the materialized iteration order
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
        @test 𝒯₁¹^1 == CartesianPower(𝒯₁¹, 1) # `^` on a `CartesianPower` nests it
        @test 𝒯₁³[1] == Taylor(1)
        @test 𝒯₁³[1:2] == 𝒯₁³[[1, 2]] == CartesianPower(Taylor(1), 2)
        @test 𝒯₁³[:] == 𝒯₁³
        @test_throws BoundsError 𝒯₁³[4]
        @test_throws BoundsError 𝒯₁³[1:4]
        @test_throws BoundsError 𝒯₁³[[1, 4]]
        @test 𝒯₁³ ⊆ Taylor(2)^3
        @test !(𝒯₁³ ⊆ Taylor(2)^2) # mismatched number of cartesian products
        @test 𝒯₁³ ∩ Taylor(2)^3 == 𝒯₁³
        @test 𝒯₁³ ∪ Taylor(2)^3 == Taylor(2)^3
        @test_throws ArgumentError intersect(𝒯₁³, Taylor(1)^2) # mismatched n
        @test_throws ArgumentError union(𝒯₁³, Taylor(1)^2)
        @test dimension(𝒯₁³) == 2 * 3 == 6 # dim(Taylor(1)) = 2, repeated 3 times
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
        @test RadiiPolynomial._component_findposition([1, 3], 𝒯₁³) == [1, 2, 5, 6] # union of non-contiguous blocks
        @test RadiiPolynomial._iscompatible(𝒯₁³, Taylor(9)^3)
        @test !RadiiPolynomial._iscompatible(𝒯₁³, Taylor(9)^2) # mismatched n
        # `_findposition` on a smaller `CartesianPower` selects the leading coefficients of every block
        s = Taylor(2)^3
        α = Taylor(1)^3
        @test RadiiPolynomial._findposition(α, s) == [1, 2, 4, 5, 7, 8]
        a = Sequence(s, collect(1.0:9.0)) # dimension(s) == 9
        @test coefficients(a[α]) == [1.0, 2.0, 4.0, 5.0, 7.0, 8.0]
        # promotion
        @test convert(CartesianPower{Taylor}, 𝒯₁³) == 𝒯₁³
        @test promote_type(CartesianPower{Taylor}, CartesianPower{Chebyshev}) == CartesianPower{BaseSpace} # common supertype
        # interval overload
        ℱ₂² = Fourier(2, 1.0)^2
        @test interval(Float64, ℱ₂²) == interval(ℱ₂²) == Fourier(2, interval(1.0))^2
        # `_prettystring` on a `CartesianPower` of a `TensorSpace` parenthesizes the tensor product (line 698)
        𝒯ℱ = Taylor(1) ⊗ Fourier(1, 1.0)
        @test RadiiPolynomial._prettystring(𝒯ℱ^2, true) == "(Taylor(1) ⊗ Fourier(1, 1.0))²"
    end

    @testset "CartesianProduct" begin
        @test_throws ArgumentError CartesianProduct() # at least one `VectorSpace` is required
        𝒯₁¹ = CartesianPower(Taylor(1), 1)
        𝒞₂² = CartesianPower(Chebyshev(2), 2)
        ℱ₃³ = CartesianPower(Fourier(3, 1.0), 3)
        𝑃 = CartesianProduct((𝒯₁¹, 𝒞₂², ℱ₃³))
        𝑃′ = CartesianProduct((CartesianPower(Taylor(2), 1), CartesianPower(Chebyshev(3), 2), ℱ₃³))
        @test spaces(𝒯₁¹) == [Taylor(1)] # `spaces` on a `CartesianPower` returns a `Vector`
        @test spaces(𝑃) == (𝒯₁¹, 𝒞₂², ℱ₃³) # `spaces` on a `CartesianProduct` returns a `Tuple`
        @test nspaces(ℱ₃³) == nspaces(𝑃) == 3
        # flattening/associativity of `×`
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
        @test !(𝑃 == (𝒯₁¹ × 𝒞₂²)) # mismatched number of factors falls back to `false`
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
        @test_throws MethodError frequency(𝑃) # `Taylor`/`Chebyshev` blocks have no frequency
        @test frequency(𝑃, 3) == frequency(ℱ₃³) == [1.0, 1.0, 1.0]
        @test frequency(ℱ₃³, 1) == 1.0
        @test RadiiPolynomial._component_findposition(1, 𝒯₁¹) == RadiiPolynomial._component_findposition(1:1, 𝒯₁¹) == 1:2
        @test RadiiPolynomial._component_findposition(2, 𝒞₂²) == RadiiPolynomial._component_findposition(2:2, 𝒞₂²) == 4:6
        @test RadiiPolynomial._component_findposition(1, 𝑃) == RadiiPolynomial._component_findposition(1:1, 𝑃) == 1:2
        @test RadiiPolynomial._component_findposition(2, 𝑃) == RadiiPolynomial._component_findposition(2:2, 𝑃) == 3:8
        @test RadiiPolynomial._component_findposition(1:2, 𝑃) == 1:8
        @test RadiiPolynomial._component_findposition([1, 3], 𝑃) == vcat(1:2, 9:29) # union of non-contiguous blocks
        @test RadiiPolynomial._iscompatible(𝑃, 𝑃′)
        # `_findposition` on a smaller `CartesianProduct` picks matching positions blockwise
        s2 = Taylor(1) × Fourier(1, 1.0) × Chebyshev(1)
        α2 = Taylor(0) × Fourier(0, 1.0) × Chebyshev(0)
        @test RadiiPolynomial._findposition(α2, s2) == [1, 4, 6]
        a2 = Sequence(s2, collect(1.0:7.0))
        @test coefficients(a2[α2]) == [1.0, 4.0, 6.0]
        # promotion
        @test convert(CartesianProduct{Tuple{CartesianPower{Taylor}, CartesianPower{Chebyshev}, CartesianPower{Fourier{Float64}}}}, 𝑃) == 𝑃
        @test convert(CartesianProduct{Tuple{CartesianPower{Taylor}, CartesianPower{Chebyshev}, CartesianPower{Fourier{Int}}}}, 𝑃) ==
            𝒯₁¹ × 𝒞₂² × Fourier(3, 1)^3
        # interval overload
        @test interval(Float64, 𝑃) == interval(𝑃) == 𝒯₁¹ × 𝒞₂² × CartesianPower(interval(Fourier(3, 1.0)), 3)
        # `_prettystring` for a 3-factor `CartesianProduct` of plain `BaseSpace` recurses via the
        # generic method (line 851) onto the 2-factor method (line 852); each factor is neither a
        # `TensorSpace` nor a `CartesianProduct`, so it falls through `_prettystring_cartesian`'s
        # generic `VectorSpace` fallback (line 854)
        𝒯ℱ𝒞 = Taylor(1) × Fourier(1, 1.0) × Chebyshev(1)
        @test RadiiPolynomial._prettystring(𝒯ℱ𝒞, true) == "Taylor(1) × Fourier(1, 1.0) × Chebyshev(1)"
    end

    @testset "_supscript / _supscript_digit" begin
        # single-digit fast path (lines 945-946), covering every digit glyph incl. 4-9 (lines 965-970)
        glyphs = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹']
        for i in 0:9
            @test RadiiPolynomial._supscript_digit(i) == glyphs[i+1]
            @test RadiiPolynomial._supscript(i) == glyphs[i+1]
        end
        # multi-digit path (lines 948-956): digits are peeled off with `divrem` and written
        # most-significant-first
        @test RadiiPolynomial._supscript(12) == "¹²"
        @test RadiiPolynomial._supscript(100) == "¹⁰⁰" # exercises an interior zero digit
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
        @test !(pow == prod_mismatch_n) # mismatched nspaces
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
        @test !RadiiPolynomial._iscompatible(pow, prod_mismatch_n) # nspaces mismatch
    end

    @testset "Nested CartesianSpace _findposition" begin
        # CartesianProduct containing a CartesianPower of a BaseSpace: works correctly since
        # every "copy" inside the power is literally the same (uniform) leaf space
        s3 = Taylor(1)^2 × Chebyshev(1)
        α3 = Taylor(0)^2 × Chebyshev(0)
        @test RadiiPolynomial._findposition(α3, s3) == [1, 3, 5]
        a3 = Sequence(s3, collect(1.0:6.0))
        @test coefficients(a3[α3]) == [1.0, 3.0, 5.0]

        # CartesianPower of a CartesianProduct with n ≥ 2 and more than one distinct sub-space:
        # `_iterate_space(s::CartesianPower, i) = _iterate_space(space(s), i)` (vector_spaces.jl)
        # reduces `i` modulo the period `_deep_nspaces(space(s))`, so beyond the first copy it
        # still correctly labels which leaf space occupies a given slot
        s4 = (Fourier(1, 1.0) × Taylor(1))^2
        α4 = (Fourier(0, 1.0) × Taylor(0))^2
        a4 = Sequence(s4, collect(1.0:10.0))
        # hand-computed via `component`: copy 1 is [1,2,3,4,5] (Fourier zero mode at 2, Taylor
        # zero mode at 4), copy 2 is [6,7,8,9,10] (Fourier zero mode at 7, Taylor zero mode at 9)
        @test RadiiPolynomial._findposition(α4, s4) == [2, 4, 7, 9]
        @test coefficients(a4[α4]) == [2.0, 4.0, 7.0, 9.0]
        # cross-check against `component`: each copy's zero-mode positions match its own
        # (Fourier(1,1.0) × Taylor(1)) block indexed by (Fourier(0,1.0) × Taylor(0))
        β4 = Fourier(0, 1.0) × Taylor(0)
        @test coefficients(component(a4, 1)[β4]) == [2.0, 4.0]
        @test coefficients(component(a4, 2)[β4]) == [7.0, 9.0]
    end

    @testset "_iscompatible" begin
        @test RadiiPolynomial._iscompatible(ScalarSpace(), ScalarSpace())
        @test RadiiPolynomial._iscompatible(Taylor(1), Taylor(5)) # order-independent
        @test RadiiPolynomial._iscompatible(Chebyshev(1), Chebyshev(9))
        @test RadiiPolynomial._iscompatible(Fourier(1, 1.0), Fourier(5, 1.0))
        @test !RadiiPolynomial._iscompatible(Fourier(1, 1.0), Fourier(1, 2.0))
        𝑇 = Taylor(1) ⊗ Fourier(1, 1.0)
        @test RadiiPolynomial._iscompatible(𝑇, Taylor(5) ⊗ Fourier(5, 1.0))
        @test !RadiiPolynomial._iscompatible(𝑇, Taylor(5) ⊗ Fourier(5, 2.0))
    end

end
