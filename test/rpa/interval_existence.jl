@testset "Interval existence" begin

    #= interval_of_existence(Y, Z₁, R): the radii-polynomial p(r) = Y + (Z₁-1)r is affine
       and decreasing (since Z₁ < 1), so p(r) ≤ 0 ⟺ r ≥ r* := Y/(1-Z₁). The validated
       interval is therefore [r*, R] provided 0 ≤ r* ≤ R. =#
    @testset "2-arg (Y, Z₁, R): success" begin
        Y, Z1, R = interval(0.25), interval(0.5), 2.0
        # r* = 0.25 / (1 - 0.5) = 0.5 exactly (both operands exactly representable)
        iv, ok = interval_of_existence(Y, Z1, R)
        @test ok == true
        @test isequal_interval(iv, interval(0.5, 2.0))
        @test isguaranteed(iv) == true
        @test decoration(iv) == decoration(Y) == decoration(Z1)
        @test iv isa Interval{Float64}

        # plain-number call form forwards through `interval(...)` and gives the same result
        iv2, ok2 = interval_of_existence(0.25, 0.5, 2.0)
        @test ok2 == true
        @test isequal_interval(iv2, iv)
    end

    @testset "2-arg (Y, Z₁, R): failures" begin
        # Z₁ ≥ 1 violates the strict contraction requirement Z₁ < 1
        iv, ok = interval_of_existence(interval(0.1), interval(1.0), 1.0)
        @test ok == false
        @test isempty_interval(iv)

        # Y < 0 is invalid (Y must be nonnegative)
        iv2, ok2 = interval_of_existence(interval(-0.1), interval(0.5), 1.0)
        @test ok2 == false
        @test isempty_interval(iv2)

        # R < 0 is an invalid threshold
        iv3, ok3 = interval_of_existence(interval(0.1), interval(0.5), -1.0)
        @test ok3 == false
        @test isempty_interval(iv3)

        # R = NaN is likewise an invalid threshold
        iv4, ok4 = interval_of_existence(interval(0.1), interval(0.5), NaN)
        @test ok4 == false
        @test isempty_interval(iv4)

        # valid Y, Z₁ but the root r* = 1.0/(1-0.5) = 2.0 exceeds R = 0.1
        iv5, ok5 = interval_of_existence(interval(1.0), interval(0.5), 0.1)
        @test ok5 == false
        @test isempty_interval(iv5)
    end

    @testset "2-arg: guarantee / decoration propagation" begin
        Ybare = bareinterval(interval(0.25))
        Ylow = IntervalArithmetic._unsafe_interval(Ybare, trv, false) # NG, decoration trv
        Z1 = interval(0.5) # guaranteed, decoration com

        iv, ok = interval_of_existence(Ylow, Z1, 2.0)
        @test ok == true
        @test isequal_interval(iv, interval(0.5, 2.0)) # bounds unaffected by guarantee/decoration
        @test isguaranteed(iv) == false # false (Ylow) & true (Z1) = false
        @test decoration(iv) == trv     # min(trv, com) = trv
    end

    @testset "2-arg: verbose flag emits @info, silent otherwise" begin
        @test_logs (:info,) interval_of_existence(interval(0.25), interval(0.5), 2.0; verbose = true)
        @test_logs (:info,) interval_of_existence(interval(0.1), interval(1.0), 1.0; verbose = true) # failure path also logs
        @test_logs interval_of_existence(interval(0.25), interval(0.5), 2.0; verbose = false)
    end

    #= interval_of_existence(Y, Z₁, Z₂, R): p(r) = Y + (Z₁-1)r + (Z₂/2)r² is a upward
       parabola (Z₂ > 0). With b = Z₁-1, its roots are r₁,₂ = (-b ∓ √Δ)/Z₂,
       Δ = b² - 2·Z₂·Y. p(r) ≤ 0 on [r₁, r₂]; the validated interval is
       [r₁, min(R, x)] where x ≤ -b/Z₂ (the parabola's vertex, where the *second*
       radii-polynomial condition Z₁ + Z₂ r < 1 stops holding), further capped by R. =#
    @testset "3-arg (Y, Z₁, Z₂, R): success" begin
        # Choose Z₁ = 0, Z₂ = 0.5 ⟹ b = -1, Δ = 1 - 2·0.5·0.75 = 0.25, √Δ = 0.5
        # r₁ = (1 - 0.5)/0.5 = 1, r₂ = (1 + 0.5)/0.5 = 3; vertex -b/Z₂ = 1/0.5 = 2
        Y, Z1, Z2 = interval(0.75), interval(0.0), interval(0.5)

        iv, ok = interval_of_existence(Y, Z1, Z2, 5.0)
        @test ok == true
        @test inf(iv) == 1.0
        # upper end is the largest float below the vertex (2 < R = 5) at which Z₂ r < 1 - Z₁ is certified
        @test sup(iv) == prevfloat(2.0)
        @test sup(Z2 * sup(iv)) < inf(one(Z1) - Z1)
        @test !(sup(Z2 * nextfloat(sup(iv))) < inf(one(Z1) - Z1))
        @test isguaranteed(iv) == true
        @test decoration(iv) == decoration(Y) == decoration(Z1) == decoration(Z2)

        # R-cap: R = 1 < vertex (2) ⟹ upper endpoint becomes R itself
        ivR, okR = interval_of_existence(Y, Z1, Z2, 1.0)
        @test okR == true
        @test isequal_interval(ivR, interval(1.0, 1.0))

        # plain-number call form gives the same result
        ivn, okn = interval_of_existence(0.75, 0.0, 0.5, 5.0)
        @test okn == true
        @test isequal_interval(ivn, iv)
    end

    @testset "3-arg: Z₂ thin-zero falls back to the 2-arg formula" begin
        Y, Z1, R = interval(0.25), interval(0.5), 2.0
        iv2, ok2 = interval_of_existence(Y, Z1, R)
        iv3, ok3 = interval_of_existence(Y, Z1, interval(0.0), R)
        @test ok3 == ok2 == true
        @test isequal_interval(iv3, iv2)
    end

    @testset "3-arg: failures" begin
        # negative discriminant: Y=2, Z₁=0.5, Z₂=1 ⟹ b=-0.5, Δ = 0.25 - 2·1·2 = -3.75 < 0
        iv, ok = interval_of_existence(interval(2.0), interval(0.5), interval(1.0), 10.0)
        @test ok == false
        @test isempty_interval(iv)

        # double root exactly at the vertex: Y=0.25, Z₁=0, Z₂=2 ⟹ Δ=0, r₁=r₂=0.5,
        # but then Z₁ + Z₂·r₁ = 0 + 2·0.5 = 1 is not < 1 ⟹ contraction fails
        ivd, okd = interval_of_existence(interval(0.25), interval(0.0), interval(2.0), 1.0)
        @test okd == false
        @test isempty_interval(ivd)

        # valid roots but R too small: r₁ = 1 > R = 0.5
        ivr, okr = interval_of_existence(interval(0.75), interval(0.0), interval(0.5), 0.5)
        @test okr == false
        @test isempty_interval(ivr)

        # Z₂ < 0 is invalid
        ivz, okz = interval_of_existence(interval(0.1), interval(0.1), interval(-0.1), 1.0)
        @test okz == false
        @test isempty_interval(ivz)

        # R < 0 is an invalid threshold
        ivR2, okR2 = interval_of_existence(interval(0.1), interval(0.1), interval(0.1), -1.0)
        @test okR2 == false
        @test isempty_interval(ivR2)
    end

    @testset "3-arg: verbose flag emits @info, silent otherwise" begin
        @test_logs (:info,) interval_of_existence(interval(0.75), interval(0.0), interval(0.5), 5.0; verbose = true)
        @test_logs (:info,) interval_of_existence(interval(2.0), interval(0.5), interval(1.0), 10.0; verbose = true) # failure path
        @test_logs interval_of_existence(interval(0.75), interval(0.0), interval(0.5), 5.0; verbose = false)
    end

    #= 3-arg: when the inputs are not natively floating-point (e.g. Rational), converting the
       exact vertex z = -b/Z₂ to Float64 can round UP past the true value; the implementation
       then walks back down with `prevfloat` until Z₂ x < 1 - Z₁ is certified. Here Z₁ = 0//1,
       Z₂ = 10//1 ⟹ b = -1//1 and z = -b/Z₂ = 1//10 exactly; Float64(1//10) = 0.1 rounds UP
       (0.1 > 1//10 as an exact rational comparison), so the correction loop must fire. =#
    @testset "3-arg: Rational inputs force the vertex float-rounding correction (prevfloat loop)" begin
        Y, Z1, Z2 = interval(1//100), interval(0//1), interval(10//1)
        b = Z1 - one(Z1)
        z = inf(-b / Z2) # exact rational vertex, 1//10
        @test z == 1//10
        @test float(z) > z # naive Float64 conversion overestimates the vertex: triggers the loop

        iv, ok = interval_of_existence(Y, Z1, Z2, 1.0)
        @test ok == true
        @test iv isa Interval{Float64}
        @test sup(iv) < 0.1 # never unsoundly reports the naive (rounded-up) float vertex
        @test sup(Z2 * sup(iv)) < inf(one(Z1) - Z1) # the contraction is certified at the upper endpoint
        @test !(sup(Z2 * nextfloat(sup(iv))) < inf(one(Z1) - Z1)) # and the endpoint is the largest such double
    end

    @testset "set_of_radii (M-dimensional)" begin
        #= a diagonal Z with an entry ≥ 1 makes P₁(r) = Y₁ + (Z₁₁ - 1) r₁ = Y₁ + r₁ > 0 for every
           r₁ ≥ 0, so no set of radii can make the radii polynomials simultaneously nonpositive =#
        @testset "radii polynomials never nonpositive" begin
            Y1 = [interval(1.0)]
            Z1_ = fill(interval(2.0), 1, 1)
            W1 = zeros(Interval{Float64}, 1, 1, 1)
            rmin1, eta1, success1 = RadiiPolynomial.set_of_radii(Y1, Z1_, W1, [100.0]; verbose = true)
            @test success1 == false
            @test isempty(rmin1) && isempty(eta1)

            Y2 = [interval(1.0), interval(1.0)]
            Z2_ = [interval(2.0) interval(0.0); interval(0.0) interval(2.0)]
            W2 = zeros(Interval{Float64}, 2, 2, 2)
            rmin2, eta2, success2 = RadiiPolynomial.set_of_radii(Y2, Z2_, W2, [100.0, 100.0]; verbose = true)
            @test success2 == false
            @test isempty(rmin2) && isempty(eta2)

            # Z = I makes the Jacobian of the radii polynomials singular at the root: no exception
            Zs = [interval(1.0) interval(0.0); interval(0.0) interval(1.0)]
            rmins, etas, successs = RadiiPolynomial.set_of_radii([interval(0.01), interval(0.01)], Zs, W2, [10.0, 10.0])
            @test successs == false
            @test isempty(rmins) && isempty(etas)
        end

        #= The decoupled scalar quadratic Y = 0.75, Z₁ = 0, Z₂ = 0.5 has root r₁ = 1, where the
           contraction holds. Taking R below that radius, the radii are found and contract yet
           exceed R. With R large enough (5) the same data succeeds. =#
        @testset "radii exceed R" begin
            Y = [interval(0.75), interval(0.75)]
            Z = zeros(Interval{Float64}, 2, 2)
            W = zeros(Interval{Float64}, 2, 2, 2)
            W[1, 1, 1] = interval(0.5)
            W[2, 2, 2] = interval(0.5)

            rmin_ok, eta_ok, success_ok = RadiiPolynomial.set_of_radii(Y, Z, W, [5.0, 5.0])
            @test success_ok == true
            @test sup.(rmin_ok) ≈ [1.0, 1.0] atol = 1e-6 # matches the hand-computed root of Y + 0.25r² - r = 0
            @test all(isguaranteed, rmin_ok)
            @test all(r -> decoration(r) == com, rmin_ok)

            rmin_bad, eta_bad, success_bad = RadiiPolynomial.set_of_radii(Y, Z, W, [0.5, 0.5]; verbose = true)
            @test success_bad == false
            @test isempty(rmin_bad) && isempty(eta_bad)

            # the guarantee flag of the inputs propagates to the radii
            Y_ng = [interval(0.75) + 0, interval(0.75)]
            @test !isguaranteed(Y_ng[1])
            rmin_ng, _, success_ng = RadiiPolynomial.set_of_radii(Y_ng, Z, W, [5.0, 5.0])
            @test success_ng == true
            @test !any(isguaranteed, rmin_ng)

            # inconsistent sizes are rejected, not broadcast
            _, _, success_sz = RadiiPolynomial.set_of_radii(Y, Z, W, [5.0]; verbose = true)
            @test success_sz == false

            # BigFloat inputs are accepted and the radii are BigFloat intervals
            rmin_big, _, success_big = RadiiPolynomial.set_of_radii(interval.(big.([0.75, 0.75])), zeros(Interval{BigFloat}, 2, 2), Interval{BigFloat}.(W), [5.0, 5.0])
            @test success_big == true
            @test eltype(rmin_big) == Interval{BigFloat}
        end

        #= "inclusion found, but no contraction": using an intentionally asymmetric W (only
           W[1,1,2] is set, not its "mirror" W[1,2,1]) decouples the contraction matrix
           A(r) = Z + Σⱼ W[:,:,j] rⱼ from the (symmetrized) Jacobian Newton differentiates.
           With Z = 0 and W[1,1,2] = 8 (all other entries 0), the quadratic term for component 1
           is (1/2)ΣᵢⱼW₁ᵢⱼrᵢrⱼ = 4r₁r₂, so
             P₁(r) = Y₁ + 4r₁r₂ - r₁,   P₂(r) = Y₂ - r₂  (component 2 is unaffected by W)
           whose exact positive root is r₂ = Y₂ and r₁ = Y₁/(1-4r₂). With Y₁=0.01, Y₂=0.2:
           r₂ = 0.2, r₁ = 0.01/(1-0.8) = 0.05 (both positive). But A(r) = [8r₂ 0; 0 0], so at
           r=(0.05,0.2): A(r) = [1.6 0; 0 0], and no positive η satisfies A(r) η < η since
           1.6 η₁ < η₁ is impossible: the contraction fails even though P(r) ≤ 0. =#
        @testset "inclusion without contraction (asymmetric W, hand-verified 2×2)" begin
            # direct sanity check of the test vector on the hand-verified matrix A(r)
            A = [interval(1.6) interval(0.0); interval(0.0) interval(0.0)]
            η = RadiiPolynomial._perron_vector(A)
            @test η[1] > η[2] > 0 # dominant eigenvector points along e₁ = (1, 0), kept positive
            @test sup(maximum(A * η ./ η)) ≈ 1.6 # Collatz-Wielandt bound equals the Perron root 1.6 of [1.6 0; 0 0]

            Y = [interval(0.01), interval(0.2)]
            Z = zeros(Interval{Float64}, 2, 2)
            W = zeros(Interval{Float64}, 2, 2, 2)
            W[1, 1, 2] = interval(8.0)
            rmin, eta, success = RadiiPolynomial.set_of_radii(Y, Z, W, [50.0, 50.0]; verbose = true)
            @test success == false
            @test isempty(rmin) && isempty(eta)
        end
    end
end
