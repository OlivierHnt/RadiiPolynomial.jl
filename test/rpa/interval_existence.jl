function _existence_fails(args...)
    ie, success = interval_of_existence(args...)
    return !success && isempty_interval(ie)
end

function _radii_fail(args...; kwargs...)
    rmin, η, success = RadiiPolynomial.set_of_radii(args...; kwargs...)
    return !success && isempty(rmin) && isempty(η)
end

@testset "Interval existence" begin

    #= p(r) = Y + (Z₁ - 1) r is affine and decreasing when Z₁ < 1, so p(r) ≤ 0 on [r*, R] with
       r* = Y / (1 - Z₁), provided r* ≤ R =#
    @testset "first order (Y, Z₁, R)" begin
        Y, Z₁ = interval(0.25), interval(0.5)   # r* = 0.5 exactly
        ie, success = interval_of_existence(Y, Z₁, 2.0)
        @test success
        @test isequal_interval(ie, interval(0.5, 2.0))
        @test isguaranteed(ie)
        @test decoration(ie) == decoration(Y) == decoration(Z₁)
        @test ie isa Interval{Float64}

        # real numbers are converted to intervals and remain guaranteed
        ie_real, success_real = interval_of_existence(0.25, 0.5, 2.0)
        @test success_real && isequal_interval(ie_real, ie) && isguaranteed(ie_real)

        # an interval threshold is taken by its lower end
        ie_R, success_R = interval_of_existence(Y, Z₁, interval(2.0, 20.0))
        @test success_R && isequal_interval(ie_R, ie)
        @test _existence_fails(Y, Z₁, interval(0.1, 1.0))

        @test _existence_fails(interval(0.1), interval(1.0), 1.0)    # Z₁ ≥ 1 is not a contraction
        @test _existence_fails(interval(-0.1), interval(0.5), 1.0)   # Y < 0
        @test _existence_fails(interval(0.1), interval(0.5), -1.0)   # R < 0
        @test _existence_fails(interval(0.1), interval(0.5), NaN)
        @test _existence_fails(interval(1.0), interval(0.5), 0.1)    # r* = 2 exceeds R

        # the guarantee and the decoration of the result are the weakest of the inputs
        Y_trv = IntervalArithmetic._unsafe_interval(bareinterval(Y), trv, false)
        ie_trv, success_trv = interval_of_existence(Y_trv, Z₁, 2.0)
        @test success_trv && isequal_interval(ie_trv, ie)
        @test !isguaranteed(ie_trv)
        @test decoration(ie_trv) == trv

        @test_logs (:info,) interval_of_existence(Y, Z₁, 2.0; verbose = true)
        @test_logs (:info,) interval_of_existence(interval(0.1), interval(1.0), 1.0; verbose = true)
        @test_logs interval_of_existence(Y, Z₁, 2.0; verbose = false)
    end

    #= p(r) = Y + (Z₁ - 1) r + (Z₂ / 2) r² is an upward parabola. With b = Z₁ - 1 and
       Δ = b² - 2 Z₂ Y its roots are r₁,₂ = (-b ∓ √Δ) / Z₂. The interval is [r₁, min(R, x)] where
       x is the largest float below the vertex -b / Z₂ at which Z₁ + Z₂ x < 1 is certified. =#
    @testset "second order (Y, Z₁, Z₂, R)" begin
        # b = -1, Δ = 0.25, r₁ = 1, r₂ = 3 and the vertex is 2
        Y, Z₁, Z₂ = interval(0.75), interval(0.0), interval(0.5)
        ie, success = interval_of_existence(Y, Z₁, Z₂, 5.0)
        @test success
        @test inf(ie) == 1.0
        @test sup(ie) == prevfloat(2.0)
        @test sup(Z₂ * sup(ie)) < inf(one(Z₁) - Z₁)
        @test !(sup(Z₂ * nextfloat(sup(ie))) < inf(one(Z₁) - Z₁))
        @test isguaranteed(ie)
        @test decoration(ie) == decoration(Y) == decoration(Z₁) == decoration(Z₂)

        # R below the vertex caps the interval
        ie_R, success_R = interval_of_existence(Y, Z₁, Z₂, 1.0)
        @test success_R && isequal_interval(ie_R, interval(1.0, 1.0))

        ie_real, success_real = interval_of_existence(0.75, 0.0, 0.5, 5.0)
        @test success_real && isequal_interval(ie_real, ie)

        # a thin zero Z₂ reduces to the first order
        ie₀, success₀ = interval_of_existence(interval(0.25), interval(0.5), interval(0.0), 2.0)
        @test success₀ && isequal_interval(ie₀, interval(0.5, 2.0))

        @test _existence_fails(interval(2.0), interval(0.5), interval(1.0), 10.0)   # Δ = -3.75 < 0
        # double root at the vertex r₁ = r₂ = 0.5, where Z₁ + Z₂ r₁ = 1 is not a contraction
        @test _existence_fails(interval(0.25), interval(0.0), interval(2.0), 1.0)
        @test _existence_fails(Y, Z₁, Z₂, 0.5)                                       # r₁ = 1 exceeds R
        @test _existence_fails(interval(0.01), interval(1.0), interval(0.2), 10.0)  # Z₁ ≥ 1
        @test _existence_fails(interval(0.1), interval(0.1), interval(-0.1), 1.0)   # Z₂ < 0
        @test _existence_fails(interval(0.1), interval(0.1), interval(0.1), -1.0)   # R < 0

        @test_logs (:info,) interval_of_existence(Y, Z₁, Z₂, 5.0; verbose = true)
        @test_logs (:info,) interval_of_existence(interval(2.0), interval(0.5), interval(1.0), 10.0; verbose = true)
        @test_logs interval_of_existence(Y, Z₁, Z₂, 5.0; verbose = false)

        #= With Y far below the ulp of (1 - Z₁)², the formula (-b - √Δ) / Z₂ cancels and the width
           of Z₁ leaks into the root (about 2.6e-12 here). The stable form 2Y / (-b + √Δ) at
           sup(Z₁) gives the true root, about 8.8e-36. =#
        @testset "tiny Y" begin
            Y, Z₁, Z₂ = interval(1e-36), interval(0.886818866398, 0.886818866400), interval(0.7693526887)
            ie, success = interval_of_existence(Y, Z₁, Z₂, Inf)
            @test success
            b = interval(sup(Z₁)) - one(Z₁)
            @test inf(ie) == sup(exact(2) * Y / (-b + sqrt(b^2 - exact(2) * Z₂ * Y)))
            @test inf(ie) < 1e-35
        end

        #= The vertex 1//10 is exact in rational arithmetic and its conversion to Float64 rounds
           up, so the upper end must be walked back down until the contraction is certified =#
        @testset "Rational inputs: the vertex rounds up in Float64" begin
            Y, Z₁, Z₂ = interval(1//100), interval(0//1), interval(10//1)
            z = inf(-(Z₁ - one(Z₁)) / Z₂)
            @test z == 1//10 && float(z) > z

            ie, success = interval_of_existence(Y, Z₁, Z₂, 1.0)
            @test success
            @test ie isa Interval{Float64}
            @test sup(ie) < 0.1
            @test sup(Z₂ * sup(ie)) < inf(one(Z₁) - Z₁)
            @test !(sup(Z₂ * nextfloat(sup(ie))) < inf(one(Z₁) - Z₁))
        end
    end

    @testset "set_of_radii" begin
        W₀ = zeros(Interval{Float64}, 2, 2, 2)

        @testset "decoupled linear problem" begin
            # rₘ = Yₘ / (1 - Zₘₘ)
            Y = [interval(0.01), interval(0.02)]
            Z = [interval(0.1) interval(0.0); interval(0.0) interval(0.2)]
            rmin, _, success = RadiiPolynomial.set_of_radii(Y, Z, W₀, [10.0, 10.0])
            @test success
            @test sup.(rmin) ≈ [0.01 / (1 - 0.1), 0.02 / (1 - 0.2)] atol = 1e-6
            @test all(isguaranteed, rmin)

            @test _radii_fail(Y, Z, W₀, [-1.0, 10.0])
            @test _radii_fail([interval(-0.01), interval(0.02)], Z, W₀, [10.0, 10.0])
        end

        @testset "one dimension" begin
            # P(r) = 0.01 - 0.9 r + 0.1 r² has the smaller root r₁ = (0.9 - √0.806) / 0.2
            Y, Z, W = [interval(0.01)], fill(interval(0.1), 1, 1), fill(interval(0.2), 1, 1, 1)
            rmin, η, success = RadiiPolynomial.set_of_radii(Y, Z, W, [10.0])
            @test success
            @test η == [1.0]
            @test sup(rmin[1]) ≈ 0.011124862507311817 atol = 1e-12
            @test _radii_fail(Y, Z, W, [0.005])   # r₁ exceeds R
        end

        @testset "radii polynomials never nonpositive" begin
            # a diagonal entry Zₘₘ = 2 gives Pₘ(r) = Yₘ + rₘ > 0
            @test _radii_fail([interval(1.0)], fill(interval(2.0), 1, 1), zeros(Interval{Float64}, 1, 1, 1), [100.0]; verbose = true)
            @test _radii_fail(fill(interval(1.0), 2), [interval(2.0) interval(0.0); interval(0.0) interval(2.0)], W₀, [100.0, 100.0]; verbose = true)
            # Z = I makes the Jacobian of the radii polynomials singular at the root
            @test _radii_fail(fill(interval(0.01), 2), [interval(1.0) interval(0.0); interval(0.0) interval(1.0)], W₀, [10.0, 10.0])
        end

        @testset "decoupled quadratic problem" begin
            # each component is the scalar problem Y = 0.75, Z₁ = 0, Z₂ = 0.5 whose root is 1
            Y = [interval(0.75), interval(0.75)]
            Z = zeros(Interval{Float64}, 2, 2)
            W = zeros(Interval{Float64}, 2, 2, 2)
            W[1, 1, 1] = W[2, 2, 2] = interval(0.5)

            rmin, _, success = RadiiPolynomial.set_of_radii(Y, Z, W, [5.0, 5.0])
            @test success
            @test sup.(rmin) ≈ [1.0, 1.0] atol = 1e-6
            @test all(isguaranteed, rmin)
            @test all(r -> decoration(r) == com, rmin)

            @test _radii_fail(Y, Z, W, [0.5, 0.5]; verbose = true)   # the radii exceed R
            @test _radii_fail(Y, Z, W, [5.0]; verbose = true)        # inconsistent sizes

            # the guarantee of the inputs propagates to the radii
            Y_ng = [interval(0.75) + 0, interval(0.75)]
            @test !isguaranteed(Y_ng[1])
            rmin_ng, _, success_ng = RadiiPolynomial.set_of_radii(Y_ng, Z, W, [5.0, 5.0])
            @test success_ng
            @test !any(isguaranteed, rmin_ng)

            rmin_big, _, success_big = RadiiPolynomial.set_of_radii(interval.(big.([0.75, 0.75])), zeros(Interval{BigFloat}, 2, 2), Interval{BigFloat}.(W), [5.0, 5.0])
            @test success_big
            @test eltype(rmin_big) == Interval{BigFloat}
        end

        #= A zero Yₘ makes Pₘ homogeneous in r, so the search must be relative to max(Y) to reach
           the root at any scale. The radii match the root (I - Z)⁻¹ Y of the linear case. =#
        @testset "zero Y entry at a tiny scale" begin
            Z = [0.1 0.2; 0.3 0.1]
            for s ∈ (1e-1, 1e-14, 1e-100), Y ∈ ([s, 0.0], [0.0, s])
                rmin, _, success = RadiiPolynomial.set_of_radii(interval.(Y), interval.(Z), W₀, [1.0, 1.0])
                @test success
                @test sup.(rmin) ≈ (I - Z) \ Y rtol = 1e-12
            end
        end

        #= With Z = 0 and only W[1,1,2] = 8 nonzero,
               P₁(r) = Y₁ + 4 r₁ r₂ - r₁,   P₂(r) = Y₂ - r₂,
           whose positive root for Y = (0.01, 0.2) is r = (0.05, 0.2). There the contraction matrix
           Z + Σⱼ W[:,:,j] rⱼ = [1.6 0; 0 0] has spectral radius 1.6, so P(r) ≤ 0 holds without
           contraction. =#
        @testset "inclusion without contraction" begin
            A = [interval(1.6) interval(0.0); interval(0.0) interval(0.0)]
            η = RadiiPolynomial._perron_vector(A)
            @test η[1] > η[2] > 0
            @test sup(maximum(A * η ./ η)) ≈ 1.6

            W = zeros(Interval{Float64}, 2, 2, 2)
            W[1, 1, 2] = interval(8.0)
            @test _radii_fail([interval(0.01), interval(0.2)], zeros(Interval{Float64}, 2, 2), W, [50.0, 50.0]; verbose = true)
        end
    end
end
