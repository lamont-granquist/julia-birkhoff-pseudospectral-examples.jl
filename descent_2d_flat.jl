#
# Reynolds, T. P., & Mesbahi, M. (2020). Optimal planar powered descent with independent
# thrust and torque. Journal of Guidance, Control, and Dynamics, 43(7), 1225-1231.
#

using JuMP
import Ipopt
import Plots
using LinearAlgebra
using Printf
using Glob

foreach(include, glob("*.jl", "lib"))

# number of grid points
const N = 25

#
# Problem constants
#

const ri = [4.5 16.5]'
const vi = [-10.0 -1.5]'
const mi = 2
const θi = 0
const ωi = 0

const rf = [0 0]'
const vf = [0 0]'
const ωf = 0
const θf = 0

const g = -1
const J = 0.25
const Γmin = 1.5
const Γmax = 6.5
const τmax = 1.0
const α = 0.0034 # reciprocal of exhaust velocity

rinit = hcat(
             range(ri[1], rf[1], N),
             range(ri[2], rf[2], N),
            )
vinit = hcat(
             range(vi[1], vf[1], N),
             range(vi[2], vf[2], N),
            )

function descent_2d_flat()
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level", 5)
    set_optimizer_attribute(model, "print_user_options", "yes")
    set_optimizer_attribute(model, "max_iter", 2000)
    set_optimizer_attribute(model, "tol", 1e-12)

    tau, w, wB, Ba, Bb = psmethod(N-1)

    #
    # Variables
    #

    @variable(model, r[i=1:N,j=1:2], start=rinit[i,j])
    @variable(model, v[i=1:N,j=1:2], start=vinit[i,j])
    @variable(model, m[i=1:N], start=mi)
    @variable(model, θ[i=1:N], start=θi)
    @variable(model, ω[i=1:N], start=ωi)
    @variable(model, Γmin <= Γ[i=1:N] <= Γmax , start=Γmin)
    @variable(model, -τmax <= τ[i=1:N] <= τmax, start=0)

    @variable(model, rdot[i=1:N,j=1:2], start=0)
    @variable(model, vdot[i=1:N,j=1:2], start=0)
    @variable(model, mdot[i=1:N], start=0)
    @variable(model, θdot[i=1:N], start=0)
    @variable(model, ωdot[i=1:N], start=0)
    @variable(model, Γdot[i=1:N], start=0)
    @variable(model, τdot[i=1:N], start=0)

    @variable(model, ra[j=1:2], start=ri[j])
    @variable(model, va[j=1:2], start=vi[j])
    @variable(model, ma, start=mi)
    @variable(model, θa, start=θi)
    @variable(model, ωa, start=ωi)

    @variable(model, rb[j=1:2], start=rf[j])
    @variable(model, vb[j=1:2], start=vf[j])
    @variable(model, mb, start=mi)
    @variable(model, θb, start=θf)
    @variable(model, ωb, start=ωf)

    @variable(model, ti, start=0)
    @variable(model, tf, start=10)

    #
    # Collected decision variables
    #

    X = hcat(r, v, m, θ, ω)
    V = hcat(rdot, vdot, mdot, θdot, ωdot)
    Xa = hcat(ra', va', ma, θa, ωa)
    Xb = hcat(rb', vb', mb, θb, ωb)

    #
    # Endpoint constraints
    #

    fix.(ra, ri, force=true)
    fix.(va, vi, force=true)
    fix(ma, mi, force=true)
    #fix(θa, θi, force=true)
    fix(ωa, ωi, force=true)

    fix.(rb, rf, force=true)
    fix.(vb, vf, force=true)
    fix(θb, θf, force=true)
    fix(ωb, ωf, force=true)

    fix(ti, 0, force=true)

    #
    # Dynamical constraints
    #

    @expression(model, sinθ[i=1:N], sin(θ[i]))
    @expression(model, cosθ[i=1:N], cos(θ[i]))
    d = hcat(-sinθ, cosθ)

    F = hcat(
             v,
             Γ ./ m .* d .+ [0 g],
             -α * Γ,
             ω,
             τ ./ J,
            )

    @constraint(model, dyn1, 0 == X - Ba * V - Xa .* ones(N))
    @constraint(model, dyn2, 0 == (tf - ti) / 2 * F - V)
    @constraint(model, dyn3, 0 == Xa + wB' * V - Xb)

    #
    # Objective
    #

    @objective(model, Min, ((tf - ti) / 2 * wB)' * Γ)

    #
    # Solve the problem
    #

    solve_and_print_solution(model)

    Ω = value.(dual(dyn1)) ./ wB
    Λ = value.(dual(dyn2)) ./ wB
    λb = value.(dual(dyn3))
    λa = λb - wB' * Ω

    #display(Λ)
    #display(Ω)
    #display(λa)
    #display(λb)

    p1 = Plots.plot(
                    tau,
                    value.(Γ),
                    xlabel = "Tau",
                    ylabel = "Thrust",
                    legend = false,
                   )

    p2 = Plots.plot(
                    tau,
                    value.(τ),
                    xlabel = "Tau",
                    ylabel = "Torque",
                    legend = false,
                   )

    p3 = Plots.plot(
                    tau,
                    value.(θ),
                    xlabel = "Tau",
                    ylabel = "Theta",
                    legend = false,
                   )

    p4 = Plots.plot(
                    tau,
                    value.(m),
                    xlabel = "Tau",
                    ylabel = "Mass",
                    legend = false,
                   )

    p5 = Plots.plot(
                    tau,
                    value.(ω),
                    xlabel = "Tau",
                    ylabel = "Omega",
                    legend = false,
                   )

    p5 = Plots.plot(
                    tau,
                    value.(ω),
                    xlabel = "Tau",
                    ylabel = "Omega",
                    legend = false,
                   )

    r = value.(r)
    x = r[:,1]
    y = r[:,2]

    p6 = Plots.plot(
                    x,
                    y,
                    label = "Position",
                    markersize=8,
                   )

    display(Plots.plot(p1, p2, p3, p4, p5, p6, layout=(2,3), legend=false))
    readline()

    @assert is_solved_and_feasible(model)
end

descent_2d_flat()
