using JuMP
import Ipopt
import Plots
using LinearAlgebra
using Printf
using Glob

foreach(include, glob("*.jl", "lib"))

# number of grid points
const N = 20

#
# Problem constants
#

const hi = 1
const vi = -0.783
const mi = 1

const hf = 0
const vf = 0

const Tmax = 1.227

function descent_1d()
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level", 5)
    set_optimizer_attribute(model, "print_user_options", "yes")
    set_optimizer_attribute(model, "max_iter", 2000)
    set_optimizer_attribute(model, "tol", 1e-12)

    tau, w, wB, Ba, Bb = psmethod(N-1)

    #
    # Variables
    #

    @variable(model, h[i=1:N], start=hi)
    @variable(model, v[i=1:N], start=vi)
    @variable(model, m[i=1:N], start=mi)
    @variable(model, hdot[i=1:N], start=0)
    @variable(model, vdot[i=1:N], start=0)
    @variable(model, mdot[i=1:N], start=0)
    @variable(model, ha, start=hi)
    @variable(model, va, start=vi)
    @variable(model, ma, start=mi)
    @variable(model, hb, start=0)
    @variable(model, vb, start=0)
    @variable(model, mb, start=0)
    @variable(model, 0 <= T[i=1:N] <= Tmax, start=0)
    @variable(model, ti, start=0)
    @variable(model, tf, start=1)

    #
    # Collected decision variables
    #

    X = hcat( h, v, m )
    V = hcat( hdot, vdot, mdot )
    Xi = hcat(ha, va, ma)
    Xf = hcat(hb, vb, mb)

    #
    # Endpoint constraints
    #

    fix(ha, hi, force=true)
    fix(va, vi, force=true)
    fix(ma, mi, force=true)
    fix(hb, hf, force=true)
    fix(vb, vf, force=true)
    fix(ti, 0, force=true)

    #
    # Dynamical constraints
    #

    F = hcat(
             v,
             -1 .+ T ./  m,
             -T ./ 2.349,
            )

    @constraint(model, dyn1, 0 == X - Ba * V - Xi .* ones(N))
    @constraint(model, dyn2, 0 == (tf - ti) / 2 * F - V)
    @constraint(model, dyn3, 0 == Xi + wB' * V - Xf)

    #
    # Objective
    #

    @objective(model, Min, ((tf - ti) / 2 * wB)' * (T ./ 2.349))
    #@objective(model, Max, m[end])

    #
    # Solve the problem
    #

    solve_and_print_solution(model)

    Ω = value.(dual(dyn1)) ./ wB
    Λ = value.(dual(dyn2)) ./ wB
    λb = value.(dual(dyn3))
    λa = λb - wB' * Ω

    display(Λ)
    display(Ω)
    display(λa)
    display(λb)

    p1 = Plots.plot(
                    tau,
                    value.(h),
                    xlabel = "Tau",
                    ylabel = "Height",
                    legend = false,
                   )

    p2 = Plots.plot(
                    tau,
                    value.(v),
                    xlabel = "Tau",
                    ylabel = "Velocity",
                    legend = false,
                   )

    p3 = Plots.plot(
                    tau,
                    value.(T),
                    xlabel = "Tau",
                    ylabel = "Thrust",
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
                    value.(Λ),
                    xlabel = "Tau",
                    ylabel = "Λ",
                    legend = false,
                    #ylim=(-0.1, 0.1),
                   )
    p6 = Plots.plot(
                    tau,
                    value.(Ω),
                    xlabel = "Tau",
                    ylabel = "Ω",
                    legend = false,
                   )

    display(Plots.plot(p1, p2, p3, p4, p5, p6, layout=(2,3), legend=false))
    readline()

    @assert is_solved_and_feasible(model)
end

descent_1d()
