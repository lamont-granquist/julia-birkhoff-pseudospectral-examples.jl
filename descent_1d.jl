using JuMP
import Ipopt
import Plots
using LinearAlgebra
using Printf
using Glob

foreach(include, glob("*.jl", "lib"))

# number of grid points
const N = 150

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
    @variable(model, 0 <= T[i=1:N] <= Tmax, start=0)
    @variable(model, ti, start=0)
    @variable(model, tf, start=1)

    #
    # Collected decision variables
    #

    X = hcat( h, v, m )
    V = hcat( hdot, vdot, mdot )
    Xi = hcat(h[1], v[1], m[1])
    Xf = hcat(h[N], v[N], m[N])

    #
    # Endpoint constraints
    #

    fix(h[1], hi, force=true)
    fix(v[1], vi, force=true)
    fix(m[1], mi, force=true)
    fix(h[N], hf, force=true)
    fix(v[N], vf, force=true)
    fix(ti, 0, force=true)

    #
    # Dynamical constraints
    #

    F = hcat(
             v,
             -1 .+ T ./ m,
             -T ./ 2.349,
            )

    @constraint(model, dyn1, X == Xi .* ones(N) + Ba * V)
    @constraint(model, dyn2, V == (tf - ti) ./ 2 * F)
    @constraint(model, dyn3, Xf == Xi + wB' * V)

    #
    # Objective
    #

    @objective(model, Max, wB' * T ./ 2.349)

    #
    # Solve the problem
    #

    solve_and_print_solution(model)

    λ1 = value.(dual(dyn1))
    λ2 = value.(dual(dyn2))
    λ3 = value.(dual(dyn3))


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
                    ylabel = "Control",
                    legend = false,
                   )

    display(Plots.plot(p1, p2, p3, layout=(2,2), legend=false))

    p4 = Plots.plot(
                    tau,
                    value.(λ1),
                    xlabel = "Tau",
                    ylabel = "Costate",
                    legend = false,
                   )
    p5 = Plots.plot(
                    tau,
                    value.(λ2),
                    xlabel = "Tau",
                    ylabel = "Costate",
                    legend = false,
                   )

    display(Plots.plot(p4, p5, layout=(2,1), legend=false))
    readline()

    @assert is_solved_and_feasible(model)
end

descent_1d()

