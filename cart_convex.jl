# Conway, B. A., & Larson, K. M. (1998). Collocation versus differential inclusion
# in direct optimization. Journal of Guidance, Control, and Dynamics, 21(5), 780-785.

using JuMP
import ECOS
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

const a = 1.0
const b = -2.694528
const c = -1.155356

const x1i = 0.0
const x2i = 0.0

const ti = 0.0
const tf = 2.0

function cart_convex()
    model = Model(ECOS.Optimizer)
    set_attribute(model, "feastol", eps())
    set_attribute(model, "reltol", eps())
    set_attribute(model, "abstol", eps())

    tau, w, wB, Ba, Bb = psmethod(N-1)

    #
    # Variables
    #

    @variable(model, x1[i=1:N], start=0)
    @variable(model, x2[i=1:N], start=0)
    @variable(model, x1dot[i=1:N], start=0)
    @variable(model, x2dot[i=1:N], start=0)
    @variable(model, x1a, start=0)
    @variable(model, x2a, start=0)
    @variable(model, x1b, start=0)
    @variable(model, x2b, start=0)
    @variable(model, u[i=1:N], start=0)

    #
    # Collected decision variables
    #

    X = hcat( x1, x2 )
    V = hcat( x1dot, x2dot )
    Xa = hcat( x1a, x2a )
    Xb = hcat( x1b, x2b )

    #
    # Endpoint constraints
    #

    fix(x1a, x1i, force=true)
    fix(x2a, x2i, force=true)

    @constraint(model, 0 == a * x1[end] + b * x2[end] - c)

    #
    # Dynamical constraints
    #

    F = hcat(
             x2,
             -x2 + u,
            )

    @constraint(model, dyn1, 0 == X - Ba * V - Xa .* ones(N))
    @constraint(model, dyn2, 0 == (tf - ti) / 2 * F - V)
    @constraint(model, dyn3, 0 == Xa + wB' * V - Xb)

    #
    # Objective
    #

    @objective(model, Min, (tf - ti) / 2.0 * wB' * u.^2)

    #
    # Solve the problem
    #

    solve_and_print_solution(model)

    p1 = Plots.plot(
                    tau,
                    value.(x1),
                    xlabel = "Tau",
                    ylabel = "State",
                    legend = false
                   )

    p2 = Plots.plot(
                    tau,
                    value.(x2),
                    xlabel = "Tau",
                    ylabel = "State",
                    legend = false
                   )

    p3 = Plots.plot(
                    tau,
                    value.(u),
                    xlabel = "Tau",
                    ylabel = "Control",
                    legend = false
                   )

    display(Plots.plot(p1, p2, p3, layout=(2,2), legend=false))
    readline()

    @assert is_solved_and_feasible(model)
end

cart_convex()

