# Human-readable reports of solver runs.

const AnyResult = Union{CGResult,ExactResult}

_iterations(r::CGResult) = r.stats.iterations
_iterations(r::ExactResult) = r.stats.simplex_iterations + r.stats.ipm_iterations
_avg_master_ms(r::CGResult) = 1e3 * mean_master_time(r.stats)
_avg_master_ms(::ExactResult) = NaN
_avg_pricing_ms(r::CGResult) = 1e3 * mean_pricing_time(r.stats)
_avg_pricing_ms(::ExactResult) = NaN
_inner_iterations(r::CGResult) = total_inner_iterations(r.stats)
_inner_iterations(r::ExactResult) = _iterations(r)
_infeasibility(r::CGResult) = r.infeasibility
_infeasibility(::ExactResult) = 0.0
_status(r::CGResult) = r.status
_status(r::ExactResult) = r.status

"""
    compare_results(results...; reference = nothing, io = stdout)

Print a comparison table in the style of thesis Tables 9.2 / 9.3 (all methods, equality /
inequality variant): iterations, total time, average master and pricing time, total inner (simplex / QP / active-set) iterations, objective, and
the gap `objective - reference` to the reference objective (e.g. the exact LP).
"""
function compare_results(results::AnyResult...; reference = nothing, io::IO = stdout)
    ref = reference === nothing ? nothing : objective_value(reference)
    header = @sprintf("%-28s %-12s %7s %10s %12s %12s %10s %16s %11s %9s",
                      "method", "status", "iter", "time [s]", "master [ms]", "pricing [ms]",
                      "inner it.", "objective", "gap", "infeas")
    println(io, header)
    println(io, "─"^length(header))
    for r in results
        # signed gap to the reference objective (e.g. exact LP optimum); NaN without reference
        gap = ref === nothing ? NaN : objective_value(r) - ref
        @printf(io, "%-28s %-12s %7d %10.3f %12.3f %12.3f %10d %16.6f %11.2e %9.1e\n",
                method_name(r), _status(r), _iterations(r), total_time(r), _avg_master_ms(r),
                _avg_pricing_ms(r), _inner_iterations(r), objective_value(r), gap,
                _infeasibility(r))
    end
    return nothing
end

"""
    print_details(result; io = stdout)

Print the full statistics of a column generation run, including solver-specific details
(e.g. QPAS working-set updates and time breakdown).
"""
function print_details(r::CGResult; io::IO = stdout)
    show(io, MIME"text/plain"(), r)
    println(io)
    s = r.stats
    @printf(io, "  setup time          : %.3f s\n", s.setup_time)
    @printf(io, "  inner iterations    : %d\n", total_inner_iterations(s))
    ff = first_feasible_iteration(s)
    @printf(io, "  first feasible iter : %s\n", ff === nothing ? "–" : string(ff))
    @printf(io, "  center updates      : %d\n", center_updates(s))
    s.duplicate_columns > 0 && @printf(io, "  duplicate columns   : %d\n", s.duplicate_columns)
    if !isempty(s.details)
        println(io, "  solver details:")
        # align the `name : value` pairs on the longest name
        width = maximum(length ∘ first, s.details)
        for (name, value) in s.details
            println(io, "    ", rpad(name, width), " : ", value)
        end
    end
    return nothing
end

print_details(r::ExactResult; io::IO = stdout) = (show(io, MIME"text/plain"(), r); println(io))
