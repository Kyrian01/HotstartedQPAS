# Derives the plotting tables (data/*.csv) and LaTeX table snippets (data/tab_*.tex) of the
# results chapter from the raw experiment files in data/raw (see generate_data.jl).
#
#   julia thesis/scripts/make_plot_data.jl
#
# Uses only Base Julia (no packages).

using Printf, Statistics

# HQ_PLOT_DATA=<dir> reads <dir>/raw and writes into <dir> instead (for testing the script)
const DATA = get(ENV, "HQ_PLOT_DATA", joinpath(@__DIR__, "..", "data"))
const RAW = joinpath(DATA, "raw")

# ── tiny CSV reader (handles "quoted, fields") ──────────────────────────────────────────
function splitcsv(line)
    out = String[]
    buf = IOBuffer()
    inq = false
    for ch in line
        if ch == '"'
            inq = !inq
        elseif ch == ',' && !inq
            push!(out, String(take!(buf)))
        else
            print(buf, ch)
        end
    end
    push!(out, String(take!(buf)))
    return out
end

function readcsv(path)
    lines = filter(!isempty, readlines(path))
    header = splitcsv(lines[1])
    rows = [Dict(zip(header, splitcsv(l))) for l in lines[2:end]]
    return header, rows
end

num(x) = x in ("", "nan", "NaN", "missing") ? NaN : parse(Float64, x)
col(rows, k) = [num(get(r, k, "")) for r in rows]

function writecsv(path, header, columns...)
    open(path, "w") do io
        println(io, join(header, ","))
        for i in eachindex(columns[1])
            println(io, join((fmt(c[i]) for c in columns), ","))
        end
    end
end
fmt(x::AbstractFloat) = isnan(x) ? "nan" : @sprintf("%.6g", x)
fmt(x) = string(x)

# indices for thinning a long trace to at most `k` points (always keeps the last one)
thin(n, k) = n <= k ? collect(1:n) : unique(vcat(round.(Int, range(1, n; length = k)), n))

moving_mean(v, w) = [mean(v[max(1, i - w + 1):i]) for i in eachindex(v)]

tex_num(x; digits = 2) = isnan(x) ? "--" : @sprintf("%.*f", digits, x)
function tex_sci(x)
    (isnan(x) || x == 0) && return x == 0 ? "\$0\$" : "--"
    e = floor(Int, log10(abs(x)))
    m = round(x / 10.0^e; digits = 1)
    abs(m) >= 10 && (m /= 10; e += 1)        # 9.96 rounds to 10.0: write 1.0·10^(e+1)
    return @sprintf("\$%.1f\\cdot 10^{%d}\$", m, e)
end
tex_int(x) = begin
    isnan(x) && return "--"
    s = string(round(Int, x))
    # thin space as thousands separator
    parts = String[]
    while length(s) > 3
        pushfirst!(parts, s[end-2:end]); s = s[1:end-3]
    end
    pushfirst!(parts, s)
    join(parts, "\\,")
end
# time in seconds with a sensible number of digits
tex_time(x) = isnan(x) ? "--" : x < 1 ? @sprintf("%.3f", x) : x < 100 ? @sprintf("%.2f", x) : @sprintf("%.0f", x)

const SOLVER_NAME = Dict("exact" => "Exact LP (HiGHS)", "lp" => "LP CG (HiGHS)",
                         "explicit_qp" => "Explicit QP CG (HiGHS)",
                         "eliminated_qp" => "Eliminated QP CG (HiGHS)",
                         "qpas" => "QPAS CG, sparse factor",
                         "qpas_dense" => "QPAS CG, dense factor")
const MAIN_INSTANCES = ["I1", "I2", "I3", "I4"]
const ALL_INSTANCES = ["I1", "I2", "I3", "I4", "S5", "S6"]
tagof(name, frac) = "$(name)_$(round(Int, 100frac))"

# ── 1. spy plot of D_W ──────────────────────────────────────────────────────────────────
let p = joinpath(RAW, "spy_DW.csv")
    if isfile(p)
        _, rows = readcsv(p)
        writecsv(joinpath(DATA, "spy_DW_orig.csv"), ["row", "col"],
                 [r["row"] for r in rows], [r["col"] for r in rows])
        writecsv(joinpath(DATA, "spy_DW_perm.csv"), ["row", "col"],
                 [r["row"] for r in rows], [r["col_perm"] for r in rows])
        println("spy data")
    end
end

# ── 2. instances table (I1–I4 from the comparison, S5/S6 from the scaling runs) ─────────
_, inst = readcsv(joinpath(RAW, "instances.csv"))
_, scal = isfile(joinpath(RAW, "scaling.csv")) ? readcsv(joinpath(RAW, "scaling.csv")) :
          (String[], Dict{String,String}[])
for name in unique(r["instance"] for r in scal), frac in (0.0, 0.3)
    rs = [r for r in scal if r["instance"] == name && num(r["frac"]) == frac]
    isempty(rs) && continue
    r = rs[1]
    push!(inst, Dict("instance" => name, "frac" => r["frac"], "V" => r["V"], "E" => r["E"],
                     "m" => r["m"], "m_ineq" => r["m_ineq"], "pilots" => r["pilots"],
                     "N" => string(round(Int, num(r["E"]) * num(r["pilots"])))))
end
instrow(name, frac) = first(r for r in inst if r["instance"] == name && num(r["frac"]) == frac)
open(joinpath(DATA, "tab_instances.tex"), "w") do io
    for name in ALL_INSTANCES
        count(r -> r["instance"] == name, inst) == 2 || continue
        r0, r3 = instrow(name, 0.0), instrow(name, 0.3)
        println(io, join([name, r0["V"], tex_int(num(r0["E"])), r0["pilots"], r0["m"],
                          r3["m_ineq"], tex_int(num(r0["N"]))], " & "), " \\\\")
    end
end

# ── 3. comparison tables (I1–I4) ────────────────────────────────────────────────────────
_, comp = readcsv(joinpath(RAW, "comparison.csv"))
crow(name, frac, solver) = begin
    rs = [r for r in comp if r["instance"] == name && num(r["frac"]) == frac && r["solver"] == solver]
    isempty(rs) ? nothing : rs[1]
end
function comparison_table(frac)
    io = IOBuffer()
    names = [n for n in MAIN_INSTANCES if any(r -> r["instance"] == n, comp)]
    for (ni, name) in enumerate(names)
        rows = [r for r in comp if r["instance"] == name && num(r["frac"]) == frac]
        isempty(rows) && continue
        order = ["exact", "lp", "explicit_qp", "eliminated_qp", "qpas", "qpas_dense"]
        sort!(rows; by = r -> findfirst(==(r["solver"]), order))
        zstar = num(rows[1]["exact_objective"])
        for (k, r) in enumerate(rows)
            solver = r["solver"]
            first_col = k == 1 ? "\\multirow{$(length(rows))}{*}{$name}" : ""
            if solver == "exact"
                println(io, join([first_col, SOLVER_NAME[solver], r["status"] == "optimal" ? "opt." : r["status"],
                                  "--", "--", tex_num(num(r["total_s"])), "--",
                                  "\$0\$", "\$0\$", "\$0\$"], " & "), " \\\\")
            else
                st = r["status"] == "optimal" ? "opt." : (r["status"] == "time_limit" ? "time" : r["status"])
                relerr = (num(r["objective"]) - zstar) / abs(zstar)
                bgap = (zstar - num(r["bound"])) / abs(zstar)
                inf = num(r["infeasibility"])
                ms = 1000 * num(r["master_s"]) / num(r["iterations"])
                println(io, join([first_col, SOLVER_NAME[solver], st, tex_int(num(r["iterations"])),
                                  tex_int(num(r["columns"])), tex_num(num(r["total_s"])),
                                  tex_num(ms; digits = 3),
                                  tex_sci(relerr), isinf(num(r["bound"])) ? "--" : tex_sci(bgap),
                                  tex_sci(inf)], " & "), " \\\\")
            end
        end
        ni < length(names) && println(io, "\\midrule")
    end
    return String(take!(io))
end
write(joinpath(DATA, "tab_comparison_eq.tex"), comparison_table(0.0))
write(joinpath(DATA, "tab_comparison_ineq.tex"), comparison_table(0.3))

# ── 4. details of the active-set method: counters, time breakdown, operations ───────────
_, det = readcsv(joinpath(RAW, "qpas_details.csv"))
getdet(name, frac, solver, key) = begin
    r = [d for d in det if d["instance"] == name && num(d["frac"]) == frac &&
                           d["solver"] == solver && d["key"] == key]
    isempty(r) ? "" : r[1]["value"]
end
# "a (b, c)" -> (a, b, c); "a / b" -> (a, b)
parse_op(s) = (m = match(r"([\d.eE+-]+) \((\d+), ([\d.eE+-]+)\)", s);
               m === nothing ? (NaN, NaN, NaN) : (num(m[1]), num(m[2]), num(m[3])))
parse_pair(s) = (p = split(s, " / "); length(p) == 2 ? (num(strip(p[1])), num(strip(p[2]))) : (NaN, NaN))

open(joinpath(DATA, "tab_qpas_counters.tex"), "w") do io
    for name in ALL_INSTANCES, frac in (0.0, 0.3)
        g(k) = getdet(name, frac, "qpas_dense", k)
        g("active-set iterations") == "" && continue
        steps = split(g("primal steps (full)"), " ")
        full = strip(steps[2], ['(', ')'])
        slack = g("slack expansions / reductions")
        slack = slack == "" ? "--" : join((tex_int(num(strip(s))) for s in split(slack, "/")), " / ")
        nfree = g("final free slacks |N_s|")
        refs = getdet(name, frac, "qpas", "refactorizations")
        println(io, join([name, frac == 0 ? "--" : "30\\%", tex_int(num(g("active-set iterations"))),
                          tex_int(num(steps[1])) * " (" * tex_int(num(full)) * ")",
                          tex_int(num(g("working-set expansions"))),
                          tex_int(num(g("of which representative swaps"))),
                          tex_int(num(g("working-set reductions"))), slack,
                          g("singular-Hessian steps"),
                          tex_int(num(g("factor updates (row add/del, rank-one)"))),
                          (refs == "" ? "--" : tex_int(num(refs))) * " / " * g("refactorizations"),
                          g("final |N_x| (dim. of H_eff)"),
                          nfree == "" ? "--" : nfree], " & "), " \\\\")
    end
end

# time breakdown: both factorizations, as shares of the total time of the sparse run
open(joinpath(DATA, "breakdown.csv"), "w") do io
    println(io, "label,direction,multipliers,step,updates,refactor,pricing")
    for name in ALL_INSTANCES[2:end], frac in (0.0, 0.3)
        getdet(name, frac, "qpas", "time direction [s]") == "" && continue
        parts = Dict{String,Vector{Float64}}()
        for solver in ("qpas", "qpas_dense")
            g(k) = num(getdet(name, frac, solver, k))
            dir, mul, stp = g("time direction [s]"), g("time multipliers [s]"), g("time step [s]")
            exp_, red = g("time expand [s]"), g("time reduce [s]")
            ref = g("time refactor [s] (within expand/reduce)")
            r = crow(name, frac, solver)
            pr = r === nothing ? NaN : num(r["pricing_s"])
            parts[solver] = [dir, mul, stp, exp_ + red - ref, ref, pr]
        end
        tot = sum(parts["qpas"])
        for (solver, lab) in (("qpas", "sparse"), ("qpas_dense", "dense"))
            label = "$(name) $(frac == 0 ? "eq." : "ineq.") $(lab)"
            println(io, join(vcat(label, [@sprintf("%.2f", x) for x in parts[solver] ./ tot .* 100]), ","))
        end
    end
end

# comparison of the two factorizations on every instance
open(joinpath(DATA, "tab_backends.tex"), "w") do io
    for name in ALL_INSTANCES, frac in (0.0, 0.3)
        rs, rd = crow(name, frac, "qpas"), crow(name, frac, "qpas_dense")
        (rs === nothing || rd === nothing) && continue
        ms, md = num(rs["master_s"]), num(rd["master_s"])
        ts, td = num(rs["total_s"]), num(rd["total_s"])
        os, od = num(rs["objective"]), num(rd["objective"])
        mems = parse_pair(getdet(name, frac, "qpas", "factor memory now / peak [MB]"))[2]
        memd = parse_pair(getdet(name, frac, "qpas_dense", "factor memory now / peak [MB]"))[2]
        m = num(instrow(name, frac)["m"])
        println(io, join([name, frac == 0 ? "--" : "30\\%", tex_int(m),
                          tex_int(num(rs["iterations"])) * " / " * tex_int(num(rd["iterations"])),
                          tex_time(ms), tex_time(md), @sprintf("%.1f", ms / md),
                          tex_time(ts), tex_time(td),
                          tex_int(num(getdet(name, frac, "qpas", "refactorizations"))) * " / " *
                              getdet(name, frac, "qpas_dense", "refactorizations"),
                          tex_num(mems; digits = 1), tex_num(memd; digits = 1),
                          tex_sci(abs(os - od) / abs(os))], " & "), " \\\\")
    end
end

# time per factor operation (µs per call) on selected instances
open(joinpath(DATA, "tab_ops.tex"), "w") do io
    for name in ("I2", "I3", "I4", "S5", "S6")
        g(solver, op) = parse_op(getdet(name, 0.0, solver, "factor op $(op) [s] (calls, µs/call)"))
        getdet(name, 0.0, "qpas", "time direction [s]") == "" && continue
        cells = String[name]
        for op in ("solve", "rowadd", "rowdel", "assemble")
            s, d = g("qpas", op)[3], g("qpas_dense", op)[3]
            push!(cells, tex_num(s; digits = 1), tex_num(d; digits = 1))
        end
        println(io, join(cells, " & "), " \\\\")
    end
end

# ── 5. convergence traces (I2, I3) ──────────────────────────────────────────────────────
for (tag, frac) in (("I2_0", 0.0), ("I2_30", 0.3), ("I3_0", 0.0), ("I3_30", 0.3))
    name = split(tag, "_")[1]
    crow(name, frac, "exact") === nothing && continue
    zstar = num(crow(name, frac, "exact")["exact_objective"])
    for solver in ("lp", "explicit_qp", "eliminated_qp", "qpas", "qpas_dense")
        p = joinpath(RAW, "trace_$(tag)_$(solver).csv")
        isfile(p) || continue
        _, tr = readcsv(p)
        it = col(tr, "iteration")
        mobj = col(tr, "master_objective")
        cost = col(tr, "operational_cost")
        lb = col(tr, "lower_bound")
        inf = col(tr, "infeasibility")
        dpi = col(tr, "dual_change")
        mt = col(tr, "master_time") .* 1000
        added = col(tr, "columns_added")
        ncol = col(tr, "n_columns")
        gap = abs.(mobj .- zstar) ./ abs(zstar)
        cgap = abs.(cost .- zstar) ./ abs(zstar)
        bgap = [isnan(x) ? NaN : max((zstar - x) / abs(zstar), 1e-16) for x in lb]
        idx = thin(length(it), 1500)
        writecsv(joinpath(DATA, "conv_$(tag)_$(solver).csv"),
                 ["it", "gap", "costgap", "infeas", "dpi", "mt", "mtavg", "ncol"],
                 it[idx], max.(gap[idx], 1e-16), max.(cgap[idx], 1e-16), max.(inf[idx], 1e-16),
                 dpi[idx], mt[idx], moving_mean(mt, 50)[idx], ncol[idx])
        # bound points (only where all pilots were priced), thinned
        bi = findall(!isnan, bgap)
        bi = bi[thin(length(bi), 600)]
        writecsv(joinpath(DATA, "bound_$(tag)_$(solver).csv"), ["it", "bgap"], it[bi], bgap[bi])
        if solver == "qpas_dense"
            # phases: per bin of 100 iterations, share of iterations that added a column
            B = 100
            nb = cld(length(it), B)
            mid = [(b - 1) * B + B / 2 for b in 1:nb]
            share = [mean(added[((b - 1) * B + 1):min(b * B, length(it))] .> 0) for b in 1:nb]
            writecsv(joinpath(DATA, "phases_$(tag).csv"), ["it", "share"], mid, share)
        end
    end
end

# ── 6. center-only statistics table (dense factor, the default) ─────────────────────────
open(joinpath(DATA, "tab_phases.tex"), "w") do io
    for name in MAIN_INSTANCES, frac in (0.0, 0.3)
        p = joinpath(RAW, "trace_$(tagof(name, frac))_qpas_dense.csv")
        isfile(p) || continue
        _, tr = readcsv(p)
        added = col(tr, "columns_added") .> 0
        inner = col(tr, "inner_iterations")
        mt = col(tr, "master_time")
        nadd, nco = count(added), count(.!added)
        firstco = findfirst(.!added)
        lastadd = findlast(added)
        println(io, join([name, frac == 0 ? "--" : "30\\%", tex_int(length(added)), tex_int(nadd),
                          tex_int(nco), firstco === nothing ? "--" : tex_int(firstco),
                          tex_int(lastadd), tex_num(mean(inner[added]); digits = 1),
                          nco == 0 ? "--" : tex_num(mean(inner[.!added]); digits = 1),
                          tex_num(100 * sum(mt[.!added]) / sum(mt); digits = 0) * "\\%"],
                     " & "), " \\\\")
    end
end

# ── 7. diagnostics: sets, fill, residual ────────────────────────────────────────────────
for name in ALL_INSTANCES, frac in (0.0, 0.3)
    tag = tagof(name, frac)
    p = joinpath(RAW, "diag_$(tag)_qpas_dense.csv")
    isfile(p) || continue
    _, d = readcsv(p)
    m = num(instrow(name, frac)["m"])
    it = col(d, "iteration")
    Ns = [isnan(x) ? 0.0 : x for x in col(d, "N_s")]
    writecsv(joinpath(DATA, "sets_$(tag).csv"), ["it", "N", "Ns", "bound", "m"],
             it, col(d, "N"), Ns, m .- Ns, fill(m, length(it)))
end
# factor residual ‖LDLᵀ − H_eff‖ over a run, both factorizations
for name in MAIN_INSTANCES, frac in (0.0, 0.3)
    tag = tagof(name, frac)
    ps, pd = joinpath(RAW, "diag_$(tag)_qpas.csv"), joinpath(RAW, "diag_$(tag)_qpas_dense.csv")
    (isfile(ps) && isfile(pd)) || continue
    for (p, suffix) in ((ps, "sparse"), (pd, "dense"))
        _, d = readcsv(p)
        haskey(d[1], "factor_residual") || continue
        writecsv(joinpath(DATA, "resid_$(tag)_$(suffix).csv"), ["it", "residual", "N"],
                 col(d, "iteration"), max.(col(d, "factor_residual"), 1e-18), col(d, "N"))
    end
end
const FILL_RUNS = [("tau0", "never"), ("tau25", "25"), ("tau150", "150"), ("tau600", "600"),
                   ("dense_tau0", "dense")]
for name in ("I2", "I3"), (suffix, _) in FILL_RUNS
    p = joinpath(RAW, "diag_$(name)_$(suffix).csv")
    isfile(p) || continue
    _, d = readcsv(p)
    writecsv(joinpath(DATA, "fill_$(name)_$(suffix).csv"),
             ["it", "stored", "effective", "dense", "reach", "N", "MB"],
             col(d, "iteration"), col(d, "factor_stored"), col(d, "factor_effective"),
             col(d, "factor_dense"), col(d, "reach") ./ max.(col(d, "N"), 1), col(d, "N"),
             col(d, "factor_MB"))
end
let p = joinpath(RAW, "refactor.csv")
    if isfile(p)
        _, rf = readcsv(p)
        for name in ("I2", "I3")
            rows = [r for r in rf if r["instance"] == name && r["backend"] == "cholmod"]
            finite = sort([r for r in rows if num(r["tau"]) > 0]; by = r -> num(r["tau"]))
            never = [r for r in rows if num(r["tau"]) == 0]
            # finite intervals as a line; "never" (tau = 0) as a separate point at x = 2000,
            # separated by a NaN row so that pgfplots (unbounded coords=jump) breaks the line
            xs = vcat([num(r["tau"]) for r in finite], 1000.0, [2000.0 for _ in never])
            ts = vcat(col(finite, "total_s"), NaN, col(never, "total_s"))
            writecsv(joinpath(DATA, "refactor_$(name).csv"), ["tau", "time"], xs, ts)
            # the dense factor without periodic refactorization, as a horizontal reference line
            dn = [r for r in rf if r["instance"] == name && r["backend"] == "dense" && num(r["tau"]) == 0]
            isempty(dn) || writecsv(joinpath(DATA, "refactor_$(name)_dense.csv"), ["tau", "time"],
                                    [20.0, 2500.0], fill(num(dn[1]["total_s"]), 2))
        end
        open(joinpath(DATA, "tab_refactor.tex"), "w") do io
            for name in ("I2", "I3")
                rows = [r for r in rf if r["instance"] == name]
                for (k, r) in enumerate(rows)
                    tau = round(Int, num(r["tau"]))
                    dense = r["backend"] == "dense"
                    dp = joinpath(RAW, "diag_$(name)_$(dense ? "dense_" : "")tau$(tau).csv")
                    _, d = readcsv(dp)
                    st, ef = col(d, "factor_stored"), col(d, "factor_effective")
                    reach = col(d, "reach") ./ max.(col(d, "N"), 1)
                    peak = maximum(col(d, "factor_peak_MB"))
                    waste = mean([s == 0 ? 0.0 : (s - e) / s for (s, e) in zip(st, ef)])
                    first_col = k == 1 ? "\\multirow{$(length(rows))}{*}{$name}" : ""
                    println(io, join([first_col, dense ? "dense" : "sparse",
                                      tau == 0 ? "never" : string(tau),
                                      tex_num(num(r["total_s"])), tex_int(num(r["refactorizations"])),
                                      tex_int(maximum(st)), tex_int(maximum(st .- ef)),
                                      tex_num(100waste; digits = 0) * "\\%",
                                      tex_num(mean(reach); digits = 2), tex_num(peak; digits = 1)],
                                 " & "), " \\\\")
                end
                name == "I2" && println(io, "\\midrule")
            end
        end
    end
end

# ── 8. scaling ──────────────────────────────────────────────────────────────────────────
let
    pts = Dict{Tuple{String,Float64},NTuple{4,Float64}}()   # (m, sparse, dense, exact)
    for name in MAIN_INSTANCES, frac in (0.0, 0.3)
        rs, rd, re = crow(name, frac, "qpas"), crow(name, frac, "qpas_dense"), crow(name, frac, "exact")
        (rs === nothing || rd === nothing || re === nothing) && continue
        pts[(name, frac)] = (num(instrow(name, frac)["m"]), num(rs["total_s"]), num(rd["total_s"]),
                             num(re["total_s"]))
    end
    for name in unique(r["instance"] for r in scal), frac in (0.0, 0.3)
        rs = [r for r in scal if r["instance"] == name && num(r["frac"]) == frac]
        length(rs) == 2 || continue
        get_(solver) = first(r for r in rs if r["solver"] == solver)
        s, d = get_("qpas"), get_("qpas_dense")
        ex = s["exact_status"] == "optimal" ? num(s["exact_s"]) : NaN
        pts[(name, frac)] = (num(s["m"]), num(s["total_s"]), num(d["total_s"]), ex)
    end
    for (frac, suffix) in ((0.0, "eq"), (0.3, "ineq"))
        keys_ = [(n, frac) for n in ALL_INSTANCES if haskey(pts, (n, frac))]
        writecsv(joinpath(DATA, "scaling_$(suffix).csv"), ["name", "m", "qpas", "dense", "exact"],
                 first.(keys_), [pts[k][1] for k in keys_], [pts[k][2] for k in keys_],
                 [pts[k][3] for k in keys_], [pts[k][4] for k in keys_])
    end
    # the large instances in detail
    open(joinpath(DATA, "tab_scaling.tex"), "w") do io
        for name in unique(r["instance"] for r in scal), frac in (0.0, 0.3)
            rs = [r for r in scal if r["instance"] == name && num(r["frac"]) == frac]
            isempty(rs) && continue
            ex_ok = rs[1]["exact_status"] == "optimal"
            zstar = num(rs[1]["exact_objective"])
            for (k, r) in enumerate(sort(rs; by = r -> r["solver"]))
                first_cols = k == 1 ? ["\\multirow{2}{*}{$name}", frac == 0 ? "\\multirow{2}{*}{--}" : "\\multirow{2}{*}{30\\%}",
                                       "\\multirow{2}{*}{" * (ex_ok ? tex_time(num(r["exact_s"])) : "\$>" * string(round(Int, num(r["exact_s"]))) * "\$") * "}"] :
                                      ["", "", ""]
                bgap = ex_ok ? (zstar - num(r["bound"])) / abs(zstar) : NaN
                println(io, join(vcat(first_cols,
                                      [r["solver"] == "qpas" ? "sparse" : "dense",
                                       r["status"] == "optimal" ? "opt." : r["status"],
                                       tex_int(num(r["iterations"])), tex_int(num(r["columns"])),
                                       tex_time(num(r["total_s"])),
                                       tex_num(1000 * num(r["master_s"]) / num(r["iterations"]); digits = 2),
                                       tex_int(num(r["refactorizations"])), r["final_N"],
                                       ex_ok ? tex_sci(bgap) : "--", tex_sci(num(r["infeasibility"]))]),
                                 " & "), " \\\\")
            end
            println(io, "\\midrule")
        end
    end
    # drop the trailing \midrule
    p = joinpath(DATA, "tab_scaling.tex")
    s = read(p, String)
    write(p, replace(s, r"\\midrule\s*$" => ""))
end

# ── 9. micro benchmark of the factor operations ─────────────────────────────────────────
let p = joinpath(RAW, "micro.csv")
    if isfile(p)
        _, mc = readcsv(p)
        ks = sort(unique(num.(getindex.(mc, "k"))))
        val(k, b, op) = (r = [x for x in mc if num(x["k"]) == k && x["backend"] == b && x["op"] == op];
                         isempty(r) ? NaN : num(r[1]["median_us"]))
        for op in ("solve", "append", "delete_first", "delete_middle", "update", "downdate",
                   "rebuild", "memory_MB")
            writecsv(joinpath(DATA, "micro_$(op).csv"), ["k", "sparse", "dense", "ratio"], ks,
                     [val(k, "cholmod", op) for k in ks], [val(k, "dense", op) for k in ks],
                     [val(k, "cholmod", op) / val(k, "dense", op) for k in ks])
        end
        open(joinpath(DATA, "tab_micro.tex"), "w") do io
            sel = [k for k in (100.0, 350.0, 1000.0, 2000.0) if k in ks]
            for (op, label) in (("solve", "solve"), ("append", "append / row addition at the end"),
                                ("delete_first", "delete the oldest index"),
                                ("delete_middle", "delete a middle index"),
                                ("update", "rank-one update"), ("downdate", "rank-one downdate"),
                                ("rebuild", "refactorization (\$k\$ appends)"),
                                ("memory_MB", "memory [MB]"))
                cells = [label]
                for k in sel
                    s, d = val(k, "cholmod", op), val(k, "dense", op)
                    f(x) = op == "memory_MB" ? tex_num(x; digits = 1) : x >= 1000 ? tex_int(x) : tex_num(x; digits = 1)
                    push!(cells, f(s), f(d))
                end
                println(io, join(cells, " & "), " \\\\")
            end
        end
    end
end

# ── 10. adaptive stabilization: fixed and geometric δ (section F of generate_data.jl) ───
const ADAPT_FILE = joinpath(RAW, "adaptive.csv")
if isfile(ADAPT_FILE)
    _, adapt = readcsv(ADAPT_FILE)
    arow(name, frac, pricing, variant) = begin
        rs = [r for r in adapt if r["instance"] == name && num(r["frac"]) == frac &&
                                  r["pricing"] == pricing && r["variant"] == variant]
        isempty(rs) ? nothing : rs[1]
    end
    const PRICINGS = ("first_improving", "all_blocks")
    const TIME_VARIANTS = ("fixed_0.1", "fixed_1", "fixed_3", "geometric_1", "geometric_3")
    present = [(n, f) for n in MAIN_INSTANCES, f in (0.0, 0.3)
               if arow(n, f, "first_improving", "fixed_1") !== nothing]
    sort!(present; by = x -> (findfirst(==(x[1]), MAIN_INSTANCES), x[2]))
    ineq(f) = f == 0 ? "--" : "30\\%"
    tex_err(x) = x < 1e-15 ? "\$<10^{-15}\$" : tex_sci(x)
    # time [s] of all variants, the fastest of each pricing rule in bold
    open(joinpath(DATA, "tab_adaptive_time.tex"), "w") do io
        for (n, f) in present
            cells = [n, ineq(f)]
            for pr in PRICINGS
                t = [num(arow(n, f, pr, v)["total_s"]) for v in TIME_VARIANTS]
                best = argmin(t)
                for (k, x) in enumerate(t)
                    push!(cells, k == best ? "\\textbf{" * tex_time(x) * "}" : tex_time(x))
                end
            end
            println(io, join(cells, " & "), " \\\\")
        end
    end
    # CG iterations (column-adding iterations) of the thesis setting and the geometric rule
    open(joinpath(DATA, "tab_adaptive_iter.tex"), "w") do io
        for (n, f) in present
            cells = [n, ineq(f)]
            for pr in PRICINGS, v in ("fixed_1", "geometric_1", "geometric_3")
                r = arow(n, f, pr, v)
                push!(cells, tex_int(num(r["iterations"])) * " (" *
                             tex_int(num(r["column_iterations"])) * ")")
            end
            println(io, join(cells, " & "), " \\\\")
        end
    end
    # accuracy: relative cost error and infeasibility, fixed δ vs the geometric rule
    open(joinpath(DATA, "tab_adaptive_acc.tex"), "w") do io
        for (n, f) in present
            zstar = num(crow(n, f, "exact")["exact_objective"])
            cells = [n, ineq(f)]
            for (pr, v) in (("first_improving", "fixed_1"), ("first_improving", "geometric_3"),
                            ("all_blocks", "fixed_1"), ("all_blocks", "geometric_1"))
                r = arow(n, f, pr, v)
                push!(cells, tex_err(abs(num(r["objective"]) - zstar) / abs(zstar)),
                      tex_err(num(r["infeasibility"])))
            end
            println(io, join(cells, " & "), " \\\\")
        end
    end
    # traces of I2 (equality variant): master objective error and δ over the iterations
    if crow("I2", 0.0, "exact") !== nothing
        zstar = num(crow("I2", 0.0, "exact")["exact_objective"])
        for pr in PRICINGS, v in ("fixed_1", "geometric_1", "geometric_3")
            tp = joinpath(RAW, "trace_I2_0_adapt_$(pr)_$(v).csv")
            isfile(tp) || continue
            _, tr = readcsv(tp)
            it = col(tr, "iteration")
            gap = abs.(col(tr, "master_objective") .- zstar) ./ abs(zstar)
            δ = col(tr, "delta")
            idx = thin(length(it), 1500)
            writecsv(joinpath(DATA, "adapt_I2_0_$(pr)_$(v).csv"), ["it", "gap", "delta"],
                     it[idx], max.(gap[idx], 1e-16), δ[idx])
        end
    end
end

# ── 11. range rows (section G of generate_data.jl) ─────────────────────────────────────
const RANGE_FILE = joinpath(RAW, "range.csv")
if isfile(RANGE_FILE)
    _, rng = readcsv(RANGE_FILE)
    rrow(name, variant) = begin
        rs = [r for r in rng if r["instance"] == name && r["variant"] == variant]
        isempty(rs) ? nothing : rs[1]
    end
    open(joinpath(DATA, "tab_range.tex"), "w") do io
        for n in MAIN_INSTANCES
            r, q = rrow(n, "range"), rrow(n, "split")
            (r === nothing || q === nothing) && continue
            zge = crow(n, 0.3, "exact") === nothing ? NaN : num(crow(n, 0.3, "exact")["exact_objective"])
            zr = num(r["exact_objective"])
            it(x) = tex_int(num(x["iterations"])) * " (" * tex_int(num(x["column_iterations"])) * ")"
            println(io, join([n, tex_int(num(r["range_rows"])), tex_num(zge), tex_num(zr),
                              tex_int(num(r["m"])), it(r), tex_int(num(r["bound_flips"])),
                              tex_time(num(r["total_s"])),
                              tex_int(num(q["m"])), it(q), tex_time(num(q["total_s"])),
                              tex_sci(abs(num(r["objective"]) - zr) / abs(zr))], " & "), " \\\\")
        end
    end
end

println("plot data written to ", DATA)
