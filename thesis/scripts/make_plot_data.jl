# Derives the plotting tables (data/*.csv) and LaTeX table snippets (data/tab_*.tex) of the
# results chapter from the raw experiment files in data/raw (see generate_data.jl).
#
#   julia thesis/scripts/make_plot_data.jl
#
# Uses only Base Julia (no packages).

using Printf, Statistics

const DATA = joinpath(@__DIR__, "..", "data")
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

num(x) = x == "" ? NaN : parse(Float64, x)
col(rows, k) = [num(r[k]) for r in rows]

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
    m = x / 10.0^e
    return @sprintf("\$%.1f\\cdot 10^{%d}\$", m, e)
end
tex_int(x) = begin
    s = string(round(Int, x))
    # thin space as thousands separator
    parts = String[]
    while length(s) > 3
        pushfirst!(parts, s[end-2:end]); s = s[1:end-3]
    end
    pushfirst!(parts, s)
    join(parts, "\\,")
end

const SOLVER_NAME = Dict("exact" => "Exact LP (HiGHS)", "lp" => "LP CG (HiGHS)",
                         "explicit_qp" => "Explicit QP CG (HiGHS)",
                         "eliminated_qp" => "Eliminated QP CG (HiGHS)",
                         "qpas" => "Hot-started QPAS CG")

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

# ── 2. instances table ──────────────────────────────────────────────────────────────────
_, inst = readcsv(joinpath(RAW, "instances.csv"))
open(joinpath(DATA, "tab_instances.tex"), "w") do io
    for name in unique(r["instance"] for r in inst)
        count(r -> r["instance"] == name, inst) == 2 || continue
        r0 = first(r for r in inst if r["instance"] == name && num(r["frac"]) == 0.0)
        r3 = first(r for r in inst if r["instance"] == name && num(r["frac"]) == 0.3)
        println(io, join([name, r0["V"], tex_int(num(r0["E"])), r0["pilots"], r0["m"],
                          r3["m_ineq"], tex_int(num(r0["N"]))], " & "), " \\\\")
    end
end

# ── 3. comparison tables ────────────────────────────────────────────────────────────────
_, comp = readcsv(joinpath(RAW, "comparison.csv"))
function comparison_table(frac)
    io = IOBuffer()
    names = unique(r["instance"] for r in comp)
    for (ni, name) in enumerate(names)
        rows = [r for r in comp if r["instance"] == name && num(r["frac"]) == frac]
        isempty(rows) && continue
        order = ["exact", "lp", "explicit_qp", "eliminated_qp", "qpas"]
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

# ── 4. QPAS counters and time breakdown ─────────────────────────────────────────────────
_, det = readcsv(joinpath(RAW, "qpas_details.csv"))
getdet(name, frac, key) = begin
    r = [d for d in det if d["instance"] == name && num(d["frac"]) == frac && d["key"] == key]
    isempty(r) ? "" : r[1]["value"]
end
open(joinpath(DATA, "tab_qpas_counters.tex"), "w") do io
    for name in unique(r["instance"] for r in inst), frac in (0.0, 0.3)
        g(k) = getdet(name, frac, k)
        g("active-set iterations") == "" && continue
        steps = split(g("primal steps (full)"), " ")
        full = strip(steps[2], ['(', ')'])
        slack = g("slack expansions / reductions")
        slack = slack == "" ? "--" : replace(slack, " / " => " / ")
        nfree = g("final free slacks |N_s|")
        println(io, join([name, frac == 0 ? "--" : "30\\%", tex_int(num(g("active-set iterations"))),
                          tex_int(num(steps[1])) * " (" * tex_int(num(full)) * ")",
                          tex_int(num(g("working-set expansions"))),
                          g("of which representative swaps"),
                          tex_int(num(g("working-set reductions"))), slack,
                          g("singular-Hessian steps"), tex_int(num(g("CHOLMOD row updates"))),
                          g("refactorizations"), g("final |N| (nullspace dim.)"),
                          nfree == "" ? "--" : nfree], " & "), " \\\\")
    end
end
open(joinpath(DATA, "breakdown.csv"), "w") do io
    println(io, "label,direction,multipliers,step,updates,refactor,pricing")
    for name in unique(r["instance"] for r in inst), frac in (0.0, 0.3)
        g(k) = num(getdet(name, frac, k))
        getdet(name, frac, "time direction [s]") == "" && continue
        dir, mul, stp = g("time direction [s]"), g("time multipliers [s]"), g("time step [s]")
        exp_, red = g("time expand [s]"), g("time reduce [s]")
        ref = g("time refactor [s] (within expand/reduce)")
        r = first(c for c in comp if c["instance"] == name && num(c["frac"]) == frac && c["solver"] == "qpas")
        label = frac == 0 ? "$(name) eq." : "$(name) ineq."
        tot = dir + mul + stp + exp_ + red + num(r["pricing_s"])
        shares = [dir, mul, stp, exp_ + red - ref, ref, num(r["pricing_s"])] ./ tot .* 100
        println(io, join(vcat(label, [@sprintf("%.2f", x) for x in shares]), ","))
    end
end

# ── 5. convergence traces (I2) ──────────────────────────────────────────────────────────
for (tag, frac) in (("I2_0", 0.0), ("I2_30", 0.3), ("I3_0", 0.0), ("I3_30", 0.3))
    name = split(tag, "_")[1]
    zstar = num(first(r for r in comp if r["instance"] == name && num(r["frac"]) == frac)["exact_objective"])
    for solver in ("lp", "explicit_qp", "eliminated_qp", "qpas")
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
        if solver == "qpas"
            # phases: per bin of 100 iterations, share of iterations that added a column
            B = 100
            nb = cld(length(it), B)
            mid = [(b - 1) * B + B / 2 for b in 1:nb]
            share = [mean(added[((b - 1) * B + 1):min(b * B, length(it))] .> 0) for b in 1:nb]
            writecsv(joinpath(DATA, "phases_$(tag).csv"), ["it", "share"], mid, share)
        end
    end
end

# ── 6. center-only statistics table ─────────────────────────────────────────────────────
open(joinpath(DATA, "tab_phases.tex"), "w") do io
    for name in unique(r["instance"] for r in inst), frac in (0.0, 0.3)
        tag = "$(name)_$(round(Int, 100frac))"
        p = joinpath(RAW, "trace_$(tag)_qpas.csv")
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

# ── 7. diagnostics: sets and fill ───────────────────────────────────────────────────────
for tag in ("I2_0", "I2_30", "I3_0", "I3_30", "I4_0", "I4_30")
    p = joinpath(RAW, "diag_$(tag)_qpas.csv")
    isfile(p) || continue
    _, d = readcsv(p)
    name = split(tag, "_")[1]
    frac = endswith(tag, "_30") ? 0.3 : 0.0
    m = num(first(r for r in inst if r["instance"] == name && num(r["frac"]) == frac)["m"])
    it = col(d, "iteration")
    N = col(d, "N")
    Ns = col(d, "N_s")
    Ns = [isnan(x) ? 0.0 : x for x in Ns]
    writecsv(joinpath(DATA, "sets_$(tag).csv"), ["it", "N", "Ns", "bound", "m"],
             it, N, Ns, m .- Ns, fill(m, length(it)))
end
for name in ("I2", "I3"), tau in (0, 25, 150, 600)
    p = joinpath(RAW, "diag_$(name)_tau$(tau).csv")
    isfile(p) || continue
    _, d = readcsv(p)
    writecsv(joinpath(DATA, "fill_$(name)_tau$(tau).csv"),
             ["it", "stored", "effective", "dense", "reach", "N", "cholmod"],
             col(d, "iteration"), col(d, "factor_stored"), col(d, "factor_effective"),
             col(d, "factor_dense"), col(d, "reach") ./ max.(col(d, "N"), 1), col(d, "N"),
             col(d, "cholmod_MB"))
end
let p = joinpath(RAW, "refactor.csv")
    if isfile(p)
        _, rf = readcsv(p)
        for name in ("I2", "I3")
            rows = [r for r in rf if r["instance"] == name]
            finite = sort([r for r in rows if num(r["tau"]) > 0]; by = r -> num(r["tau"]))
            never = [r for r in rows if num(r["tau"]) == 0]
            # finite intervals as a line; "never" (tau = 0) as a separate point at x = 2000,
            # separated by a NaN row so that pgfplots (unbounded coords=jump) breaks the line
            xs = vcat([num(r["tau"]) for r in finite], 1000.0, [2000.0 for _ in never])
            ts = vcat(col(finite, "total_s"), NaN, col(never, "total_s"))
            writecsv(joinpath(DATA, "refactor_$(name).csv"), ["tau", "time"], xs, ts)
        end
        open(joinpath(DATA, "tab_refactor.tex"), "w") do io
            for name in ("I2", "I3")
                rows = [r for r in rf if r["instance"] == name]
                for (k, r) in enumerate(rows)
                    tau = num(r["tau"])
                    dp = joinpath(RAW, "diag_$(name)_tau$(round(Int, tau)).csv")
                    _, d = readcsv(dp)
                    st, ef = col(d, "factor_stored"), col(d, "factor_effective")
                    reach = col(d, "reach") ./ max.(col(d, "N"), 1)
                    peak = maximum(col(d, "cholmod_peak_MB"))
                    waste = mean([s == 0 ? 0.0 : (s - e) / s for (s, e) in zip(st, ef)])
                    first_col = k == 1 ? "\\multirow{$(length(rows))}{*}{$name}" : ""
                    println(io, join([first_col, tau == 0 ? "never" : string(round(Int, tau)),
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
    # I1, I2 from the comparison run, I3–S6 from the stress runs (same machine and code)
    pts = Dict{Tuple{String,Float64},Tuple{Float64,Float64,Float64}}()   # (m, qpas, exact)
    for name in ("I1", "I2", "I3", "I4"), frac in (0.0, 0.3)
        rq = first(r for r in comp if r["instance"] == name && num(r["frac"]) == frac && r["solver"] == "qpas")
        re = first(r for r in comp if r["instance"] == name && num(r["frac"]) == frac && r["solver"] == "exact")
        m = num(first(r for r in inst if r["instance"] == name && num(r["frac"]) == frac)["m"])
        pts[(name, frac)] = (m, num(rq["total_s"]), num(re["total_s"]))
    end
    for f in ("scaling_stress1.csv", "scaling_stress2.csv")
        _, rows = readcsv(joinpath(RAW, f))
        for r in rows
            r["instance"] in ("I1", "I2", "I3", "I4") && continue   # taken from the comparison
            ex = haskey(r, "exact_s") && num(r["exact_s"]) > 0 ? num(r["exact_s"]) : NaN
            pts[(r["instance"], num(r["ineq"]))] = (num(r["m"]), num(r["total_s"]), ex)
        end
    end
    order = ["I1", "I2", "I3", "I4", "S5", "S6"]
    for (frac, suffix) in ((0.0, "eq"), (0.3, "ineq"))
        keys_ = [(n, frac) for n in order if haskey(pts, (n, frac))]
        writecsv(joinpath(DATA, "scaling_$(suffix).csv"), ["name", "m", "qpas", "exact"],
                 first.(keys_), [pts[k][1] for k in keys_], [pts[k][2] for k in keys_],
                 [pts[k][3] for k in keys_])
    end
end

println("plot data written to ", DATA)
