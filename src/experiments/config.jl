# Experiment configuration files (TOML).
#
# A config file has an optional [defaults] section and any number of [[experiment]] entries.
# Every experiment is `defaults` deep-merged with its own settings, so an experiment only
# lists what differs. See `experiments.toml` in the project root for the full annotated list.
#
#   [defaults.solvers]          which solvers run (exact, lp, explicit_qp, eliminated_qp, qpas)
#   [defaults.instance]         InstanceConfig fields (instance generator)
#   [defaults.column_generation] CGOptions fields
#   [defaults.stabilization]    StabilizationOptions fields
#   [defaults.qpas]             QPASOptions fields
#   [defaults.lp]               big_m
#   [defaults.output]           directory, traces, save_instance

"Solver keys accepted in the `[solvers]` section, in the order they are run."
const SOLVER_KEYS = ("exact", "lp", "explicit_qp", "eliminated_qp", "qpas")

const _SECTIONS = ("name", "solvers", "instance", "column_generation", "stabilization", "qpas",
                   "lp", "output")

"""
    Experiment

One fully resolved experiment from a config file: the instance to generate, the solvers to
run and all solver settings.
"""
struct Experiment
    name::String
    instance::InstanceConfig
    solvers::Vector{String}
    cg_options::CGOptions
    qpas_options::QPASOptions
    big_m::Float64
    output_dir::String
    write_traces::Bool
    save_instance::Bool
end

"""
    load_experiments(path) -> Vector{Experiment}

Read a TOML experiment file. Unknown sections or keys raise an error that lists the allowed
names, so typos never pass silently.
"""
function load_experiments(path::AbstractString)
    isfile(path) || throw(ArgumentError("config file not found: $path"))
    data = TOML.parsefile(path)
    # only [defaults] and [[experiment]] are allowed at the top level
    for key in keys(data)
        key in ("defaults", "experiment") ||
            error("$path: unknown top-level entry `$key` (allowed: [defaults], [[experiment]])")
    end
    defaults = get(data, "defaults", Dict{String,Any}())
    entries = get(data, "experiment", Any[])
    entries isa Vector || error("$path: use [[experiment]] (double brackets) for experiments")
    isempty(entries) && error("$path: no [[experiment]] entries")

    experiments = Experiment[]
    for (k, entry) in enumerate(entries)
        # each experiment = defaults overridden by its own entries; errors name the experiment
        name = get(entry, "name", "experiment_$k")
        settings = _deep_merge(defaults, entry)
        try
            push!(experiments, _build_experiment(String(name), settings))
        catch err
            err isa ErrorException || rethrow()
            error("$path, experiment \"$name\": $(err.msg)")
        end
    end
    names = [e.name for e in experiments]
    allunique(names) || error("$path: experiment names must be unique, got $names")
    return experiments
end

function _build_experiment(name::String, s::Dict{String,Any})
    for key in keys(s)
        key in _SECTIONS || error("unknown section `$key` (allowed: $(join(_SECTIONS, ", ")))")
    end

    # enabled solvers, validated and kept in the fixed SOLVER_KEYS run order
    solver_table = get(s, "solvers", Dict{String,Any}())
    for key in keys(solver_table)
        key in SOLVER_KEYS || error("unknown solver `$key` in [solvers] (allowed: $(join(SOLVER_KEYS, ", ")))")
        solver_table[key] isa Bool || error("[solvers] $key must be true or false")
    end
    solvers = [key for key in SOLVER_KEYS if get(solver_table, key, false)]
    isempty(solvers) && error("no solver enabled in [solvers]")

    # instance name comes from the experiment; stabilization is nested into CGOptions
    instance = _build(InstanceConfig, get(s, "instance", Dict{String,Any}()), "instance";
                      exclude = (:name,), extra = (name = name,))
    stabilization = _build(StabilizationOptions, get(s, "stabilization", Dict{String,Any}()),
                           "stabilization"; exclude = (:initial_center,))
    cg = _build(CGOptions, get(s, "column_generation", Dict{String,Any}()), "column_generation";
                exclude = (:stabilization,), extra = (stabilization = stabilization,))
    validate(cg)
    qpas = validate(_build(QPASOptions, get(s, "qpas", Dict{String,Any}()), "qpas"))

    # [lp] only configures the Big-M penalty of the LP master
    lp = get(s, "lp", Dict{String,Any}())
    for key in keys(lp)
        key == "big_m" || error("unknown key `$key` in [lp] (allowed: big_m)")
    end
    big_m = Float64(get(lp, "big_m", 1e4))

    out = get(s, "output", Dict{String,Any}())
    for key in keys(out)
        key in ("directory", "write_traces", "save_instance") ||
            error("unknown key `$key` in [output] (allowed: directory, write_traces, save_instance)")
    end
    return Experiment(name, instance, solvers, cg, qpas, big_m,
                      String(get(out, "directory", "results")),
                      Bool(get(out, "write_traces", true)), Bool(get(out, "save_instance", false)))
end

# Build a @kwdef struct from a TOML table, converting values to the field types
# (strings → Symbol where a Symbol is expected, integers → Float64, ...).
function _build(::Type{T}, table::AbstractDict, section::String;
                exclude::Tuple = (), extra::NamedTuple = NamedTuple()) where {T}
    allowed = [f for f in fieldnames(T) if !(f in exclude)]
    kwargs = Pair{Symbol,Any}[]
    for (key, value) in table
        field = Symbol(key)
        field in allowed ||
            error("unknown key `$key` in [$section] (allowed: $(join(allowed, ", ")))")
        push!(kwargs, field => _convert_field(fieldtype(T, field), value, section, key))
    end
    return T(; kwargs..., pairs(extra)...)
end

_convert_field(::Type{Symbol}, v::AbstractString, _, _) = Symbol(v)
function _convert_field(::Type{F}, v, section, key) where {F}
    try
        return convert(F, v)
    catch
        error("[$section] $key = $(repr(v)) cannot be converted to $F")
    end
end

function _deep_merge(a::AbstractDict, b::AbstractDict)
    out = Dict{String,Any}(a)
    for (k, v) in b
        # recurse into sub-tables present on both sides; otherwise b's value wins
        out[k] = (haskey(out, k) && out[k] isa AbstractDict && v isa AbstractDict) ?
                 _deep_merge(out[k], v) : v
    end
    return out
end
