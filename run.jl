# Run the experiments of a TOML config file.
#
#   julia --project run.jl                          # all experiments in experiments.toml
#   julia --project run.jl --only I1,I2             # a selection, by name
#   julia --project run.jl my_experiments.toml      # another config file
#   julia --project run.jl --list                   # show the experiments and their solvers
#
# See experiments.toml for all parameters.

using HotstartedQPAS

function main(args)
    config = "experiments.toml"
    only = nothing
    list = false
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--only"
            i < length(args) || error("--only needs a comma-separated list of experiment names")
            only = String.(split(args[i + 1], ','))
            i += 2
        elseif a == "--list"
            list = true
            i += 1
        elseif startswith(a, "--")
            error("unknown option $a (use --only NAMES or --list)")
        else
            config = a
            i += 1
        end
    end

    if list
        for e in load_experiments(config)
            println(rpad(e.name, 16), join(e.solvers, ", "))
        end
        return
    end
    run_experiments(config; only)
end

main(ARGS)
