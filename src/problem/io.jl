"""
    save_instance(path, instance)

Write an instance to disk (Julia `Serialization` format; read back with [`load_instance`](@ref)
using the same package version).
"""
function save_instance(path::AbstractString, inst::CrewSchedulingInstance)
    mkpath(dirname(abspath(path)))
    open(io -> serialize(io, inst), path, "w")
    return path
end

"""
    load_instance(path) -> CrewSchedulingInstance
"""
function load_instance(path::AbstractString)
    inst = open(deserialize, path)
    inst isa CrewSchedulingInstance || error("$path does not contain a CrewSchedulingInstance")
    return inst
end
