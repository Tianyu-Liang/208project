using Distributed
@everywhere using Distributed
@everywhere using CSV
@everywhere using DataFrames
@everywhere using Statistics
@everywhere using Distributions
@everywhere using LinearAlgebra
@everywhere using IterativeSolvers
include("algorithm.jl")

@everywhere struct sparse_data_node{T1<:Union{Array{Float64, 1}, Array{Int64, 1}}, 
    T2<:Array{Int64, 1}, T3<:Int64}
    value::T1
    index::T2
    actual_location::T3
end


# id_csv = CSV.File(open("ml-25m/movies.csv"); select=[1,2])
# movieId = id_csv.movieId
# id_mapping = zeros(Int64, maximum(movieId))
# for i in eachindex(movieId)
#     id_mapping[movieId[i]] = i
# end

# read in data
rating_csv = CSV.File(open("ml-25m/ratings.csv"); limit=20000, select=[1,2,3])
df = DataFrame(rating_csv)

# stored in compressed column format
gd_col = groupby(df, :movieId; sort=true)
sparse_data_matrix_column = Array{sparse_data_node}(undef, length(gd_col))
for i = 1 : length(gd_col)
    sparse_data_matrix_column[i] = sparse_data_node(copy(gd_col[i].rating), copy(gd_col[i].userId), i)
end

gd_col_keys = keys(gd_col)
id_mapping = zeros(Int64, gd_col_keys[end].movieId)
for i = 1 : length(gd_col)
    id_mapping[gd_col_keys[i].movieId] = i
end

# stored in compressed row format
gd = groupby(df, :userId)
sparse_data_matrix_row = Array{sparse_data_node}(undef, length(gd))
for i = 1 : length(gd)
    sparse_data_matrix_row[i] = sparse_data_node(copy(gd[i].rating), copy(id_mapping[gd[i].movieId]), i)
end


iter = [1, 5, 10, 50]
for i in iter
    @time Ut, M = compute_factor(sparse_data_matrix_row, sparse_data_matrix_column, 100, 0.1, i)

    # verify
    actual = zeros(length(gd), length(gd_col))
    for i = 1 : length(gd)
        actual[i, id_mapping[gd[i].movieId]] = gd[i].rating
    end
    zero_index = findall(x -> x == 0, actual)
    calculated = Ut * M
    calculated[zero_index] .= 0
    println(norm(calculated - actual) / norm(actual))
end