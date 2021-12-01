using Distributed
@everywhere using Distributed
@everywhere using CSV
@everywhere using DataFrames
@everywhere using Statistics
@everywhere using Distributions
@everywhere using LinearAlgebra
@everywhere using IterativeSolvers
@everywhere using FFTW
@everywhere using LowRankApprox
@everywhere using Krylov
# @everywhere using Krylov
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

function gooby()
    # read in data
    rating_csv = CSV.File(open("ml-25m/ratings.csv"); limit=2000000, select=[1,2,3])
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


    #iter = [1, 5, 10, 50]
    iter = [5]
    rank_k = 100
    regularization = 0.01
    for i in iter
        @time Ut, M = compute_factor(sparse_data_matrix_row, sparse_data_matrix_column, rank_k, regularization, i)

        # verify
        actual = zeros(length(gd), length(gd_col))
        for i = 1 : length(gd)
            actual[i, id_mapping[gd[i].movieId]] = gd[i].rating
        end
        zero_index = findall(x -> x == 0, actual)
        calculated = Ut * M
        calculated[zero_index] .= 0
        println(norm(calculated - actual) / norm(actual))
        println(size(actual))
    end
    

end

function dolan()
    m = 5000
    n = 6000
    k = 50
    observe_rate = 0.5
    regularization = 0.00001
    generate_U = rand(m, k) .* sqrt(5 / k)
    generate_M = rand(n, k) .* sqrt(5 / k)
    generator = Bernoulli(observe_rate)
    all_indices = CartesianIndex[]

    # compressed row scheme
    sparse_data_matrix_row = Array{sparse_data_node}(undef, m)
    element_mapping = Vector{Dict{Int64, Int64}}(undef, m)
    for i = eachindex(element_mapping)
        element_mapping[i] = Dict{Int64, Int64}()
    end
    for r = 1 : m
        count = 1
        for c = 1 : n
            if rand(generator) != 0
                push!(all_indices, CartesianIndex(r, c))
                get!(element_mapping[r], c, count)
                count += 1
            end
        end
    end
    for i = 1 : m 
        sparse_data_matrix_row[i] = sparse_data_node(zeros(Float64, length(element_mapping[i])), zeros(Int64, length(element_mapping[i])), i)
    end
    max_val = 0
    min_val = 5
    weight = [2,4,6,15,17]
    @time Threads.@threads for i in eachindex(all_indices)
        cur_index = all_indices[i]
        row = cur_index[1]
        col = cur_index[2]
        map_idx = get(element_mapping[row], col, -1)
        @assert map_idx != -1
        @views val = dot(generate_U[row, :], generate_M[col, :])
        #val = searchsortedfirst(weight, rand() * 17)
        sparse_data_matrix_row[row].value[map_idx] = val
        max_val = max(val, max_val)
        min_val = min(val, min_val)
        sparse_data_matrix_row[row].index[map_idx] = col
    end
    println("max_val: ", max_val)
    println("min_val: ", min_val)
    # compressed column scheme
    sparse_data_matrix_column = Array{sparse_data_node}(undef, n)
    for i = 1 : n
        sparse_data_matrix_column[i] = sparse_data_node(zeros(Float64, 0), zeros(Int64, 0), i)
    end
    for r = 1 : m
        cur_row = sparse_data_matrix_row[r]
        cur_index = cur_row.index
        cur_value = cur_row.value 
        for c = eachindex(cur_index)
            push!(sparse_data_matrix_column[cur_index[c]].index, r)
            push!(sparse_data_matrix_column[cur_index[c]].value, cur_value[c])
        end
    end

    
    iter = [1]
    rank_k = 1500
    for i in iter
        @time Ut, M = compute_factor(sparse_data_matrix_row, sparse_data_matrix_column, rank_k, regularization, i)

        # verify
        #=
        actual = zeros(length(gd), length(gd_col))
        for i = 1 : length(gd)
            actual[i, id_mapping[gd[i].movieId]] = gd[i].rating
        end
        zero_index = findall(x -> x == 0, actual)
        calculated = Ut * M
        calculated[zero_index] .= 0
        println(norm(calculated - actual) / norm(actual))
        println(size(actual))
        =#
    end
    
end