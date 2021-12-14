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
@everywhere using DistributedArrays
@everywhere using CUDA
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


@everywhere function init_distributed_data_node(tup)
    tup_len = length(tup[1])
    
    ret = Array{sparse_data_node, 1}(undef, tup_len)
    for i = 1 : tup_len
        ret[i] = sparse_data_node(zeros(Float64, 0), zeros(Int64, 0), 0)
    end
    return ret
end

@everywhere function fill_with_data!(dis::DArray, data; id_mapping=[])
    loc = localpart(dis)
    loc_idx = localindices(dis)[1]
    if isempty(id_mapping)
        for i = 1 : length(data)
            loc[i] = sparse_data_node(copy(data[i].rating), copy(data[i].userId), loc_idx[1] + i - 1)
        end
    else
        if id_mapping == -1
            id_mapping = 1 : length(data)
        end
        for i = 1 : length(data)
            loc[i] = sparse_data_node(copy(data[i].rating), copy(id_mapping[data[i].movieId]), loc_idx[1] + i - 1)
        end
    end
end



@everywhere function initialize_movie(how_much_to_read::Int64)
     # read in data
     rating_csv = []
     if how_much_to_read > 0    
        rating_csv = CSV.File(open("ml-25m/ratings.csv"); limit=how_much_to_read, select=[1,2,3])
     else
        rating_csv = CSV.File(open("ml-25m/ratings.csv"); select=[1,2,3])
     end
     df = DataFrame(rating_csv)

     return df
end

@everywhere function select_subset(df)
    max_user = maximum(df[:, 1])
    max_product = maximum(df[:, 2])
    user_count = zeros(Int64, max_user)
    product_count = zeros(Int64, max_product)
    for i = 1 : size(df, 1)
        user_count[df[i, 1]] += 1
        product_count[df[i, 2]] += 1
    end

    gen = Binomial(1, 0.8)
    train_indices = zeros(Int64, 0)
    test_indices = zeros(Int64, 0)
    for i = 1 : size(df, 1)
        decide = rand(gen)
        if decide == 0
            if user_count[df[i, 1]] - 1 > 0 && product_count[df[i, 2]] - 1 > 0
                push!(test_indices, i)
                user_count[df[i, 1]] -= 1
                product_count[df[i, 2]] -= 1
            else
                push!(train_indices, i)
            end
        else
            push!(train_indices, i)
        end
    end
    
    return df[train_indices, :], df[test_indices, :]

end


@everywhere function movie_problem(how_much_to_read::Int64; rank_k=100, regularization=0.1)
    df_total = initialize_movie(how_much_to_read)
    df, df_test = select_subset(df_total)

    gd_col = groupby(df, :movieId; sort=true)
    probes = workers()
    
    

    index_set = Vector{Tuple{Int64, Int64}}(undef, length(probes))
    total = 0
    for i = 1 : length(gd_col)
        total += size(gd_col[i], 1)
    end
    tracker = 0
    last_one = 1
    cur_idx = 1
    for i = 1 : length(gd_col)
        tracker += size(gd_col[i], 1)
        if tracker > 1 / length(probes) * total
            index_set[cur_idx] = (last_one, i)
            tracker = 0
            last_one = i + 1
            cur_idx += 1
        end
    end
    if last_one != length(gd_col) + 1
        index_set[end] = (last_one, length(gd_col))
    end

    

    # stored in compressed column format
    #dis_col = DArray(tup -> init_distributed_data_node(tup), (length(gd_col),), probes, (length(probes),))
    fut = Vector{Future}(undef, length(probes))
    @sync for p_idx in eachindex(probes)
        @async fut[p_idx] = remotecall_wait(init_distributed_data_node, probes[p_idx], (index_set[p_idx][1] : index_set[p_idx][2],))
    end
    dis_col = DArray(fut)

    dis_indices = dis_col.cuts[1]
    @sync for p_idx in eachindex(probes)
        @async remotecall_fetch(fill_with_data!, probes[p_idx], dis_col, gd_col[dis_indices[p_idx] : dis_indices[p_idx + 1] - 1])
    end

    #=
    sparse_data_matrix_column = Array{sparse_data_node}(undef, length(gd_col))
    for i = 1 : length(gd_col)
        sparse_data_matrix_column[i] = sparse_data_node(copy(gd_col[i].rating), copy(gd_col[i].userId), i)
    end
    =#

    gd_col_keys = keys(gd_col)
    id_mapping = zeros(Int64, gd_col_keys[end].movieId)
    for i = 1 : length(gd_col)
        id_mapping[gd_col_keys[i].movieId] = i
    end

    # stored in compressed row format

    gd = groupby(df, :userId; sort=true)

    #=
    sparse_data_matrix_row = Array{sparse_data_node}(undef, length(gd))
    for i = 1 : length(gd)
        sparse_data_matrix_row[i] = sparse_data_node(copy(gd[i].rating), copy(id_mapping[gd[i].movieId]), i)
    end
    =#

    # compressed row format
    dis_row = DArray(tup -> init_distributed_data_node(tup), (length(gd),), probes, (length(probes),))
    dis_indices = dis_row.cuts[1]
    @sync for p_idx in eachindex(probes)
        @async remotecall_fetch(fill_with_data!, probes[p_idx], dis_row, gd[dis_indices[p_idx] : dis_indices[p_idx + 1] - 1], id_mapping=id_mapping)
    end

    

    
    #iter = [1, 5, 10, 50]
    iter = [5]
    # create U and M
    dis_U = dzeros((rank_k, length(dis_row)), probes, [1, length(probes)])
    #dis_M = dzeros((rank_k, length(dis_col)), probes, [1, length(probes)])
    fut = Matrix{Future}(undef, 1, length(probes))
    @sync for p_idx in eachindex(probes)
        @async fut[p_idx] = remotecall_wait(zeros, probes[p_idx], rank_k, index_set[p_idx][2] - index_set[p_idx][1] + 1)
    end
    dis_M = DArray(fut)


    # calculate fetching index
    M_indices = Vector{Future}(undef, length(probes))
    @sync for p_idx in eachindex(probes)
        @async M_indices[p_idx] = remotecall_wait(locate_indices, probes[p_idx], dis_row, dis_M)
    end

    U_indices = Vector{Future}(undef, length(probes))
    @sync for p_idx in eachindex(probes)
        @async U_indices[p_idx] = remotecall_wait(locate_indices, probes[p_idx], dis_col, dis_U)
    end

    
    for max_iter in iter
        @sync for p_idx in eachindex(probes)
            @async remotecall_wait(initialize_U_M!, probes[p_idx], dis_U, dis_M, dis_row, dis_col, rank_k)
        end

        @time for i = 1 : max_iter
            # prefetch needed information 
            M_mat = Vector{Future}(undef, length(probes))
            @sync for p_idx in eachindex(probes)
                @async M_mat[p_idx] = remotecall_wait(communicate_info, probes[p_idx], M_indices[p_idx], dis_M, rank_k)
            end
            # perform calculations (fix M, optimize over U)
            @sync for p_idx in eachindex(probes)
                @async remotecall_fetch(compute_factor!, probes[p_idx], dis_U, dis_row, rank_k, regularization, M_indices[p_idx], M_mat[p_idx])
            end

            # other way around, fix U optimize over M
            U_mat = Vector{Future}(undef, length(probes))
            @sync for p_idx in eachindex(probes)
                @async U_mat[p_idx] = remotecall_wait(communicate_info, probes[p_idx], U_indices[p_idx], dis_U, rank_k)
            end

            @sync for p_idx in eachindex(probes)
                @async remotecall_fetch(compute_factor!, probes[p_idx], dis_M, dis_col, rank_k, regularization, U_indices[p_idx], U_mat[p_idx])
            end

            
            
        end

        # MSRE
        U = Array(dis_U)
        M = Array(dis_M)
        mse = 0
        for i = 1 : size(df_test, 1)
            x = df_test[i, 1]
            y = id_mapping[df_test[i, 2]]
            @views err = dot(U[:, x], M[:, y]) - df_test[i, 3]
            mse += err^2
        end
        println("MSRE after iter ", max_iter, " : ", sqrt(mse / size(df_test, 1)))
        
        
        # verify
        println(size(dis_U))
        println(size(dis_M))
        #=
        Ut = Array(dis_U)'
        M = Array(dis_M)
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



