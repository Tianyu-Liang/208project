
overall = 0
iter_overall = 0
@everywhere PART_ONE = 1
@everywhere PART_TWO = 2
# return U and M such that A ~= U^TM, k is the rank, lambda is the regularizing constant
@everywhere function compute_factor(dis_U, dis_M, dis_row, dis_col, k::Int64, lambda::Float64, max_iter::Int64)
    A_row = localpart(dis_row)
    A_col = localpart(dis_col)
    U = localpart(dis_U)
    M = localpart(dis_M)
    sampling_factor = 2.5
    # initialize M by assigning average movie of that movie as first row

    for i = 1 : length(A_col)
        M[1, i] = mean(A_col[i].value)
    end
    M[2 : end, :] = rand(Normal(0, 2), k - 1, size(M, 2))
    

    
    # find dimension of placeholder
    max_len = 0
    for i = 1 : size(U, 2)
        max_len = max(length(A_row[i].index), max_len)
    end
    for i = 1 : size(M, 2)
        max_len = max(length(A_col[i].index), max_len)
    end
    max_len = max(max_len, k)

    # placeholder1 stores the sketch matrix
    placeholder1 = zeros(Int64(ceil(sampling_factor * max_len)) + 1, k + 1)
    # placeholder 2 stores the concatenated matrix of the new system
    placeholder2 = zeros(max_len + k + 1, k + 1)

    for iter = 1 : max_iter
        println("iter: ", iter)
        # fix M, optimize over U
        @time Threads.@threads for i = 1 : size(U, 2)
        #for i = 1 : 10
            user_movie = A_row[i].index
            M_sub = M[:, user_movie]
            #U[:, i] = (M_sub * M_sub' + lambda * I) \ (M_sub * A_row[i].value)
            t1 = time()
            #@time x, ch = IterativeSolvers.lsqr(M_sub', A_row[i].value; damp=sqrt(lambda), atol=0, log=true)
            U[:, i], stats = Krylov.lsqr(M_sub', A_row[i].value, λ=sqrt(lambda))
            #println("no pre cond: ", cond(M_sub'))
            #println("no pre time: ", time() - t1)
            #println(ch)
            #@time U[:, i] = solve_with_preconditioner(M_sub', A_row[i].value, sampling_factor, lambda, placeholder1, placeholder2)
            #U[:, i] = solve_with_preconditioner(M_sub', A_row[i].value, sampling_factor, lambda)
            if mod(i, 1000) == 0
                println("no pre cond: ", cond(M_sub'))
            end
        end
        println("qr overall: ", overall)
        println("iter overall: ", iter_overall)
        # fix U, optimize over M
        
        #@time Threads.@threads for i = 1 : size(M, 2)
        for i = 1 : 10
            movie_user = A_col[i].index
            U_sub = U[:, movie_user]
            #M[:, i] = (U_sub * U_sub' + lambda * I) \ (U_sub * A_col[i].value)
            #M[:, i] = IterativeSolvers.lsqr(U_sub', A_col[i].value; damp=sqrt(lambda), atol=0)
            #M[:, i] = solve_with_preconditioner(U_sub', A_col[i].value, sampling_factor, lambda, placeholder1, placeholder2)
            M[:, i] = solve_with_preconditioner(U_sub', A_col[i].value, sampling_factor, lambda)
            @time M[:, i], stats = Krylov.lsqr(U_sub', A_col[i].value, λ=sqrt(lambda), history=true)
            println("non preconditioned iter: ", length(stats.residuals))
            println("no pre cond: ", cond([U_sub'; sqrt(lambda) * I]))
            if mod(i, 1000) == 0
                println("no pre cond: ", cond([U_sub'; sqrt(lambda) * I]))
            end
        end
        
    end

    return U', M
end

@everywhere in_range(x::Int64, idx_range::Array{Int64, 1}) = (x >= idx_range[1] && x <= idx_range[2])


@everywhere function locate_indices(A_row_or_col, dis_U_or_M)
    s = Dict{Int64, Tuple}() # tracks repetition
    s_location_tup = Vector{Vector{Tuple{Int64, Int64}}}() # track for each index, where their location (in the fetched array) is
    probes = procs(dis_U_or_M) 
    process_idx = Vector{Vector{Int64}}() # tracks the indices each machine has
    for p in probes
        push!(process_idx, zeros(Int64, 0))
    end

    for i in eachindex(A_row_or_col)
        indices = A_row_or_col[i].index
        push!(s_location_tup, Vector{Tuple{Int64, Int64}}())
        loc_vec = s_location_tup[end]

        for j_idx in eachindex(indices)
            j = indices[j_idx]
            relative_idx = DistributedArrays.locate(dis_U_or_M, j)[PART_ONE]
            idx = probes[relative_idx]
            # put j and the tuple(which machine, how many from machine so far) in map
            correspond = length(process_idx[relative_idx]) + 1
            tup = get!(s, j, (relative_idx, correspond))

            # check if j has already been added
            if tup == (relative_idx, correspond) # if new
                idx_array = process_idx[relative_idx]
                push!(idx_array, j)
            end

            # store the location for look up later
            push!(loc_vec, tup)
                
            
        end
    end

    # find the direct index (rather than tuple) that corresponds to the location in fetched array
    individual_len = zeros(Int64, length(process_idx))
    for i in eachindex(individual_len)
        individual_len[i] = length(process_idx[i])
    end
    cumulative = [0; cumsum(individual_len)]
    s_location_vec = Vector{Vector{Int64}}()
    for vec in s_location
        last_vec = zeros(Int64, length(vec))
        for i in eachindex(vec)
            tup = vec[i]
            last_vec[i] = cumulative[tup[PART_ONE]] + tup[PART_TWO]
        end
        push!(s_location_vec, last_vec)
    end
    
    return process_idx, s_location_vec
    
end


why = []
other=[]
@everywhere function solve_with_preconditioner(A,  b::Vector, sampling_factor, lambda::Float64, placeholder1, placeholder2)
    m, n = size(A)
    if m < n
        A = A'
    end
    height = Int64(ceil(min(n, m) * sampling_factor))
    #S = rand(Normal(), height, max(n, m) + min(m, n))
    S = SRFT(height)

    # svd method
    
    #=
    @time svd_object = svd([S * A; sqrt(lambda) * I])
    cutoff = findall(x -> x < 1e-9, svd_object.S)
    C = []
    if isempty(cutoff)
        C = svd_object.V * Diagonal(inv.(svd_object.S))
    else
        C = svd_object.V[:, 1 : cutoff[1]] * Diagonal(inv.(svd_object.S[1 : cutoff]))
    end
    =#
    
    # qr method
    

    # setup placeholder 2
    placeholder2[1 : max(m, n), 1 : min(m, n)] .= A
    placeholder2[1 : m, min(m, n) + 1] .= b
    for r = max(m, n) + 1 : max(m, n) + min(m, n)
        comp = r - max(m, n)
        for c = 1 : min(m, n)
            if comp != c
                placeholder2[r, c] = 0
            else
                placeholder2[r, c] = sqrt(lambda)
            end
        end
    end
    
    # set up placeholder 1
    t1 = time()
    @time @views mul!(placeholder1[1 : height, 1 : min(n, m) + 1], S, placeholder2[1 : max(m, n) + min(m, n), 1 : min(m, n) + 1])
    #@views mul!(placeholder1[1 : height, 1 : min(n, m)], S, A)
    #=
    for r = height + 1 : height + min(n, m)
        comp = r - height
        for c = 1 : min(n, m)
            if comp != c
                placeholder1[r, c] = 0
            else
                placeholder1[r, c] = sqrt(lambda)
            end
        end
    end
    =#
    
    global overall += (time() - t1)

    
    
    @time @views Q, R = qr(placeholder1[1 : height, 1 : min(n, m)])
    #~, R = qr([S * A; sqrt(lambda) * I])
    
    
    #println("qr time: ", time() - t1)
    println(size(A))
    


    
    #println("prev_error: ", norm(placeholder2[1 : max(m, n) + min(m, n), 1 : min(m, n)] - [A; sqrt(lambda) * I]))
    @time @views BLAS.trsm!('R', 'U', 'N', 'N', 1.0, R, placeholder2[1 : max(m, n) + min(m, n), 1 : min(m, n)])
    @views new_system = placeholder2[1 : max(m, n) + min(m, n), 1 : min(m, n)]
    initial = placeholder1[1 : height, min(m, n) + 1]
    @time initial = Matrix(Q)' * initial


    #new_system = (R' \ [A; sqrt(lambda) * I]')'
    #println("without pre cond: ", cond([A; sqrt(lambda) * I]))
    #println("with pre cond: ", cond(new_system))
    t2 = time()
    if m >= n
        #z = lsqr((R' \ [A; sqrt(lambda) * I]')', [b; zeros(n)])
        #z, ch = lsqr(new_system, [b; zeros(n)]; log=true, verbose=true)
        @time lsqr!(initial, new_system, [b; zeros(n)])
        #gg, ch1 = lsqr(A, b; damp=sqrt(lambda), log=true, verbose=true)
        #lsqr(A, rand(m); damp=sqrt(lambda), log=true, verbose=true)
        #println(norm(A * (R \ initial) - b) + norm(R \ initial))
        #println(norm(A * gg - b) + norm(gg))
        #println(cond([A; sqrt(lambda) * I]))
        #println(cond(new_system))
        global iter_overall += (time() - t2)
        return R \ initial
    else
        #@time z = lsqr((A * C)', C' * b * sqrt(lambda))
        
        
        #z, ch = lsqr(new_system, R' \ (b * sqrt(lambda)); log=true)
        #println(ch)
        #trsv!(ul, tA, dA, A, b)
        rightside = BLAS.trsv('U', 'T', 'N', R, (b * sqrt(lambda)))
        z, ch = lsmr(new_system', rightside; log=true)
        #global iter_overall += (time() - t2)
        return z[1 : n] / sqrt(lambda)
    end
    
end


@everywhere function solve_with_preconditioner(A,  b::Vector, sampling_factor, lambda::Float64)
    m, n = size(A)
    if m < n
        A = A'
    end
    height = Int64(ceil(min(n, m) * sampling_factor))
    #S = rand(Normal(), height, max(n, m) + min(m, n))
    #S = rand(Normal(), height, max(n, m))
    S = SRFT(height)

    t1 = time()
    extended = [A; sqrt(lambda) * I]
    A_hat = S * extended
    b_hat = S * reshape(b, (length(b), 1))
    b_hat = b_hat[:]
    global overall += (time() - t1)

    println("t1: ", time() - t1)
    println("size: ", size(A))
    
    #@time Q, R = qr([A_hat; sqrt(lambda) * I])
    Q, R = qr(A_hat)
    #~, R = qr([S * A; sqrt(lambda) * I])
    
    
    #println("qr time: ", time() - t1)
    # println(size(A))
    # println(size(Q))
    # println(size(b_hat))


    
    #println("prev_error: ", norm(placeholder2[1 : max(m, n) + min(m, n), 1 : min(m, n)] - [A; sqrt(lambda) * I]))

    BLAS.trsm!('R', 'U', 'N', 'N', 1.0, R, extended)
    new_system = extended
    #initial = Matrix(Q)'[:, 1 : height] * b_hat
    #initial = zeros(n)

    #new_system = (R' \ [A; sqrt(lambda) * I]')'
    #println("without pre cond: ", cond([A; sqrt(lambda) * I]))
    #println("with pre cond: ", cond(new_system))
    t2 = time()
    if m >= n
        #z = lsqr((R' \ [A; sqrt(lambda) * I]')', [b; zeros(n)])
        #z, ch = lsqr(new_system, [b; zeros(n)]; log=true, verbose=true)
        #IterativeSolvers.lsqr!(initial, new_system, [b; zeros(n)]; log=true, verbose=true)
        #gg, ch1 = lsqr(A, b; damp=sqrt(lambda), log=true, verbose=true)
        #lsqr(A, rand(m); damp=sqrt(lambda), log=true, verbose=true)
        @time initial, info = Krylov.lsqr(new_system, [b; zeros(n)], history=true)
        println("preconditioned num iter: ", length(info.residuals))
        #println(norm(A * (R \ initial) - b) + norm(R \ initial))
        #println(norm(A * gg - b) + norm(gg))
        #println(cond([A; sqrt(lambda) * I]))
        println("precond cond number: ", cond(new_system))
        global iter_overall += (time() - t2)
        return R \ initial
    else
        #@time z = lsqr((A * C)', C' * b * sqrt(lambda))
        
        
        #z, ch = lsqr(new_system, R' \ (b * sqrt(lambda)); log=true)
        #println(ch)
        #trsv!(ul, tA, dA, A, b)
        rightside = BLAS.trsv('U', 'T', 'N', R, (b * sqrt(lambda)))
        #z, ch = IterativeSolvers.lsmr(new_system', rightside; log=true)
        z, stat = Krylov.lsmr(new_system', rightside)
        #global iter_overall += (time() - t2)
        return z[1 : n] / sqrt(lambda)
    end
    
end



@everywhere function compute_global_preconditioner(A, UM, total_len, sampling_factor, lambda::Float64)
    count = 0
    for i = 1 : total_len
        count += length(A[i].index)
    end

    all_indices = zeros(Int64, count)
    current = 1
    for i = 1 : total_len
        cur_indices = A[i].index
        all_indices[current : current + length(cur_indices) - 1] .= cur_indices
        current += length(cur_indices)
    end

    O = UM[:, all_indices]
    m, n = size(O)
    if m < n
        O = O'
    end
    S = rand(Normal(), min(n, m) * sampling_factor, max(n, m))
    @time Q, R = qr([S * O; sqrt(lambda) * I])
    println("size R: ", size(R))
    return R

end







