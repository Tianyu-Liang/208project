

# return U and M such that A ~= U^TM, k is the rank, lambda is the regularizing constant
@everywhere function compute_factor(A_row, A_col, k::Int64, lambda::Float64, max_iter::Int64)
    U = zeros(k, length(A_row))
    M = zeros(k, length(A_col))

    # initialize M by assigning average movie of that movie as first row

    for i = 1 : length(A_col)
        M[1, i] = mean(A_col[i].value)
    end
    M[2 : end, :] = rand(k - 1, size(M, 2))

    for iter = 1 : max_iter
        # fix M, optimize over U
        for i = 1 : size(U, 2)
            user_movie = A_row[i].index
            M_sub = M[:, user_movie]
            #U[:, i] = (M_sub * M_sub' + lambda * I) \ (M_sub * A_row[i].value)
            U[:, i] = lsqr(M_sub', A_row[i].value; damp=sqrt(lambda), atol=0)

        end

        # fix U, optimize over M
        for i = 1 : size(M, 2)
            movie_user = A_col[i].index
            U_sub = U[:, movie_user]
            #M[:, i] = (U_sub * U_sub' + lambda * I) \ (U_sub * A_col[i].value)
            M[:, i] = lsqr(U_sub', A_col[i].value; damp=sqrt(lambda), atol=1e-12)
        end
    end

    return U', M
end

