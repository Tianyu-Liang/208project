using CSV
using DataFrames

id_csv = CSV.File(open("ml-25m/movies.csv"); select=[1,2])
movieId = id_csv.movieId
id_mapping = zeros(Int64, maximum(movieId))
for i in eachindex(movieId)
    id_mapping[movieId[i]] = i
end

rating_csv = CSV.File(open("ml-25m/ratings.csv"); limit=10000, select=[1,2,3])
df = DataFrame(rating_csv)
gd = groupby(df, :userId)

a = [0]
for i in eachindex(gd)
    gd[i]
end