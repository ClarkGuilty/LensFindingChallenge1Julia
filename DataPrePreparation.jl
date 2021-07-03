using DataFrames, CSV
struct ImageDataset
    files::Vector{String}
    dir::String
    IDs::Vector{Int64}
end

#Gets the ID from a filename.
get_id(ss::String) = parse(Int64,(split(split(ss,"-")[2],".")[1]))
const data_dir = "DataTest/"

#"Initializer".
function ImageDataset(data_dir::String)
  ImageDataset(readdir(data_dir),data_dir,get_id.(readdir(data_dir)))
end
data = ImageDataset(data_dir)
data_df = DataFrame(CSV.File("classifications.csv"))
is_in_ID(n::Int64, df::ImageDataset)= Bool(n ∈ df.IDs)
is_in_ID(n::Int64) = is_in_ID(n::Int64, data)

data_df = data_df[is_in_ID.(data_df.ID),:]
CSV.write("ClassificationsTest.csv", data_df)
