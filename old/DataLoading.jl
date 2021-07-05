using AstroImages
using Flux, CUDA
using DataLoaders
import LearnBase: nobs, getobs
using Images
using Parameters
using DataFrames, CSV
#Using DataLoaders' example.
struct ImageDataset
    files::Vector{String}
end

@with_kw mutable struct Args
    lr::Float64 = 1e-3
    savepath::String = "./logs"
    data_dir = "Data/"
    feature_size::Int = 101*101
    target_size::Int = 2
    batch_size::Int = 64
    n_iters::Int = 80
    epochs::Int = 120
    device::Function = gpu
end
const train_data_dir = "Data/"
const test_data_dir = "TestData/"

nobs(ds::ImageDataset) = length(ds.files)
getobs(ds::ImageDataset, idx::Int) = return_data(AstroImage(data_dir*ds.files[idx]))
getobs(ds::ImageDataset, range::UnitRange{Int64}) = return_data.(AstroImage.(data_dir .* ds.files[range]))
return_data(ai::AstroImage) = ai.data

data = ImageDataset(readdir(data_dir)[1:30])

dataloader = DataLoaders.DataLoader(data, 5; collate = false, partial = false)


for images in dataloader
    println(nobs(images))
#    for image in images
 #       println(typeof(image))
  #  end
end


println("Igual continuó ejecutando")
#loaded = AstroImage.(data_dir .* data.files[1:3])
#loaded[1].data
