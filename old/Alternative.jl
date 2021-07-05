using AstroImages
using Flux, ParameterSchedulers
using Images
using DataFrames, CSV
using Plots
using Statistics: mean, std
using BenchmarkTools
using CUDA
#Using DataLoaders' example.
##

#Dataset struct, holds filenames, the dir_path, IDs, and classes.
struct ImageDataset
    files::Vector{String}
    dir::String
    IDs::Vector{Int64}
    classes::Vector{Int64}
end

#Gets the ID from a filename.
get_id(ss::String) = parse(Int64,(split(split(ss,"-")[2],".")[1]))
const train_data_dir = "Data/"
const test_data_dir = "TestData/"

#Constructor.
function ImageDataset(data_dir::String, classification_dir::String)
  ImageDataset(readdir(data_dir),data_dir,get_id.(readdir(data_dir)),
  DataFrame(CSV.File(classification_dir)).is_lens)
end
trainData = ImageDataset(train_data_dir,"trainClassifications.csv")
testData = ImageDataset(test_data_dir,"testClassifications.csv")

function nobs(ds::ImageDataset)
  length(ds.IDs)
end

function getobs(dataset::ImageDataset, i::Int)
  subpath = dataset.files[i]
  file = joinpath(dataset.dir, subpath)
  data = reshape(AstroImage(file).data[1],101*101)
  label = dataset.classes[i]
  data, label
end
give_data(ai::AstroImage) = reshape(ai.data[1],10201)

#Having a getobs(dataset::ImageDataset, i) fails, so I wrote each version.
function getobs(dataset::ImageDataset, range::UnitRange{Int64})
  subpath = dataset.files[range]
  file = joinpath.(dataset.dir, subpath)
  data = hcat(give_data.(AstroImage.(file))...)
  label = dataset.classes[range]
  data, label
end

##
using Flux: train!, @epochs, Chain, throttle, onecold, onehotbatch, logitcrossentropy

#accuracy(x,y) = mean(round.(ff(x)) .== y)

##
#Trains once on the entire trainData in the number of "batches" defined.
function my_train_single_epoch!(trainData, testData, batches, loss , opt, parameters,accuracy)
  batchSize = round(Int,nobs(trainData)/batches)
  index(index,batch,batchSize=batchSize) = index + (batch-1)*batchSize
  #evalcb() = @show(accuracy(images,labels))
  s = Exp(λ = 1e-1, γ = 0.9)

  #Serial and batchwise loading. Move to dataloaders when
  for (η , i) in zip(s,1:batches)
    (images,labels) = getobs(trainData,index(1,i):index(batchSize,i))
    #i % 10 == 0 ? println(η,  " - ", accuracy(testImages,testLabels)) : nothing
    g = Flux.gradient(() -> loss((images), labels), parameters)
    Flux.update!(opt, parameters, g)
  end
  println(accuracy(testData...))
end
##
#Train in single batches
function my_train_epocs!((images,labels), testData, nepochs, loss , opt, parameters,accuracy)
  #evalcb() = @show(accuracy(images,labels))
  s = Exp(λ = 1e-1, γ = 0.9)
  #Serial and batchwise loading. Move to dataloaders when
  for (η , i) in zip(s,1:nepochs)
    #i % 10 == 0 ? println(η,  " - ", accuracy(testImages,testLabels)) : nothing
    g = Flux.gradient(() -> loss(images, labels), parameters)
    Flux.update!(opt, parameters, g)
  end
  println(accuracy(testData...))
end
##
function preprocess!(images) 
  scale!(images)
  standarize!(images)
 end
function scale!(images) 
 @views for i in 1:size(images)[2]
    images[:,i] = images[:,i] ./ maximum(images[:,i])
  end
end
function standarize!(images)
  @views for i in 1:size(images)[2]
    images[:,i] = (images[:,i] .- mean(images[:,i])) ./ std(images[:,i])
  end
end
##
(train_images, train_labels) = getobs(trainData,1:nobs(trainData))
(test_images, test_labels) = getobs(testData,1:nobs(testData))
train_labels = onehotbatch(train_labels,0:1)
test_labels = onehotbatch(test_labels,0:1)
preprocess!(train_images)
preprocess!(test_images)
##

f = gpu(Chain(Dense(101*101,101,relu), Dense(101,2)))
parameters = Flux.params(f)
loss_f(x,y) = logitcrossentropy(f(x),y)
#accuracy(x,y) = mean(round.(f(x)) .== y)
accuracy(x,y) = mean(onecold(cpu(f(x))) .== onecold(cpu(y)))
println(accuracy(test_images,test_labels))
##


train_images = gpu(train_images)
train_labels = gpu(train_labels)
test_images = gpu(test_images)
test_labels = gpu(test_labels)
##
my_train!((train_images, train_labels), (test_images, test_labels), 4, 3,
 loss_f, Descent(),parameters,accuracy)


function my_train!((images,labels), testData, batches, nepochs, loss , opt, parameters,accuracy)
  batchSize = round(Int,nobs(trainData)/batches)
  index(index,batch,batchSize=batchSize) = index + (batch-1)*batchSize
  #evalcb() = @show(accuracy(images,labels))
  s = ParameterSchedulers.Stateful((Exp(λ = 5e-1, γ = 0.9)))

  #Serial and batchwise loading. Move to dataloaders when
  for j in 1:nepochs
    for i in 1:batches
      opt.eta = ParameterSchedulers.next!(s)
      i % 5 == 0 ? println(opt.eta,  " - ", accuracy(testData[1],testData[2])) : nothing
      g = Flux.gradient(() -> loss(images, labels), parameters)
      Flux.update!(opt, parameters, g)
    end
  end
  println(accuracy(testData[1],testData[2]))
end

##
#Exploring new architectures.
od(W,F,P=0,S=1) = 1 + Int((W-F+2*P)/S)
od(101,3,1)

test = copy(reshape(test_images,101,101,1,:))
F = 4
C_out = 16
#Using LASTRO architecture
##
l1c1 =  Conv((4, 4), 1=>8, pad=(0,0), relu) # 98,98,8
l1c2 =  Conv((3, 3), 8=>8, pad=(0,0), relu) # 96,96,8
l1p1 = MaxPool((2,2),pad=0, stride=2)       # 48,48,8
l1 = Chain(l1c1,l1c2,l1p1)
#
l2c1 =  Conv((3, 3), 8=>16, pad=(0,0), relu)# 46,46,16
l2c2 =  Conv((3, 3), 16=>16, pad=(0,0), relu)#44,44,16
l2p1 = MaxPool((2,2),pad=0, stride=2)        #22,22,16
l2 = Chain(l2c1,l2c2,l2p1)
#
l3c1 =  Conv((3, 3), 16=>16, pad=(0,0), relu)#20,20,16
l3c2 =  Conv((3, 3), 16=>16, pad=(0,0), relu)#18,18,16
l3p1 = MaxPool((2,2),pad=0, stride=2)        #9, 9, 16
l3 = Chain(l3c1,l3c2,l3p1)
#
l4c1 =  Conv((3, 3), 16=>32, pad=(0,0), relu)#7, 7, 32
l4c2 =  Conv((3, 3), 32=>32, pad=(0,0), relu)#5, 5, 32
l4 = Chain(l4c1,l4c2)
#
lf0 = Flux.flatten
lf1 = Dense(800,512)
lf2 = Dense(512,2)
l5 = Chain(lf0,lf1,lf2)

f = gpu(Chain(l1,l2,l3,l4,l5))
##

size(f(test))
t4 = f(test)
onecold(cpu(t4))

#f = gpu(f)
parameters = Flux.params(f)
loss_f(x,y) = logitcrossentropy(f(x),y)
accuracy(x,y) = mean(onecold(cpu(f(x))) .== onecold(cpu(y)))

##
(train_images, train_labels) = getobs(trainData,1:nobs(trainData))
(test_images, test_labels) = getobs(testData,1:nobs(testData))
train_labels = onehotbatch(train_labels,0:1)
test_labels = onehotbatch(test_labels,0:1)
preprocess!(train_images)
preprocess!(test_images)

train_images = gpu(reshape(train_images,101,101,1,:))
train_labels = gpu(train_labels)
test_images = gpu(reshape(test_images,101,101,1,:))
test_labels = gpu(test_labels)
##
accuracy(test_images,test_labels)
##
my_train!((train_images, train_labels), (test_images, test_labels), 10, 1,
 loss_f, Descent(),parameters,accuracy)

##
if CUDA.functional()
  @info "Training on CUDA GPU"
  CUDA.allowscalar(false)
  device = gpu
else
  @info "Training on CPU"
  device = cpu
end