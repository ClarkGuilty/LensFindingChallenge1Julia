using AstroImages
using Flux, ParameterSchedulers
using Images
using DataFrames, CSV
using Plots
using Statistics: mean, std
using BenchmarkTools
using CUDA
using ProgressMeter
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
const train_data_dir = "DataTrain/"
const test_data_dir = "DataTest/"

#Constructor.
function ImageDataset(data_dir::String, classification_dir::String)
  ImageDataset(readdir(data_dir),data_dir,get_id.(readdir(data_dir)),
  DataFrame(CSV.File(classification_dir)).is_lens)
end
trainData = ImageDataset(train_data_dir,"ClassificationsTrain.csv")
testData = ImageDataset(test_data_dir,"ClassificationsTest.csv")

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

function preprocess_images_gpu!(images)
  scale!(images)
  standarize!(images)
  return reshape(gpu(images),101,101,1,:)
end

function preprocess_images!(images)
  scale!(images)
  standarize!(images)
  return reshape(images,101,101,1,:)
end

function preprocess_labels_gpu!(labels)
  gpu(onehotbatch(labels,0:1))
end
od(W,F,P=0,S=1) = 1 + Int((W-F+2*P)/S)

#Using LASTRO architecture
##
mrelu(x) = (sqrt(2*π) * max(zero(x),x)-1)/sqrt(π-1)

l1c1 =  Conv((4, 4), 1=>8, pad=(0,0), mrelu) # 98,98,8 -> 128, 136
l1c2 =  Conv((3, 3), 8=>8, pad=(0,0), mrelu) # 96,96,8 -> 576, 584
l1p1 = MaxPool((2,2),pad=0, stride=2)       # 48,48,8
l1b1 = Flux.BatchNorm(8)
l1 = Chain(l1c1,l1c2,l1p1,l1b1)
#
l2c1 =  Conv((3, 3), 8=>16, pad=(0,0), mrelu)# 46,46,16 -> 1152, 1168
l2c2 =  Conv((3, 3), 16=>16, pad=(0,0), mrelu)#44,44,16 -> 2304, 2320
l2p1 = MaxPool((2,2),pad=0, stride=2)        #22,22,16
l2b1 = Flux.BatchNorm(16)
l2 = Chain(l2c1,l2c2,l2p1,l2b1)
#
l3c1 =  Conv((3, 3), 16=>16, pad=(0,0), mrelu)#20,20,16 -> 2304, 2320
l3c2 =  Conv((3, 3), 16=>16, pad=(0,0), mrelu)#18,18,16 -> 2304, 2320
l3p1 = MaxPool((2,2),pad=0, stride=2)        #9, 9, 16
l3b1 = Flux.BatchNorm(16)
l3 = Chain(l3c1,l3c2,l3p1,l3b1)
#
l4c1 =  Conv((3, 3), 16=>32, pad=(0,0), mrelu)#7, 7, 32 -> 4608, 4640
l4c2 =  Conv((3, 3), 32=>32, pad=(0,0), mrelu)#5, 5, 32 -> 9216, 9248
l4b1 = Flux.BatchNorm(32)
l4 = Chain(l4c1,l4c2,l4b1)
#
lf0 = Flux.flatten
lf1 = Dense(800,2) #-> 1602
#lf2 = Dense(512,2)
l5 = Chain(lf0,lf1)

f = gpu(Chain(l1c1,l1c2,l1p1,l1b1,l2c1,l2c2,l2p1,l2b1,l3c1,l3c2,l3p1,l3b1,l4c1,l4c2,l4b1,lf0,lf1)) # 24338
##
parameters = Flux.params(f)
loss_f(x,y) = logitcrossentropy(f(x),y)
accuracy(x,y) = mean(onecold(cpu(f(x))) .== onecold(cpu(y)))

##Load data


#(train_images, train_labels) = getobs(trainData,1:nobs(trainData))
(train_images, train_labels) = getobs(trainData,1:7)
(test_images, test_labels) = getobs(testData,1:nobs(testData))

train_images = preprocess_images_gpu!(train_images)
test_images = preprocess_images_gpu!(test_images)
train_labels = preprocess_labels_gpu!(train_labels)
test_labels = preprocess_labels_gpu!(test_labels)
#accuracy(train_images,train_labels)
#accuracy(test_images,test_labels)
##
#loss_f(test_images,test_labels)



##Assumes test data is already in gpu.
function my_train!(trainData::ImageDataset, testData, batches, nepochs, loss , opt, parameters,accuracy)
  batchSize = floor(Int,nobs(trainData)/batches)
  index(index,batch,batchSize=batchSize) = index + (batch-1)*batchSize
  #evalcb() = @show(accuracy(images,labels))
  s = ParameterSchedulers.Stateful((Exp(λ = 5e-1, γ = 0.9)))
  #println(batchSize,"-",index(1,1),"-",index(1,batches))
  #Serial and batchwise loading. Move to dataloaders when
  p = Progress(nepochs*batches)
  for j in 1:nepochs
    for i in 1:batches
      k = index(1,i)
      (train_images, train_labels) = getobs(trainData,k:k+batchSize)
      train_images = preprocess_images_gpu!(train_images)
      train_labels = preprocess_labels_gpu!(train_labels)

      opt.eta = ParameterSchedulers.next!(s)
      #i+j % 5 == 0 ? println(opt.eta,  " - ", accuracy(testData[1],testData[2])) : nothing
      g = Flux.gradient(() -> loss(train_images, train_labels), parameters)
      Flux.update!(opt, parameters, g)
      ProgressMeter.next!(p)
    end
  end
end
#my_train!(trainData, (test_images, test_labels), 10, 1,
#loss_f, Descent(),parameters,accuracy)


my_train!(trainData, (test_images, test_labels), 50, 2,
loss_f, ADAM( ),parameters,accuracy)
accuracy(test_images,test_labels)

