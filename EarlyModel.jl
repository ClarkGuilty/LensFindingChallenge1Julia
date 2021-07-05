using AstroImages
using Flux, ParameterSchedulers
using Images
using DataFrames, CSV
using Plots
using Statistics: mean, std
using BenchmarkTools
using CUDA
using ProgressMeter
using BSON: @save, @load
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

function preprocess_labels!(labels)
  onehotbatch(labels,0:1)
end
od(W,F,P=0,S=1) = 1 + Int((W-F+2*P)/S)

loss_f(x,y) = logitcrossentropy(f(x),y)
using Flux: OneHotMatrix
compare(y::OneHotMatrix, y′) = maximum(y′, dims = 1) .== maximum(y .* y′, dims = 1)
accuracy(x, y::OneHotMatrix) = mean(compare(y, f(x)))

function loss_and_accuracy(x,y)
  ŷ = f(x)
  logitcrossentropy(ŷ,y), mean(compare(y, ŷ))
end

function loss_and_accuracy((x,y),batchsize::Int64)
  loss = 0; accuracy = 0; i = 0
  dataloader = Flux.Data.DataLoader((x, y), batchsize=batchsize, shuffle=false)
  for (a,b) in dataloader
    a,b = gpu(a), gpu(b)
    ŷ = f(a)
    loss += logitcrossentropy(ŷ,b)
    accuracy += mean(compare(b, ŷ))
    i += 1
    #CUDA.unsafe_free!(a)
    a = nothing
    b = nothing
  end
  #dataloader = nothing
  loss / i, accuracy / i, i
end

#Using LASTRO architecture
##
oftf(x, y) = oftype(float(x), y)
function mrelu(x) 
  a = oftf(x,2*π)
  b = oftf(x,sqrt(π-1))
  (a * max(zero(x),x)-1)/b
end

l1c1 =  Conv((4, 4), 1=>16, pad=(0,0), mrelu) # 98,98,16 -> 256, 272
l1c2 =  Conv((3, 3), 16=>16, pad=(0,0), mrelu) # 96,96,16 -> 2304, 2320
l1p1 = MaxPool((2,2),pad=0, stride=2)       # 48,48,16
l1b1 = Flux.BatchNorm(16,affine=true)       # 48,48,16 -> 32
l1 = Chain(l1c1,l1c2,l1p1,l1b1)
#
l2c1 =  Conv((3, 3), 16=>16, pad=(0,0), mrelu)# 46,46,16 -> 2304, 2320
l2c2 =  Conv((3, 3), 16=>16, pad=(0,0), mrelu)#44,44,16 -> 2304, 2320
l2p1 = MaxPool((2,2),pad=0, stride=2)        #22,22,16
l2b1 = Flux.BatchNorm(16,affine=true)        #22,22,16 -> 32

l2 = Chain(l2c1,l2c2,l2p1,l2b1)
#
l3c1 =  Conv((3, 3), 16=>16, pad=(0,0), mrelu)#20,20,16 -> 2304, 2320
l3c2 =  Conv((3, 3), 16=>16, pad=(0,0), mrelu)#18,18,16 -> 2304, 2320
l3p1 = MaxPool((2,2),pad=0, stride=2)        #9, 9, 16
l3b1 = Flux.BatchNorm(16,affine=true)        #9, 9, 16 -> 32
l3b2 = Flux.Dropout(0.5,dims=3)
l3 = Chain(l3c1,l3c2,l3p1,l3b1,l3b2)
#
l4c1 =  Conv((3, 3), 16=>32, pad=(0,0), mrelu)#7, 7, 32 -> 4608, 4640
l4d1 = Flux.Dropout(0.5,dims=3)
l4c2 =  Conv((3, 3), 32=>32, pad=(0,0), mrelu)#5, 5, 32 -> 9216, 9248
l4b1 = Flux.BatchNorm(32,affine=true)        #5, 5, 32 -> 64
l4d2 = Flux.Dropout(0.5,dims=3)
l4 = Chain(l4c1,l4c2,l4b1)
#
lf0 = Flux.flatten
lf1 = Dense(800,512)                          #512  -> 409600, 410112
lfd1 = Flux.Dropout(0.5,dims=1)
lfb1 = Flux.BatchNorm(512,affine=true)        #512 -> 1026
lf2 = Dense(512,2)
l5 = Chain(lf0,lf1)

f = Chain(l1c1,l1c2,l1p1,l1b1,
              l2c1,l2c2,l2p1,l2b1,
              l3c1,l3c2,l3p1,l3b1,l3b2,
              l4c1,l4d1,l4c2,l4b1,l4d2,
              lf0,lf1,lfd1,lfb1,lf2) # 437058
#f0 = gpu(Chain(l1c1,l1c2,l1p1,l1b1,l2c1,l2c2,l2p1,l2b1,l3c1,l3c2,l3p1,l3b1,l3b2,l4c1,l4d1,l4c2,l4b1)) # 24338
##

first_execution = true
if first_execution
  opt = ADAM()
  history_loss = []
  history_accuracy = []
  iterations = []
else
  @load "earlyModel.bson" weights opt history_loss history_accuracy iterations#Comment if first execution.
  Flux.loadparams!(f, weights) #Comment if first execution.
end
f = gpu(f)
parameters = Flux.params(f) 


##


##Load data

#(train_images, train_labels) = getobs(trainData,1:nobs(trainData))
#(train_images, train_labels) = getobs(trainData,1:100)
(test_images, test_labels) = getobs(testData,1:nobs(testData))

#train_images = preprocess_images_gpu!(train_images)
test_images = preprocess_images!(test_images)
#train_labels = preprocess_labels_gpu!(train_labels)
test_labels = preprocess_labels!(test_labels)

#test_loader = Flux.Data.DataLoader((test_images, test_labels), batchsize=60, shuffle=false)
#@benchmark CUDA.@sync loss_and_accuracy((test_images, test_labels), 800)
#accuracy(train_images,train_labels)
a,b, _ = loss_and_accuracy((test_images, test_labels), 30)
##

##Assumes test data is already in gpu.
"Trains model. trainData must be an ImageDataset, testData a gpu tuple, nbatches and nepochs are number of batches and epochs respectectively, loss is the loss function, opt and parameters are the optimizer and its parameters, and accuracy is the accuracy function."
function my_train!(trainData::ImageDataset, testData, nbatches, nepochs, loss , opt, parameters,accuracy, history_loss, history_accuracy, iterations)
  batchSize = floor(Int,nobs(trainData)/nbatches)
  index(index,batch,batchSize=batchSize) = index + (batch-1)*batchSize
  i_base = 0
  length(iterations) > 0 ? i_base = iterations[end] : nothing
  #s = ParameterSchedulers.Stateful((Exp(λ = 5e-1, γ = 0.9)))
  t_loss, t_accuracy, _ = loss_and_accuracy((testData[1],testData[2]),400)
  push!(history_accuracy, t_accuracy)
  push!(history_loss, t_loss)
  push!(iterations, i_base)
  println("Initial state, accuracy: $t_accuracy, loss: $t_loss")
  #Serial and batchwise loading. Move to dataloaders when
  println("Training $nepochs epochs with $nbatches batches of size $batchSize.")
  p = Progress(nepochs*nbatches)
  for j in 1:nepochs
    for i in 1:nbatches
      k = index(1,i)
      (train_images, train_labels) = getobs(trainData,k:k+batchSize)
      train_images = preprocess_images_gpu!(train_images)
      train_labels = preprocess_labels_gpu!(train_labels)

      #opt.eta = ParameterSchedulers.next!(s)
      #i % 5 == 0 ? push!(history_loss, accuracy(testData[1],testData[2])) : nothing
      if (i+(j-1)*nbatches+i_base) % 100 == 0
        testmode!(f,true)
        t_loss, t_accuracy, _ = loss_and_accuracy((testData[1],testData[2]),400)
        push!(history_accuracy, t_accuracy)
        push!(history_loss, t_loss)
        push!(iterations, i+i_base)
        println("Epoch $j, Batch $i, accuracy: $t_accuracy, loss: $t_loss")
        trainmode!(f,true)
      end
      g = Flux.gradient(() -> loss(train_images, train_labels), parameters)
      Flux.update!(opt, parameters, g)
      ProgressMeter.next!(p)
    end
  end
end

function alternative_train!(trainData::ImageDataset, testData, nbatches, nepochs, loss , opt, parameters,accuracy, history_loss, history_accuracy, iterations)
  batchSize = floor(Int,nobs(trainData)/nbatches)
  index(index,batch,batchSize=batchSize) = index + (batch-1)*batchSize
  i_base = 0
  length(iterations) > 0 ? i_base = iterations[end] : nothing
  #s = ParameterSchedulers.Stateful((Exp(λ = 5e-1, γ = 0.9)))
  t_loss, t_accuracy, _ = loss_and_accuracy((testData[1],testData[2]),400)
  push!(history_accuracy, t_accuracy)
  push!(history_loss, t_loss)
  push!(iterations, i_base)
  println("Initial state, accuracy: $t_accuracy, loss: $t_loss")
  #Serial and batchwise loading. Move to dataloaders when
  println("Training $nbatches batches (of size $batchSize) for $nepochs epochs each.")
  p = Progress(nepochs*nbatches)
  for i in 1:nbatches
    k = index(1,i)
    (train_images, train_labels) = getobs(trainData,k:k+batchSize)
    train_images = preprocess_images_gpu!(train_images)
    train_labels = preprocess_labels_gpu!(train_labels)
    for j in 1:nepochs
      #opt.eta = ParameterSchedulers.next!(s)
      if Int((i-1)*nepochs+j+i_base) % 100 == 0
        testmode!(f,true)
        t_loss, t_accuracy, _ = loss_and_accuracy((testData[1],testData[2]),400)
        push!(history_accuracy, t_accuracy)
        push!(history_loss, t_loss)
        push!(iterations, i+(j-1)*nbatches+i_base)
        println("\nBatch $i, Epoch $j, accuracy: $t_accuracy, loss: $t_loss")
        trainmode!(f,true)
      end
      g = Flux.gradient(() -> loss(train_images, train_labels), parameters)
      Flux.update!(opt, parameters, g)
      ProgressMeter.next!(p)
    end
  end
end
#my_train!(trainData, (test_images, test_labels), 10, 1,


##
alternative_train!(trainData, (test_images, test_labels), 50, 1,loss_f,
  opt,parameters,accuracy,history_loss, history_accuracy,iterations)
sleep(60)
alternative_train!(trainData, (test_images, test_labels), 50, 2,loss_f,
  opt,parameters,accuracy,history_loss, history_accuracy,iterations)
  
#accuracy(test_images,test_labels)
##


plot(log.(history_loss),label=:none); plot!(twinx(),history_accuracy, color = :orange, label=:none)

#f = cpu(f)

# my_train!(trainData, (test_images, test_labels), 50, 10,loss_f,
#     opt,parameters,accuracy,history_loss, history_accuracy,iterations)



    
weights = collect(Flux.params(cpu(f)))
@save "earlyModel.bson" weights opt history_loss history_accuracy iterations
