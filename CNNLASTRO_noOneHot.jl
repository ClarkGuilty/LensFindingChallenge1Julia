using Images: data
#using ParameterSchedulers: IteratorEltype
using AstroImages
using Flux, ParameterSchedulers, Augmentor
using Images, Random
using DataFrames, CSV
using Plots
using Statistics: mean, std
using BenchmarkTools
using CUDA
using ProgressMeter, ParameterSchedulers
using BSON: @save, @load
using LsqFit
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
using Flux: train!, @epochs, Chain, throttle, onecold, onehotbatch, logitbinarycrossentropy,  logσ, σ, glorot_normal, label_smoothing

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
  gpu(label_smoothing(labels,0.2))
end

function preprocess_labels!(labels)
  label_smoothing(labels,0.2)
end
od(W,F,P=0,S=1) = 1 + Int((W-F+2*P)/S)

function fitnplot(x,y,model,p0)
    fit = LsqFit.curve_fit(model,x,y,p0)
    plot(x,[y,model(iterations,fit.param)], label = nothing)
end

function fit_exp(x,y)
    @. model(x,p) = p[1] * exp(-x*p[2]) + p[3]
    p0 = [1, 0.05, 0.5]
    fitnplot(x,y,model,p0)
end

#Using (similar) LASTRO architecture.
##
"Returns y in the float type of x."
oftf(x, y) = oftype(float(x), y)
"Returns the activation function used by lastro."
function mrelu(x) 
  a = oftf(x,2*π)
  b = oftf(x,sqrt(π-1))
  (a * max(zero(x),x)-1)/b
end
javier_normal(rng::AbstractRNG, dims...) = randn(rng, Float32, dims...) .* sqrt(1.0f0 / sum(Flux.nfan(dims...)))
javier_normal(dims...) = javier_normal(Random.GLOBAL_RNG, dims...)
javier_normal(rng::AbstractRNG) = (dims...) -> javier_normal(rng, dims...)


function instantiate_model()
                                           # shape -> number of parameters
  l1c1 =  Conv((4, 4), 1=>16, pad=(0,0), mrelu) # 98,98,16 -> 256, 272
  l1c2 =  Conv((3, 3), 16=>16, pad=(0,0), mrelu) # 96,96,16 -> 2304, 2320
  l1p1 = MaxPool((2,2),pad=0, stride=2)       # 48,48,16
  l1b1 = Flux.BatchNorm(16,affine=true)       # 48,48,16 -> 32
  #
  l2c1 =  Conv((3, 3), 16=>32, pad=(0,0), mrelu)# 46,46,32 -> 4608, 4640
  l2c2 =  Conv((3, 3), 32=>32, pad=(0,0), mrelu)#44,44,32 -> 9216, 9248
  l2p1 = MaxPool((2,2),pad=0, stride=2)        #22,22,32
  l2b1 = Flux.BatchNorm(32,affine=true)        #22,22,32 -> 64
  #
  l3c1 =  Conv((3, 3), 32=>64, pad=(0,0), mrelu)#20,20,64 -> 18432, 18496
  l3c2 =  Conv((3, 3), 64=>64, pad=(0,0), mrelu)#18,18,64 -> 36864, 36928
  l3p1 = MaxPool((2,2),pad=0, stride=2)        #9, 9, 64
  l3b1 = Flux.BatchNorm(64,affine=true)        #9, 9, 64 -> 128
  l3b2 = Flux.Dropout(0.5,dims=3)
  #
  l4c1 =  Conv((3, 3), 64=>128, pad=(0,0), mrelu)#7, 7, 128 -> 73728, 73856
  l4d1 = Flux.Dropout(0.5,dims=3)
  l4c2 =  Conv((3, 3), 128=>128, pad=(0,0), mrelu)#5, 5, 128 -> 147456, 147584
  l4b1 = Flux.BatchNorm(128,affine=true)        #5, 5, 128 -> 256
  l4d2 = Flux.Dropout(0.5,dims=3)
  #
  lf0 = Flux.flatten                              #800
  lf1 = Dense(3200,1024, mrelu, init=glorot_normal)                          #1024  -> 3276800, 3277824
  lfd1 = Flux.Dropout(0.5,dims=1)
  lf2 = Dense(1024,1024, mrelu, init=glorot_normal)                          #1024 -> 1048576, 1049600
  lfd2 = Flux.Dropout(0.5,dims=1)
  lf3 = Dense(1024,1024, mrelu,init=glorot_normal)                          #1024 -> 1048576, 1049600
  lfb1 = Flux.BatchNorm(1024,affine=true)         #1024-> 2048
  lf4 = Dense(1024,1,init=javier_normal)                             #1024 -> 1025
  lf4.bias[1] = -1.0
  #lf4 = Dense(W,[-5])                             #1024 -> 1025


  f = Chain(l1c1,l1c2,l1p1,l1b1,
            l2c1,l2c2,l2p1,l2b1,
            l3c1,l3c2,l3p1,l3b1,l3b2,
            l4c1,l4d1,l4c2,l4b1,l4d2,
            lf0,lf1,lfd1,lf2,lfd2,lf3,lfb1,lf4) # 5674946
end
f = instantiate_model()
##
loss_f(x,y) = logitbinarycrossentropy(f(x),y)
#accuracy2(x, y) = 1-mean(abs.(round.(sigmoid.(f(train_images))) .- train_labels'))
accuracy(x, y) = mean((round.(sigmoid.(f(x))) .== y))
f = gpu(f)

#@btime logitbinarycrossentropy(f(train_images),train_labels)
#loss_and_accuracy(train_images,train_labels)
#f(train_images)

function loss_and_accuracy(x,y)
  ŷ = f(x)
  logitbinarycrossentropy(ŷ,y), mean((round.(sigmoid.(ŷ))' .== y))
end

function loss_and_accuracy(dataloader)
  loss = 0; accuracy = 0; i = 0
  for (a,b) in dataloader
    a,b = gpu(a), gpu(b)
    ŷ = f(a)
    loss += logitbinarycrossentropy(ŷ,b)
    ŷ = round.(σ.(ŷ))
    accuracy += mean(ŷ .== b)
    i += 1
    #CUDA.unsafe_free!(a)
    a = nothing
    b = nothing
  end
  #dataloader = nothing
  loss / i, accuracy / i, i
end

##

first_execution = true
if first_execution
  opt = ADAM()
  history_loss = []
  history_accuracy = []
  iterations = []
  scheduler = ParameterSchedulers.Stateful((ParameterSchedulers.Inv(λ = 1e-4, γ = 0.05, p = 11)))
else
  @load "lastro_decayRateInv.bson" weights opt history_loss history_accuracy iterations scheduler
  Flux.loadparams!(f, weights)
end
f = gpu(f)
parameters = Flux.params(f) 

##Load data 593
(test_images, test_labels) = getobs(testData,1:nobs(testData))
test_images = preprocess_images!(test_images)
#test_labels = preprocess_labels!(test_labels)
test_dataloader = Flux.Data.DataLoader((test_images, test_labels), batchsize=593, shuffle=true,partial = true)


#(train_images, train_labels) = getobs(trainData,1:593)
#train_images = preprocess_images_gpu!(train_images)
#train_labels = gpu(train_labels)
##
#accuracy(train_images,train_labels)

#round.(σ.(y0)) .== train_labels

a,b, _ = loss_and_accuracy(test_dataloader)
println(a,b)
##

"""Trains model. trainData must be an ImageDataset, testData a gpu tuple,
nbatches and nepochs are number of batches and epochs per batch respectectively,
loss is the loss function, opt and parameters are the optimizer and its parameters from Flux.params,
and accuracy is the accuracy function.
This method loads a batch, trains nepochs times with it then loades the next.
A single run will train with the entire trainData one time. @epochs n for multiple iterations on the entire train dataset.
"""
function alternative_train!(trainData::ImageDataset, testDataloader, nbatches, nepochs, loss , opt, parameters,accuracy, history_loss, history_accuracy, iterations)
  batchSize = floor(Int,nobs(trainData)/nbatches)
  index(index,batch,batchSize=batchSize) = index + (batch-1)*batchSize
  i_base = 0
  length(iterations) > 0 ? i_base = iterations[end] : nothing
  #s = ParameterSchedulers.Stateful((Exp(λ = 5e-1, γ = 0.9)))
  t_loss, t_accuracy, _ = loss_and_accuracy(testDataloader)
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
    #train_labels = gpu(train_labels)
    train_labels = preprocess_labels_gpu!(train_labels)
    for j in 1:nepochs
      
      if Int((i-1)*nepochs+j+i_base) % 100 == 0
        testmode!(f,true)
        t_loss, t_accuracy, _ = loss_and_accuracy(testDataloader)
        push!(history_accuracy, t_accuracy)
        push!(history_loss, t_loss)
        push!(iterations, i+(j-1)*nbatches+i_base)
        println("\nBatch $i, iteration $j, accuracy: $t_accuracy, loss: $t_loss")
        trainmode!(f,true)
      end
      g = Flux.gradient(() -> loss(train_images, train_labels), parameters)
      Flux.update!(opt, parameters, g)
      ProgressMeter.next!(p)
    end
  end
end

"Same as alternative_train but with a sleep time at the end. The GPU must not burn."
function long_train(trainData::ImageDataset, testData, nbatches,
 nepochs, loss , opt, parameters,accuracy, 
 history_loss, history_accuracy, iterations, sleep_time::Int, scheduler = nothing)
 alternative_train!(trainData, testData, nbatches, nepochs,loss,
  opt,parameters,accuracy,history_loss, history_accuracy,iterations)
  println("sleeping $sleep_time seconds")
  sleep(sleep_time)
  if scheduler != nothing
	  @show opt.eta = ParameterSchedulers.next!(scheduler)
  end
  f[13].p = f[13].p*1.028
  f[15].p = f[15].p*1.028
  f[18].p = f[18].p*1.028
  f[21].p = f[21].p*1.028
  @show f[23].p = f[23].p*1.028
end
##


#opt.eta = ParameterSchedulers.next!(scheduler)

@epochs 1 long_train(trainData, test_dataloader, 160, 12,loss_f,
  opt,parameters,accuracy,history_loss, history_accuracy,iterations,10)

#a,b, _ = loss_and_accuracy((test_images, test_labels), 30)
##

#plot(log.(history_loss),label=:none); plot!(twinx(),history_accuracy, color = :orange, label=:none)
println(iterations[4:end],history_accuracy[4:end])
    
weights = collect(Flux.params(cpu(f)))
@save "lastro_decayRateInv.bson" weights opt history_loss history_accuracy iterations scheduler
#exit()
