#ENV["JULIA_PROJECT"]="/home/hipparcos/pre_PhD/phd_pretesis/LensFindingChallenge1Julia/"
using Plots
using AstroImages

const data_dir = "Data/"

show_image(name::String) =  plot(AstroImage(name))
f = "Data/imageEUC_VIS-100004.fits"
img = AstroImage(f)
plot(img)

show_image(data_dir*"imageEUC_VIS-100004.fits")
file_list = readdir(data_dir)

for file in file_list[1:100]
    println(file)
end

##
