using Test
using SSIMLoss, CUDA, cuDNN

using Zygote
using Random

using MLUtils
using Images, TestImages, ImageQualityIndexes

Random.seed!(0)

@testset "SSIMLoss" begin
    include("ssim.jl")

    @testset "CUDA" begin
        if CUDA.functional()
            include("cuda.jl")
        else
            @warn "CUDA unavailable, not testing GPU support"
        end
    end
end
