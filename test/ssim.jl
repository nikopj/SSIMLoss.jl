# monarch_color_256 and fabio_color_256 testimages 
# used to obtain below numbers.
# true/false denote `assess_ssim(...; crop=true/false)`
const iqi_rgb_true = 0.1299260389807608
const iqi_gry_true = 0.13380159790218638
const iqi_rgb_false = 0.13683875886675542
const iqi_gry_false = 0.14181793989104552

@testset "IQI consistency" begin
    # color-image testing
    # ssim values for monarch-fabio
    @test SSIMLoss.SSIM_KERNEL == ImageQualityIndexes.SSIM_KERNEL.parent

    # get reference images
    imx_rgb = testimage("monarch_color_256")
    imy_rgb = testimage("fabio_color_256")
    imx_gry = Gray.(imx_rgb)
    imy_gry = Gray.(imy_rgb)
    x_rgb = permutedims(channelview(imx_rgb), (2, 3, 1)) .|> Float64 |> unsqueeze(dims=4)
    y_rgb = permutedims(channelview(imy_rgb), (2, 3, 1)) .|> Float64 |> unsqueeze(dims=4)
    x_gry = imx_gry .|> Float64 |> unsqueeze(dims=3) |> unsqueeze(dims=4)
    y_gry = imy_gry .|> Float64 |> unsqueeze(dims=3) |> unsqueeze(dims=4)

    # 8 tests enumerating rgb/gray, crop/nocrop, iqi/flux vs. ref
    for (ssim_iqi, crop) in 
        zip(((iqi_rgb_true, iqi_gry_true), (iqi_rgb_false, iqi_gry_false)), (true, false))

        for (imx, imy, x, y, ssim_ref) in 
            zip((imx_rgb, imx_gry), (imy_rgb, imy_gry), (x_rgb, x_gry), (y_rgb, y_gry), ssim_iqi)

            color = eltype(imx) <: RGB ? "RGB" : "Gray"
            @testset "crop=$crop, color=$color" begin
                # make sure IQI is same
                @test assess_ssim(imx, imy; crop=crop) ≈ ssim_ref
                # test flux against IQI 
                @test ssim(x, y; crop=crop) ≈ ssim_ref atol=1e-6
            end
        end
    end
end

@testset "$T, ndims=$N" for T in (Float64, Float32, Float16), N=1:3
    x = rand(T, 16*ones(Int, N)..., 2, 2)
    y = rand(T, 16*ones(Int, N)..., 2, 2)

    @testset "sanity check" begin
        @test ssim(x, x) ≈ 1
        @test ssim_loss(x, x) ≈ 0
        @test ssim_loss_fast(x, x) ≈ 0
    end

    @testset "$f" for f in (ssim, ssim_loss, ssim_loss_fast)
        @testset "no spurious promotions" begin
            fwd, back = pullback(f, x, y)
            @test fwd isa T
            @test eltype(back(one(T))[1]) == T
        end
    end
end
