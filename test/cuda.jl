CUDA.allowscalar(false)

@testset "type $T, ndims=$N, $loss" for T=(Float64, Float32, Float16), N=1:3, loss in (ssim, ssim_loss, ssim_loss_fast)
    # see https://github.com/FluxML/NNlib.jl/issues/505
    # Float16 conv is broken for 5D tensors
    if T==Float16 && (N==3 || loss in (ssim, ssim_loss))
        continue
    end

    x_cpu = rand(T, 16*ones(Int, N)..., 2, 2)
    y_cpu = rand(T, 16*ones(Int, N)..., 2, 2)
    x_gpu = cu(x_cpu)
    y_gpu = cu(y_cpu)

    @testset "sanity check" begin
        @test ssim(x_gpu, x_gpu) ≈ 1
        @test ssim_loss(x_gpu, x_gpu) ≈ 0
        @test ssim_loss_fast(x_gpu, x_gpu) ≈ 0
    end

    @testset "cpu == gpu" loss(x_cpu, y_cpu) ≈ loss(x_gpu, y_gpu) 

    @testset "grad cpu == gpu" begin
        out_cpu, back_cpu = pullback((x, y) -> loss(x, y), x_cpu, y_cpu)
        c = randn(T)
        gs_cpu = back_cpu(c)

        out_gpu, back_gpu = pullback((x, y) -> loss(x, y), x_gpu, y_gpu)
        gs_gpu = back_gpu(c)

        @test collect(out_cpu) ≈ collect(out_gpu)
        for (g_cpu, g_gpu) in zip(gs_cpu, gs_gpu)
            @test collect(g_cpu) ≈ collect(g_gpu)
        end
    end
end
