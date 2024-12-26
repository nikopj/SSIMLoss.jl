module SSIMLossCUDAExt

using SSIMLoss, CUDA, cuDNN

SSIMLoss.ssim_kernel(x::AnyCuArray{T, N}) where {T, N} = CuArray(ssim_kernel(T, N))

end
