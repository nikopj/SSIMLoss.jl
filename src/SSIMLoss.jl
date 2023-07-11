module SSIMLoss

using NNlib, MLUtils, Statistics
using ChainRulesCore

include("utils.jl")
export ssim_kernel

"""
    ssim(x, y, kernel=ssim_kernel(x); peakval=1, crop=true, dims=:)
                                    
Return the [structural similarity index
measure](https://en.wikipedia.org/wiki/Structural_similarity) (SSIM) between
two signals. SSIM is computed via the mean of a sliding window of
statistics computed between the two signals. By default, the sliding window is
a Gaussian with side-length 11 in each signal dimension and σ=1.5. `crop=false` will pad `x` and `y` 
such that the sliding window computes statistics centered at every pixel of the input (via same-size convolution). 
`ssim` computes statistics independently over channel and batch dimensions.
`x` and `y` may be 3D/4D/5D tensors with channel and batch-dimensions.

`peakval=1` is the standard for image comparisons, but in practice should be
set to the maximum value of your signal type. 

`dims` determines which dimensions to average the computed statistics over. If
`dims=1:ndims(x)-1`, SSIM will be computed for each batch-element separately.

The results of `ssim` are matched against those of
[ImageQualityIndexes](https://github.com/JuliaImages/ImageQualityIndexes.jl)
for grayscale and RGB images (i.e. x, y both of size (N1, N2, 1, B) and (N1, N2, 3, B) for grayscale and color images, resp.).

See also [`ssim_loss`](@ref), [`ssim_loss_fast`](@ref).
"""
function ssim(x::AbstractArray{T,N}, y::AbstractArray{T,N}, kernel=ssim_kernel(x); peakval=T(1.0), crop=true, dims=:) where {T,N}
  _check_sizes(x, y)

  # apply same kernel on each channel dimension separately via groups=in_channels
  groups = size(x, N-1)
  kernel = repeat(kernel, ones(Int, N-1)..., groups)

  # constants to avoid division by zero
  SSIM_K = (0.01, 0.03) 
  C₁, C₂ = @. T(peakval * SSIM_K)^2

  # crop==true -> valid-sized conv (do nothing), 
  # otherwise, pad for same-sized conv
  if !crop
    # from Flux.jl:src/layers/conv.jl (calc_padding)
    padding = Tuple(mapfoldl(i -> [cld(i, 2), fld(i,2)], vcat, size(kernel)[1:N-2] .- 1))
    x = pad_symmetric(x, padding) 
    y = pad_symmetric(y, padding) 
  end

  μx  = conv(x, kernel; groups=groups)
  μy  = conv(y, kernel; groups=groups)
  μx² = μx.^2
  μy² = μy.^2
  μxy = μx.*μy
  σx² = conv(x.^2, kernel; groups=groups) .- μx²
  σy² = conv(y.^2, kernel; groups=groups) .- μy²
  σxy = conv(x.*y, kernel; groups=groups) .- μxy

  ssim_map = @. (2μxy + C₁)*(2σxy + C₂)/((μx² + μy² + C₁)*(σx² + σy² + C₂))
  return mean(ssim_map, dims=dims)
end

"""
    ssim_loss(x, y, kernel=ssim_kernel(x); peakval=1, crop=true, dims=:)

Computes `1 - ssim(x, y)`, suitable for use as a loss function with gradient descent.
For faster training, it is recommended to store a kernel and reuse it, ex.,
```julia
kernel = ssim_kernel(Float32, 2) |> gpu
# or alternatively for faster computation
# kernel = ones(Float32, 5, 5, 1, num_channels) |> gpu

for (x, y) in dataloader
  x, y = (x, y) .|> gpu
  grads = gradient(model) do m
    x̂ = m(y)
    ssim_loss(x, x̂, kernel)
  end
  # update the model ...
end
```
See [`ssim`](@ref) for a detailed description of SSIM and the above arguments.
See also [`ssim_loss_fast`](@ref).
"""
ssim_loss(x::AbstractArray{T}, args...; kws...) where T = one(T) - ssim(x, args...; kws...)

"""
    ssim_loss_fast(x, y; kernel_length=5, peakval=1, crop=true, dims=:)

Computes `ssim_loss` with an averaging kernel instead of a large Gaussian
kernel for faster computation. `kernel_length` specifies the averaging kernel
side-length in each signal dimension of x, y.  See [`ssim`](@ref) for a
detailed description of SSIM and the above arguments. 

See also [`ssim_loss`](@ref).
"""
function ssim_loss_fast(x::AbstractArray{T, N}, y::AbstractArray{T, N}; kernel_length=5, kws...) where {T, N}
  kernel = ones_like(x, (kernel_length*ones(Int, N-2)..., 1, 1))
  kernel = kernel ./ sum(kernel)
  return ssim_loss(x, y, kernel; kws...)
end

export ssim, ssim_loss, ssim_loss_fast

end
