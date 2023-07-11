# Gaussian kernel std=1.5, length=11
const SSIM_KERNEL = 
    [0.00102838008447911,
    0.007598758135239185,
    0.03600077212843083,
    0.10936068950970002,
    0.2130055377112537,
    0.26601172486179436,
    0.2130055377112537,
    0.10936068950970002,
    0.03600077212843083,
    0.007598758135239185,
    0.00102838008447911]

"""
    ssim_kernel(T, N)

Return Gaussian kernel with σ=1.5 and side-length 11 for use in [`ssim`](@ref).
Returned kernel will be `N-2` dimensional of type `T`.
"""
function ssim_kernel(T::Type, N::Integer)
    if N-2 == 1
        kernel = SSIM_KERNEL
    elseif N-2 == 2
        kernel = SSIM_KERNEL*SSIM_KERNEL' 
    elseif N-2 == 3
        ks = length(SSIM_KERNEL)
        kernel = reshape(SSIM_KERNEL*SSIM_KERNEL', 1, ks, ks).*SSIM_KERNEL
    else
        throw("SSIM is only implemented for 3D/4D/5D inputs, dimension=$N provided.")
    end
    return reshape(T.(kernel), size(kernel)..., 1, 1)
end
ChainRulesCore.@non_differentiable ssim_kernel(T::Any, N::Any)

"""
    ssim_kernel(x::AbstractArray{T, N}) where {T, N}

Return Gaussian kernel with σ=1.5 and side-length 11 for use in [`ssim`](@ref). 
Returned array will be on the same device as `x`.
"""
ssim_kernel(x::Array{T, N}) where {T, N} = ssim_kernel(T, N)
ChainRulesCore.@non_differentiable ssim_kernel(x::Any)

function _check_sizes(x::AbstractArray, y::AbstractArray)
    for d in 1:max(ndims(x), ndims(y)) 
        size(x,d) == size(y,d) || throw(DimensionMismatch(
          "loss function expects size(ŷ) = $(size(ŷ)) to match size(y) = $(size(y))"
        ))
    end
end
_check_sizes(ŷ, y) = nothing  # pass-through, for constant label e.g. y = 1
ChainRulesCore.@non_differentiable _check_sizes(ŷ::Any, y::Any)
