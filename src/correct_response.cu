#include "opencv2/core/cuda.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"
#include "opencv2/core/cuda/common.hpp"

#define loadpix(addr, offset) *((const uchar *)(addr) + (offset))
#define storepix(val, addr, offset) *((float *)(addr) + (offset)) = val

template <uint8_t CN>
__global__ void correct_response(const cv::cuda::PtrStepb src,
                                 const cv::cuda::PtrStepf inv_resp,
                                 cv::cuda::PtrStepSzf dst)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < dst.cols && y < dst.rows)
    {
        const uchar *srciter = src.ptr(y) + x * CN;
        float *dstiter = dst.ptr(y) + x * CN;
        if constexpr (CN >= 1)
            storepix(*(inv_resp.ptr() + loadpix(srciter, 0)), dstiter, 0);
        if constexpr (CN >= 3)
        {
            storepix(*(inv_resp.ptr() + loadpix(srciter, 1)), dstiter, 1);
            storepix(*(inv_resp.ptr() + loadpix(srciter, 2)), dstiter, 2);
        }
        if constexpr (CN == 4)
            storepix(loadpix(srciter, 3), dstiter, 3);
    }
}

extern "C" void cuda_correctResponse(cv::InputArray _src, cv::InputArray _inv_resp, cv::OutputArray _dst, cv::cuda::Stream &_stream)
{
    cv::cuda::GpuMat src = _src.getGpuMat();
    cv::cuda::GpuMat inv_resp = _inv_resp.getGpuMat();
    cv::cuda::GpuMat dst = _dst.getGpuMat();

    const dim3 block(16, 16);
    const dim3 grid(cv::divUp(dst.cols, block.x), cv::divUp(dst.rows, block.y));

    const int cn = src.channels();
    cudaStream_t stream = cv::cuda::StreamAccessor::getStream(_stream);
    if (cn == 1)
        correct_response<1><<<grid, block, 0, stream>>>(src, inv_resp, dst);
    else if (cn == 3)
        correct_response<3><<<grid, block, 0, stream>>>(src, inv_resp, dst);
    else if (cn == 4)
        correct_response<4><<<grid, block, 0, stream>>>(src, inv_resp, dst);

    cudaSafeCall(cudaGetLastError());
    if (stream == 0)
        cudaSafeCall(cudaDeviceSynchronize());
}
