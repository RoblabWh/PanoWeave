#define loadpix(addr, offset)  *((__global const uchar*)(addr) + (offset))
#define storepix(val, addr, offset)  *((__global float*)(addr) + (offset)) = val

__kernel void correct_response(__global const uchar * srcptr, int src_step, int src_offset,
                               __global const uchar * inv_resp_ptr, int inv_resp_step, int inv_resp_offset,
                               __global uchar * dstptr, int dst_step, int dst_offset, int dst_rows, int dst_cols) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x < dst_cols && y < dst_rows) {
      __global const uchar *srciter = srcptr + mad24(y, src_step, mad24(x, CN, src_offset));
      __global uchar *dstiter = dstptr + mad24(y, dst_step, mad24(x, sizeof(float) * CN, dst_offset));
      #if CN >= 1
        storepix(*(__global float*)(inv_resp_ptr + mad24(loadpix(srciter, 0), sizeof(float), inv_resp_offset)), dstiter, 0);
      #endif
      #if CN >= 3
        storepix(*(__global float*)(inv_resp_ptr + mad24(loadpix(srciter, 1), sizeof(float), inv_resp_offset)), dstiter, 1);
        storepix(*(__global float*)(inv_resp_ptr + mad24(loadpix(srciter, 2), sizeof(float), inv_resp_offset)), dstiter, 2);
      #endif
      #if CN == 4
        storepix(loadpix(srciter, 3), dstiter, 3);
      #endif
    }
}
