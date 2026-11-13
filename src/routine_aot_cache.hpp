#ifndef ROUTINE_AOT_CACHE_HPP
#define ROUTINE_AOT_CACHE_HPP

#include "clblast_vk.h"
#include <cstdint>
#include "tart.hpp"

namespace clblast
{

tart::kernel_ptr getAmax(const tart::device_ptr& device, const tart::DType& dtype);
tart::kernel_ptr getAmaxEpilogue(const tart::device_ptr& device, const tart::DType& dtype);

tart::kernel_ptr getAsum(const tart::device_ptr& device, const tart::DType& dtype);
tart::kernel_ptr getAsumEpilogue(const tart::device_ptr& device, const tart::DType& dtype);

tart::kernel_ptr getDot(const tart::device_ptr& device, const tart::DType& dtype);
tart::kernel_ptr getDotEpilogue(const tart::device_ptr& device, const tart::DType& dtype);

tart::kernel_ptr getNrm2(const tart::device_ptr& device, const tart::DType& dtype);
tart::kernel_ptr getNrm2Epilogue(const tart::device_ptr& device, const tart::DType& dtype);

tart::kernel_ptr getCopy(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth);
tart::kernel_ptr getCopyFast(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth);

tart::kernel_ptr getScal(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth);
tart::kernel_ptr getScalFast(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth);

tart::kernel_ptr getSwap(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth);
tart::kernel_ptr getSwapFast(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth);

tart::kernel_ptr getAxpy(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth);
tart::kernel_ptr getAxpyFaster(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth);
tart::kernel_ptr getAxpyFastest(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth);

tart::kernel_ptr getGemv(const tart::device_ptr& device, const tart::DType& dtype);
tart::kernel_ptr getGemvFast(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth);
tart::kernel_ptr getGemvFastRot(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth);

tart::kernel_ptr getGer(const tart::device_ptr& device, const tart::DType& dtype);
tart::kernel_ptr getHer(const tart::device_ptr& device, const tart::DType& dtype);
tart::kernel_ptr getHer2(const tart::device_ptr& device, const tart::DType& dtype);
tart::kernel_ptr getTrsvForward(const tart::device_ptr& device, const tart::DType& dtype);
tart::kernel_ptr getTrsvBackward(const tart::device_ptr& device, const tart::DType& dtype);
tart::kernel_ptr getTrsvFillVector(const tart::device_ptr& device, const tart::DType& dtype);

tart::kernel_ptr getGemm(const tart::device_ptr& device, const tart::DType& dtype, const bool subgroupSupport, const uint32_t vwm, const uint32_t vwn);
tart::kernel_ptr getGemmDirect(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vwm, const uint32_t vwn);

} // namespace clblast

#endif
