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

tart::kernel_ptr getCopy(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth);
tart::kernel_ptr getCopyFast(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth);

tart::kernel_ptr getAxpy(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth);
tart::kernel_ptr getAxpyFaster(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth);
tart::kernel_ptr getAxpyFastest(const tart::device_ptr& device, const tart::DType& dtype, const uint32_t vectorWidth);




} // namespace clblast

#endif
