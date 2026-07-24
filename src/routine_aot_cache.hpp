#ifndef ROUTINE_AOT_CACHE_HPP
#define ROUTINE_AOT_CACHE_HPP

#include "clblast_vk.h"
#include <cstdint>



namespace clblast
{

// A per-device object holding instances of every routine.
// The aim is to eliminate the overhead of all Routine object initialization that happens
// with every single CLBlast API call.
class AllRoutines
{
	tart::device_ref mDevice;
	
public:
	AllRoutines(const tart::device_ptr& device);
	
	
};

} // namespace clblast

#endif
