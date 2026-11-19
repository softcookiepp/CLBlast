#ifndef INVERT_DIAGONAL_BLOCKS_PARAMETERS
#define INVERT_DIAGONAL_BLOCKS_PARAMETERS

#ifdef INTERNAL_BLOCK_SIZE
	#undef INTERNAL_BLOCK_SIZE //16		 
#endif
layout(constant_id = 0) const int INTERNAL_BLOCK_SIZE = 16; // Internal block size of the invert kernel

#ifdef LOCALPAD
	#undef LOCALPAD //0								 
#endif
layout(constant_id = 1) const int LOCALPAD = 0; // Padding in the x-dimension of the local memory to avoid bank conflicts

// keep this as is, it is not a tunable parameter in the database
#ifndef LOCALX
	#define LOCALX (16 + LOCALPAD)		 // Local memory size in x-dimension of TripleMatMul kernels
#endif
// and this one too. Really not sure what can even be done here
#ifndef LOCALY
	#define LOCALY 16									// Local memory size in y-dimension of TripleMatMul kernels
#endif

#ifdef TMMWGSX
	#undef TMMWGSX //4									
#endif
layout(constant_id = 2) const int TMMWGSX = 4; // Work-group size in x-dimension of TripleMatMul kernels

#ifdef TMMWGSY
	#undef TMMWGSY
#endif
layout(constant_id = 3) const int TMMWGSY = 4; // Work-group size in y-dimension of TripleMatMul kernels
layout(constant_id = 4) const int SIZE = 64; // Before, there was a separate kernel for each of these for some reason
layout(constant_id = 5) const bool UPPER = true; // and a separate one for this, too.

#endif //INVERT_DIAGONAL_BLOCKS_PARAMETERS
