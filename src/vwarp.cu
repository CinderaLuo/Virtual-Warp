#include"cuda_runtime.h"
#define MAX_THREADS_PER_BLOCK 512
#define VWARP_WIDTH 32
#define BATCH_SIZE 32
const int DEAFAULT_THREADS_PER_BLOCK=256;
const int MAX_BLOCK_PER_DIMENSION=65535;

/*
*Global linear thread index
*/
#define THREAD_GLOBAL_INDEX (threadIdx.x+blockDim.x  \
                             *(gridDim.x*blockIdx.y+blockIdx.x))
/*
*Global linear thread-block index
*/
#define BLOCK_GLOBAL_INDEX (gridDim.x*blockIdx.y+blockIdx.x)
#define THREAD_BLOCK_INDEX (threadIdx.x)

/*
*The number of batches of vertices a workload has
*/
inline __device__ __host__
int vwarp_batch_count(const int vertex_count,const int vwarp_batch)
{
	return ((vertex_count/vwarp_batch)+
		    (vertex_count%vwarp_batch==0?0:1));
}
/*
*The number of threads needed to process a given number of vertices
*/
inline __device__ __host__
int vwarp_thread_count(const int vertex_count,const int vwarp_width,const int vwarp_batch)
{
	return vwarp_width * vwarp_batch_count(vertex_count,vwarp_batch);
}

inline __device__ __host__
int vwarp_warps_per_block(int vwarp_width,int threads_per_block=MAX_THREADS_PER_BLOCK)
{
	return (threads_per_block/vwarp_width);
}
/*
*The id of the first vertex in the batch of vertices to be processed by a thread-block
*/
inline __device__
int vwarp_block_start_vertex(const int vwarp_width,const int vwarp_batch,int threads_per_block=MAX_THREADS_PER_BLOCK)
{
	return(vwarp_warps_per_block(vwarp_width,threads_per_block)*
		  BLOCK_GLOBAL_INDEX*vwarp_batch);
}
inline __device__ __host__
int vwarp_block_max_batch_size(int vwarp_width,int vwarp_batch,int threads_per_block=MAX_THREADS_PER_BLOCK)
{
	return (vwarp_warps_per_block(vwarp_width,threads_per_block)*vwarp_batch);
}
/*
*The amount of work assigned to a specific thread_block
*/
inline __device__
int vwarp_block_batch_size(const int vertex_count,const int vwarp_width,
	                       const int vwarp_batch,
	                       int threads_per_block=MAX_THREADS_PER_BLOCK)
{
	int start_vertex=vwarp_block_start_vertex(vwarp_width,vwarp_batch,threads_per_block);
	int last_vertex=start_vertex+
	vwarp_block_max_batch_size(vwarp_width,vwarp_batch,threads_per_block);
	return (last_vertex>vertex_count?(vertex_count - start_vertex):
		vwarp_block_max_batch_size(vwarp_width,vwarp_batch,threads_per_block));
}
inline __device__
int vwarp_warp_index(const int vwarp_width)
{
	return THREAD_BLOCK_INDEX/vwarp_width;
}
inline __device__
int vwarp_warp_start_vertex(const int vwarp_width,const int vwarp_batch)
{
	return vwarp_warp_index(vwarp_width)*vwarp_batch;
}
/*
*The size of the batch of work assigned to a specific visual warp
*/
inline __device__
int vwarp_warp_batch_size(const int vertex_count,
	const int vwarp_width,
	const int vwarp_batch,
	int threads_per_block=MAX_THREADS_PER_BLOCK
	)
{
	int block_batch_size=
	vwarp_block_batch_size(vertex_count,vwarp_width,vwarp_batch,threads_per_block);
	int start_vertex=vwarp_warp_start_vertex(vwarp_width,vwarp_batch);
	int last_vertex=start_vertex+vwarp_batch;
	return (last_vertex>=block_batch_size?(block_batch_size - start_vertex):vwarp_batch);
}
/*
*The index of a thread in the virtual warp which it belongs to
*/
inline __device__
int vwarp_thread_index (const int vwarp_width)
{
	return THREAD_BLOCK_INDEX%vwarp_width;
}

inline int __host__ __device__
kernel_configure (int thread_count,dim3 &blocks,int threads_per_block=DEAFAULT_THREADS_PER_BLOCK)
{ 
	if(threads_per_block>MAX_THREADS_PER_BLOCK)
		return 0;
	int blocks_left =((thread_count%threads_per_block==0)?
		              thread_count/threads_per_block:(thread_count/threads_per_block)+1);
	int x_blcoks=(blocks_left>=MAX_BLOCK_PER_DIMENSION)?MAX_BLOCK_PER_DIMENSION:blocks_left;
	blocks_left=((blocks_left%x_blcoks==0)?(blocks_left)/x_blcoks:(blocks_left)/x_blcoks+1);
	if (blocks_left>MAX_BLOCK_PER_DIMENSION)
	{
		return 0;
	}
    dim3 my_blocks(x_blcoks,blocks_left);
    blocks=my_blocks;
    return 1;
}
