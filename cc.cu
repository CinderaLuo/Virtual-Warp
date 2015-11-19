#include "cuda_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "src/frog.cuh"
#include "src/vwarp.cu"

// print info about CC values
void report_cc_values(const int * const values, int n) {
	int * c = (int *)calloc(n, sizeof(int));
	int cc = 0;
	for (int i = 0; i < n; i++) {
		int r = values[i];
		if (c[r] == 0) cc++;
		c[r]++;
	}
	printf("Number of Connected Components: %d\n", cc);
	int k = 0;
	printf("\tID\tRoot\tN\n");
	for (int i = 0; i < n; i++) {
		if (c[i] != 0)
			printf("\t%d\t%d\t%d\n", k++, i, c[i]);
		if (k > 20) {
			printf("\t...\n");
			break;
		}
	}
	free(c);
}

// check if arrays v1 & v2 have the same first n elements (no boundary check)
static void check_values(const int * const v1, const int * const v2, int n) {
	for (int i = 0; i < n; i++) {
		if (v1[i] != v2[i]) {
			printf("Check Fail\n");
			return;
		}
	}
	printf("Check PASS\n");
}
// MFset function - Reset                                                                             
static void Reset(int s[], int n) {
	for (int i = 0; i < n; i++) s[i] = i;
}

//simple cc without tracebacking
static void cc_on_cpu(
		const int vertex_num,
		const int * const vertex_begin,
		const int * const edge_dest,
		int * const values
		) {
	timer_start();
	// Initializing values
	Reset(values, vertex_num);
	int flag=1;
	int step=0;
	// MFset calculating
	while(flag)
	{
		flag=0;
		for (int i = 0; i < vertex_num; i++)
		{
			int new_label=values[i];
			for (int k = vertex_begin[i]; k < vertex_begin[i + 1]; k++)
			{
				int old_label=values[edge_dest[k]];
				if(old_label>new_label)
				{
					values[edge_dest[k]]=new_label;
					flag=1;
				}
			}
		}
		step++;
	}
	printf("\t\t%.2f\tCC on CPU\tstep=%d\n", timer_stop(),step);
}

// CC Kernel on edges with inner loops
static __global__ void kernel_edge_loop (
		int const edge_num,
		const int * const edge_src,
		const int * const edge_dest,
		int * const values,
		int * const continue_flag
		) {
	// total thread number & thread index of this thread
	int n = blockDim.x * gridDim.x;
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	// continue flag for each thread
	int flag = 0;
	// proceeding loop
	for (int i = index; i < edge_num; i += n) {
		int src = edge_src[i];
		int dest = edge_dest[i];
		if (values[src] < values[dest]) {
			// combined to the smaller Component ID
				values[dest] = values[src];
				flag=1;
		}
	}
	if (flag == 1) *continue_flag = 1;
}

// CC algorithm on graph g, not partitioned, run on edges with inner loop
static void gpu_cc_edge_loop(const Graph * const g, int * const values) {
	Auto_Utility();
	timer_start();
	int vertex_num = g->vertex_num;
	int edge_num = g->edge_num;
	// GPU buffer
	for (int i = 0; i < vertex_num; i++) values[i] = i;
	CudaBufferCopy(int, dev_edge_src, edge_num, g->edge_src);
	CudaBufferCopy(int, dev_edge_dest, edge_num, g->edge_dest);
	CudaBufferCopy(int, dev_value, vertex_num, values);
	CudaBufferZero(int, dev_continue_flag, 1);
	// settings
	int bn = 204;
	int tn = 128;
	//luo
	tn=256;
	bn=(edge_num+tn-1)/tn;
	int flag = 0;
	int step = 0;
	float execTime = 0.0;
	// Main Loop
	do {
		// Clear Flag
		CudaMemset(dev_continue_flag, 0, sizeof(int));
		CudaTimerBegin();
		kernel_edge_loop<<<bn, tn>>>(
				edge_num,
				dev_edge_src,
				dev_edge_dest,
				dev_value,
				dev_continue_flag
				);
		execTime += CudaTimerEnd();
		// Copy Back Flag
		CudaMemcpyD2H(&flag, dev_continue_flag, sizeof(int));
		step++;
	} while(flag != 0 && step < 100);
	// Copy Back Values
	CudaMemcpyD2H(values, dev_value, vertex_num * sizeof(int));
	printf("\t%.2f\t%.2f\tcc_edge_loop\tstep=%d\t",
			execTime, timer_stop(), step);
}

// CC algorithm on graph g, partitioned, run on edges with inner loop
static void gpu_cc_edge_part_loop(
		const Graph * const * const g,
		const struct part_table * const t,
		int * const values
		) {
	Auto_Utility();
	timer_start();
	int part_num = t->part_num;
	// GPU buffer
	for (int i = 0; i < t->vertex_num; i++) values[i] = i;
	int ** dev_edge_src = (int **) Calloc(part_num, sizeof(int *));
	int ** dev_edge_dest = (int **) Calloc(part_num, sizeof(int *));
	for (int i = 0; i < part_num; i++) {
		int size = g[i]->edge_num * sizeof(int);
		CudaBufferFill(dev_edge_src[i], size, g[i]->edge_src);
		CudaBufferFill(dev_edge_dest[i], size, g[i]->edge_dest);
	}
	CudaBufferCopy(int, dev_value, t->vertex_num, values);
	CudaBufferZero(int, dev_continue_flag, 1);
	// settings
	int bn = 204;
	int tn = 128;
	int flag = 0;
	int step = 0;
	tn=256;

	float execTime = 0.0;
	// Main Loop
	do {
		// Clear Flag
		CudaMemset(dev_continue_flag, 0, sizeof(int));
		// Launch Kernel for this Iteration
		for (int i = 0; i < part_num; i++) {
			int j=g[i]->edge_num%tn==0?0:1;
			bn=g[i]->edge_num/tn+j;
			CudaTimerBegin();
			kernel_edge_loop<<<bn, tn>>>
				(
				 g[i]->edge_num,
				 dev_edge_src[i],
				 dev_edge_dest[i],
				 dev_value,
				 dev_continue_flag
				);
			execTime += CudaTimerEnd();
		}
		// Copy Back Flag
		CudaMemcpyD2H(&flag, dev_continue_flag, sizeof(int));
		step++;
	} while(flag != 0 && step < 100);
	// Copy Back Values
	CudaMemcpyD2H(values, dev_value, t->vertex_num * sizeof(int));
	printf("\t%.2f\t%.2f\tpart_cc_edge_loop\tstep=%d\t",
			execTime, timer_stop(), step);
}

// CC Kernel on vertices without inner loops
static __global__ void kernel_vertex (
		int const vertex_num,
		const int * const vertex_begin,
		const int * const edge_dest,
		int * const values,
		int * const continue_flag
		) {
	// thread index of this thread
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	// proceed
	if (i < vertex_num) {
		int new_value = values[i];
		int flag = 0;
		// find the best new_value (smallest)
		for (int e = vertex_begin[i]; e < vertex_begin[i + 1]; e++) {
			int dest_value = values[edge_dest[e]];
			if (dest_value > new_value)
			{
				values[edge_dest[e]]=new_value;
				flag=1;
			}
		}
		// update values
		if (flag) {
			*continue_flag = 1;
		}	
	}
}

// CC algorithm on graph g, not partitioned, run on vertices without inner loop
static void gpu_cc_vertex(const Graph * const g, int * const values) {
	Auto_Utility();
	timer_start();
	int vertex_num = g->vertex_num;
	int edge_num = g->edge_num;
	// GPU buffer
	for (int i = 0; i < vertex_num; i++) values[i] = i;
	CudaBufferCopy(int, dev_vertex_begin, vertex_num, g->vertex_begin);
	CudaBufferCopy(int, dev_edge_dest, edge_num, g->edge_dest);
	CudaBufferCopy(int, dev_value, vertex_num, values);
	CudaBufferZero(int, dev_continue_flag, 1);
	// settings
	int bn = (vertex_num + 255) / 256;
	int tn = 256;
	int flag = 0;
	int step = 0;
	float execTime = 0.0;
	// Main Loop
	do {
		// Clear Flag
		CudaMemset(dev_continue_flag, 0, sizeof(int));
		CudaTimerBegin();
		kernel_vertex<<<bn, tn>>>(
				vertex_num,
				dev_vertex_begin,
				dev_edge_dest,
				dev_value,
				dev_continue_flag
				);
		execTime += CudaTimerEnd();
		// Copy Back Flag
		CudaMemcpyD2H(&flag, dev_continue_flag, sizeof(int));
		step++;
	} while(flag != 0 && step < 100);
	// Copy Back Values
	CudaMemcpyD2H(values, dev_value, vertex_num * sizeof(int));
	printf("\t%.2f\t%.2f\tcc_vertex\tstep=%d\t",
			execTime, timer_stop(), step);
}
//virtual warp similar to Totem
template<int VWARP_WIDTH1,int VWARP_BATCH1>
static  __global__ void kernel_vertex_part (
		int const vertex_num,
		const int * const vertex_id,
		const int * const vertex_begin,
		const int * const edge_dest,
		int * const values,
		int * const continue_flag
		) {
	if(THREAD_GLOBAL_INDEX>=
			vwarp_thread_count(vertex_num,VWARP_WIDTH1,VWARP_BATCH1)){return ;}
	__shared__ bool finish_block;
	finish_block=true;
	__syncthreads();

	int start_vertex=vwarp_block_start_vertex(VWARP_WIDTH1,VWARP_BATCH1)+
		vwarp_warp_start_vertex(VWARP_WIDTH1,VWARP_BATCH1);
	int end_vertex=start_vertex+
		vwarp_warp_batch_size(vertex_num,VWARP_WIDTH1,VWARP_BATCH1);
	int warp_offset=vwarp_thread_index(VWARP_WIDTH1);
	// proceed
	for(int i=start_vertex;i<end_vertex;i++) {
		int id = vertex_id[i];
		int new_value = values[id];
		// find the best new_value (smallest)
		const int nbr_count=vertex_begin[i+1]-vertex_begin[i];
		const int *edge=edge_dest+vertex_begin[i];
		for (int e =warp_offset; e < nbr_count; e+=VWARP_WIDTH1)
		{
			int nbr=edge[e];
			int dest_value = values[nbr];
			if (dest_value > new_value) 
			{
				values[nbr]=new_value;
				finish_block=false;
			}
		}
	}
	//__syncthreads();
	//if(!finish_block&&THREAD_GLOBAL_INDEX==0) *continue_flag=1;
	if(!finish_block) *continue_flag=1;
}
// virtual warp by myself 
static __global__ void kernel_vertex_part1 (
		int const vertex_num,
		const int * const vertex_id,
		const int * const vertex_begin,
		const int * const edge_dest,
		int * const values,
		int * const continue_flag
		) {
#define VWARP_SIZE 2
	// thread index of this thread
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int n=blockDim.x*gridDim.x;
	int vwarp_num = n / VWARP_SIZE;
	int vwarp_id = index / VWARP_SIZE;
	int vwarp_tid = index % VWARP_SIZE;
	// offset of each vwarp for cache indexing (vwarp_offset + vwarp_tid == threadIdx.x)
	int vwarp_offset = threadIdx.x / VWARP_SIZE * VWARP_SIZE;
	// proceed
	for(int i=vwarp_id;i<vertex_num;i+=vwarp_num) {
		int id = vertex_id[i];
		int new_value = values[id];
		int flag = 0;
		// find the best new_value (smallest)
		for (int e = vertex_begin[i]+vwarp_tid; e < vertex_begin[i + 1]; e+=VWARP_SIZE) {
			int dest_value = values[edge_dest[e]];
			if (dest_value > new_value)
			{			
				flag=1;
				values[edge_dest[e]]=new_value;
			}
		}
		// update values
		if (flag) {
			*continue_flag = 1;
		}
	}
}

// BFS algorithm on graph g, partitioned, run on vertices without inner loop
static void gpu_cc_vertex_part(
		const Graph * const * const g,
		const struct part_table * const t,
		int * const values
		) {
	Auto_Utility();
	timer_start();
	int part_num = t->part_num;
	// GPU buffer
	int ** dev_vertex_begin = (int **) Calloc(part_num, sizeof(int *));
	int ** dev_edge_dest = (int **) Calloc(part_num, sizeof(int *));
	int ** dev_vertex_id = (int **) Calloc(part_num, sizeof(int *));
	for (int i = 0; i < part_num; i++) {
		int size = (g[i]->vertex_num + 1) * sizeof(int);
		CudaBufferFill(dev_vertex_begin[i], size, g[i]->vertex_begin);
		size = g[i]->edge_num * sizeof(int);
		CudaBufferFill(dev_edge_dest[i], size, g[i]->edge_dest);
		size = g[i]->vertex_num * sizeof(int);
		CudaBufferFill(dev_vertex_id[i], size, t->part_vertex[i]);
	}
	for (int i = 0; i < t->vertex_num; i++) values[i] = i;
	CudaBufferCopy(int, dev_value, t->vertex_num, values);
	CudaBufferZero(int, dev_continue_flag, 1);
	// settings
	int flag = 0;
	int step = 0;
	float execTime = 0.0;

	//luo
	const int threads=MAX_THREADS_PER_BLOCK;
	dim3 blocks;
	// Main Loop
	do {
		// Clear Flag
		CudaMemset(dev_continue_flag, 0, sizeof(int));
		// Launch Kernel for this Iteration
		for (int i = 0; i < part_num; i++) {
			kernel_configure(vwarp_thread_count(g[i]->vertex_num,VWARP_WIDTH,BATCH_SIZE),
					blocks,threads);
			CudaTimerBegin();
			// kernel_vertex_part1<<<blocks,threads>>>	 
			kernel_vertex_part<VWARP_WIDTH,BATCH_SIZE><<<blocks,threads>>>
				(
				 g[i]->vertex_num,
				 dev_vertex_id[i],
				 dev_vertex_begin[i],
				 dev_edge_dest[i],
				 dev_value,
				 dev_continue_flag
				);
			execTime += CudaTimerEnd();
		}
		// Copy Back Flag
		CudaMemcpyD2H(&flag, dev_continue_flag, sizeof(int));
		step++;
	} while(flag != 0 && step < 100);
	// Copy Back Values
	CudaMemcpyD2H(values, dev_value, t->vertex_num * sizeof(int));
	printf("\n");
	printf("\t%.2f\t%.2f\tpart_cc_vertex\tstep=%d\t",
			execTime, timer_stop(), step);
}
// BFS algorithm on graph g, partitioned, run on vertices without inner loop
static void gpu_cc_vertex_part1(
		const Graph * const * const g,
		const struct part_table * const t,
		int * const values
		) {
	Auto_Utility();
	timer_start();
	int part_num = t->part_num;
	// GPU buffer
	int ** dev_vertex_begin = (int **) Calloc(part_num, sizeof(int *));
	int ** dev_edge_dest = (int **) Calloc(part_num, sizeof(int *));
	int ** dev_vertex_id = (int **) Calloc(part_num, sizeof(int *));
	for (int i = 0; i < part_num; i++) {
		int size = (g[i]->vertex_num + 1) * sizeof(int);
		CudaBufferFill(dev_vertex_begin[i], size, g[i]->vertex_begin);
		size = g[i]->edge_num * sizeof(int);
		CudaBufferFill(dev_edge_dest[i], size, g[i]->edge_dest);
		size = g[i]->vertex_num * sizeof(int);
		CudaBufferFill(dev_vertex_id[i], size, t->part_vertex[i]);
	}
	for (int i = 0; i < t->vertex_num; i++) values[i] = i;
	CudaBufferCopy(int, dev_value, t->vertex_num, values);
	CudaBufferZero(int, dev_continue_flag, 1);
	// settings
	int flag = 0;
	int step = 0;
	float execTime = 0.0;

	//luo
	const int threads=MAX_THREADS_PER_BLOCK;
	dim3 blocks;
	// Main Loop
	do {
		// Clear Flag
		CudaMemset(dev_continue_flag, 0, sizeof(int));
		// Launch Kernel for this Iteration
		for (int i = 0; i < part_num; i++) {
			kernel_configure(vwarp_thread_count(g[i]->vertex_num,VWARP_WIDTH,BATCH_SIZE),
					blocks,threads);
			CudaTimerBegin();
			kernel_vertex_part1<<<blocks,threads>>>	 
				(
				 g[i]->vertex_num,
				 dev_vertex_id[i],
				 dev_vertex_begin[i],
				 dev_edge_dest[i],
				 dev_value,
				 dev_continue_flag
				);
			execTime += CudaTimerEnd();
		}
		// Copy Back Flag
		CudaMemcpyD2H(&flag, dev_continue_flag, sizeof(int));
		step++;
	} while(flag != 0 && step < 100);
	// Copy Back Values
	CudaMemcpyD2H(values, dev_value, t->vertex_num * sizeof(int));
	printf("\t%.2f\t%.2f\tpart_cc_vertex1\tstep=%d\t",
			execTime, timer_stop(), step);
}
// experiments of BFS on Graph g with Partition Table t and partitions
void cc_experiments(const Graph * const g) {

	// partition on the Graph
	printf("Partitioning ... ");
	timer_start();
	struct part_table * t =
		partition(g->vertex_num, g->edge_num, g->vertex_begin, g->edge_dest, 5);
	if (t == NULL) {
		perror("Failed !");
		exit(1);
	} else {
		printf("%.2f ms ... ", timer_stop());
	}
	// get Paritions
	printf("Get partitioins ... ");
	timer_start();
	Graph ** part = get_cut_graphs(g, t);
	if (part == NULL) {
		perror("Failed !");
		exit(1);
	} else {
		printf("%.2f ms\n", timer_stop());
	}

	int * value_cpu = (int *) calloc(g->vertex_num, sizeof(int));
	int * value_gpu = (int *) calloc(g->vertex_num, sizeof(int));
	if (value_cpu == NULL || value_gpu == NULL) {
		perror("Out of Memory for values");
		exit(1);
	}

	printf("\tTime\tTotal\tTips\n");

	cc_on_cpu(g->vertex_num, g->vertex_begin, g->edge_dest, value_cpu);
	//report_cc_values(value_cpu, g->vertex_num);

	gpu_cc_edge_loop(g, value_gpu);
	check_values(value_cpu, value_gpu, g->vertex_num);
	gpu_cc_edge_part_loop(part, t, value_gpu);
	check_values(value_cpu, value_gpu, g->vertex_num);

	gpu_cc_vertex(g, value_gpu);
	check_values(value_cpu, value_gpu, g->vertex_num);
	gpu_cc_vertex_part(part, t, value_gpu);
	check_values(value_cpu, value_gpu, g->vertex_num);
    gpu_cc_vertex_part1(part, t, value_gpu);
	check_values(value_cpu, value_gpu, g->vertex_num);

	release_table(t);
	for (int i = 0; i < 5; i++) release_graph(part[i]);
	free(part);
	free(value_cpu);
	free(value_gpu);
}

