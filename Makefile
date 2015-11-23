exp : main.cu graph.o partition.o vwarp.o cc.o                                          
  nvcc -arch=sm_35 -o exp main.cu obj/graph.o obj/partition.o obj/vwarp.o obj/cc.o 
 
graph.o : src/graph.c
  gcc -c -o obj/graph.o src/graph.c

partition.o : src/partition.c
  gcc -c -o obj/partition.o src/partition.c

vwarp.o:src/vwarp.cu
  nvcc -c -arch=sm_35 -o obj/vwarp.o src/vwarp.cu
  
cc.o : cc.cu
  nvcc -c -arch=sm_35 -o obj/cc.o cc.cu 
