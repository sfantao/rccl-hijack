#!/bin/bash -e
#SBATCH -J rccl-unbalanced
#SBATCH --threads-per-core 1
#SBATCH --exclusive 
#SBATCH -N 1 
#SBATCH --gpus 8
#SBATCH -t 0:10:00 
#SBATCH --mem 0
#SBATCH -o test.out
#SBATCH -e test.err

set -o pipefail

module load rocm/6.2.2 cray-mpich
bash -e preload-me.cpp

export NCCL_SOCKET_IFNAME='hsn0,hsn1,hsn2,hsn3'
export NCCL_NET_GDR_LEVEL=PHB
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL

export PATH="/home/sfantao/tickets/swdev-525949/rccl-tests/build:$PATH"

c=fe
#
# Bind mask for one and  thread per core
#
MYMASKS1="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

Nodes=1
srun \
 -l \
  -N $Nodes \
  -n $((Nodes*8)) \
  --gpus $((Nodes*8)) \
  --cpu-bind=mask_cpu:$MYMASKS1 \
  bash -e -c 'echo "$(taskset -p $$) -> $ROCR_VISIBLE_DEVICES" ; LD_PRELOAD=/home/sfantao/tickets/rccl-sync/libpreload-me.so all_reduce_perf -z 1 -b 1024M -e 2048M -f 2 -g 1 -t 1 -R 1 -n 80 -w 5 -d half'


exit 0

N=1; salloc --threads-per-core 1 -N $((N)) -c 64 --gpus $((N*8)) --time 10:00:00


# module load rocm/6.2.2 cray-mpich
# 
# srun \
#  -l \
#   -N 2 \
#   -n 9 \
#   -c 8 \
#   --gpus-per-node 8 -m plane=8 \
#   bash -e -c 'echo "$(taskset -p $$) -> $ROCR_VISIBLE_DEVICES" ; /home/sfantao/tickets/swdev-525949/rccl-tests/build/all_reduce_perf -z 1 -b 2M -e 2048M -f 2 -g 1 -t 1 -R 1 -n 80 -w 5 -d half'
