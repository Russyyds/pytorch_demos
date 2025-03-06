# https://pytorch.org/docs/stable/distributed.html
import os
import argparse
from cv2 import broadcast
from scipy.fft import dst
import torch
import torch.distributed as dist

def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1

def get_local_rank():
    return dist.get_rank() if dist.is_initialized() else 0

def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl")
    
def destroy_ddp():
    dist.destroy_process_group()

class DistContext:
    def __init__(self, name, rank=0):
        self.name = name
        self.rank = rank
    def __enter__(self):
        if self.rank == 0:
            print(f"#### Func:{self.name} ####")
    def __exit__(self, exc_type, exc_value, traceback):
        dist.barrier()
        if self.rank == 0:
            print("#############################\n")
        
if __name__ == "__main__":
    # torchrun --nproc-per-node=3 dist_op.py
    world_size = 1
    rank = 0
    reduce_op = dist.ReduceOp.SUM
    # print("is_mpi_available:", dist.is_mpi_available()) # False
    # print("is_nccl_available:", dist.is_nccl_available()) # True
    # print("is_gloo_available:", dist.is_gloo_available()) # True
    if torch.cuda.device_count() > 1:
        ddp_setup()
        world_size = get_world_size()
        rank = get_local_rank()
    dist.barrier()
    
    # Point-to-point communication
    # 1. send & recv
    with DistContext("send & recv", rank=rank) as dc:
        src_rank = 0
        dst_rank = 1
        if rank == src_rank:
            src_tensor = torch.arange(world_size).cuda()
            dist.isend(src_tensor, dst=dst_rank)
            print(f"Rank {rank} send data to rank {dst_rank} :{src_tensor}")
        if rank == dst_rank:
            dst_tensor = torch.zeros(world_size).cuda()
            dist.recv(dst_tensor, src=src_rank)
            print(f"Rank {rank} receive data from rank {src_rank} :{dst_tensor}")

    # 2. send_object_list/recv_object_list
    with DistContext("send/recv_object_list", rank=rank) as dc:
        src_rank = 0
        dst_rank = 1
        if rank == src_rank:
            objects = ["send", 123, {"name": "Tim"}]
            dist.send_object_list(objects, dst=dst_rank)
        elif rank == dst_rank:
            objects = [None, None, None]
            dist.recv_object_list(objects, src=src_rank)
            print(f"Rank {rank} receive objects from rank {rank} :{objects}")
    
    # Collective communication
    # 1. broadcast
    # 将src_rank中的broadcast_tensor的值广播到其他rank中的broadcast_tensor中
    with DistContext("broadcast", rank=rank) as dc:
        broadcast_tensor = torch.ones((4,)).cuda() * (rank + 1)
        src_rank = 2
        dist.broadcast(broadcast_tensor, src=src_rank)
        print(f"Rank:{rank} result:{broadcast_tensor}")
    
    # 2. broadcast_object_list  
    # 将picklable的列表数据广播到其他rank中，不过列表的长度需要一致
    with DistContext("broadcast_object_list", rank=rank) as dc:
        src_rank= 1
        if dist.get_rank() == src_rank:
            objs = ["hello", "torch", 1, { "map" : 3 }]
        else :
            objs = [None, None, None, None]
        dist.broadcast_object_list(objs, src=src_rank)
        print(f"Rank:{rank} result:{objs}")
    
    # 3. reduce
    # 规约所有rank的数据到dst rank中，其余rank的数据不变
    with DistContext("reduce", rank=rank) as dc:
        dst_rank = world_size - 1
        reduce_tensor = torch.ones((3)).cuda() * (rank + 2)
        dist.reduce(reduce_tensor, dst=dst_rank, op=reduce_op)
        print(f"Rank:{rank} result:{reduce_tensor}")
    
    # 4. all_reduce
    # 规约所有rank的数据，并广播到所有的rank中
    with DistContext("all_reduce", rank=rank) as dc:
        all_reduce_tensor = torch.ones((3)).cuda() * (rank + 1)
        dist.all_reduce(all_reduce_tensor, op=reduce_op)
        print(f"Rank:{rank} result:{all_reduce_tensor}")
        
    # 5. gather
    # 将所有rank的数据汇集到dst_rank中
    with DistContext("gather", rank=rank) as dc:
        dst_rank = 0
        gather_tensor = torch.ones(3).cuda() * (rank + 2)
        if rank == dst_rank:
            gather_list = [torch.zeros_like(gather_tensor).cuda() for _ in range(world_size)]
            
        else:
            gather_list = None
        dist.gather(gather_tensor, gather_list=gather_list, dst=dst_rank)
        print(f"Rank:{rank} result:{gather_list}")
        
    # 6. all_gather
    # 将group内所有rank的数据汇集到tensor_list列表中，并分发到所有rank中
    with DistContext("all_gather", rank=rank) as dc:
        all_gather_tensor = torch.ones((3)).cuda() * (rank + 1)
        tensor_list = [torch.zeros((3), dtype=torch.float32).cuda() for _ in range(world_size)]
        dist.all_gather(tensor_list=tensor_list, tensor=all_gather_tensor)
        print(f"Rank:{rank} result:{tensor_list}")
        
    # 7. all_gather_object
    # 将group内的所有rank的picklable的数据汇集到tensor_list列表中，并分发到所有rank中
    with DistContext("all_gather_object", rank=rank) as dc:
        gather_objs = ["hey", 1, { "index": 2 }]
        output = [None for _ in gather_objs]
        dist.all_gather_object(output, gather_objs)
        print(f"Rank:{rank} result:{output}")
    
    # 8. scatter
    # 将src_rank的Tensor列表分发到所有rank中
    with DistContext("scatter", rank=rank) as dc:
        src_rank = 1
        tensor_size = 3
        scatter_tensor = torch.zeros(tensor_size).cuda()
        if rank == src_rank:
            # 在src_rank中创建一个递增的Tensor列表
            scatter_list = [torch.ones(tensor_size).cuda() * (i + 1) for i in range(world_size)]
        else :
            scatter_list = None
        # 将该递增的Tensor列表的每个元素依次分发到所有rank的scatter_tensor中
        dist.scatter(scatter_tensor, scatter_list, src=src_rank)
        print(f"Rank:{rank} result:{scatter_tensor}")

    # 9. scatter_object_list
    # 将picklable的数据分发到group内所有rank
    with DistContext("scatter_object_output_list", rank=rank) as dc:
        src_rank = 2
        if rank == src_rank:
            scatter_obj_list = ["scatter_object_list", 2, {"tmp" : 1}]
        else :
            scatter_obj_list = [None, None, None]
        scatter_obj_output_list = [None]
        dist.scatter_object_list(scatter_object_output_list=scatter_obj_output_list, scatter_object_input_list=scatter_obj_list, src=src_rank)
        print(f"Rank:{rank} result:{scatter_obj_output_list}")
    
    # 10. reduce_scatter
    # 先对不同rank中的Tensor列表里的每个元素进行逐元素规约，然后将规约结果依次分发到所有rank
    # 和all_reduce的区别主要在于输入数据格式不一样，然后分发的结果不一样，all_reduce分发的是全局的规约结果，reduce_scatter的规约结果是一个列表，将列表的每个元素按顺序分发到不同rank中
    with DistContext("reduce_scatter", rank=rank) as dc:
        tensor_size = 3
        input_list = [torch.ones(tensor_size).cuda() * (rank + i) for i in range(world_size)]
        # rank 0 : [0, 1, 2]
        # rank 1 : [1, 2, 3]
        # rank 2 : [2, 3, 4]
        output_tensor = torch.zeros(tensor_size).cuda()
        dist.reduce_scatter(output=output_tensor, input_list=input_list, op=reduce_op)
        print(f"Rank:{rank} result:{output_tensor}")
        # rank 0 : [3, 3, 3]
        # rank 1 : [6, 6, 6]
        # rank 2 : [9, 9, 9]
    
    # 11. reduce_scatter_tensor
    # 将不同rank的Tensor逐元素进行规约，然后规约的结果按顺序分发到不同rank上
    with DistContext("reduce_scatter_tensor", rank=rank) as dc:
        tensor_size = 3
        tensor_out = torch.zeros(tensor_size, dtype=torch.int64).cuda()
        tensor_in = torch.arange(tensor_size * world_size).cuda()
        # rank 0 : [0, 1, 2, 3, 4, 5, 6, 7, 8]
        # rank 1 : [0, 1, 2, 3, 4, 5, 6, 7, 8]
        # rank 2 : [0, 1, 2, 3, 4, 5, 6, 7, 8]
        dist.reduce_scatter_tensor(tensor_out, tensor_in)
        print(f"Rank:{rank} result:{tensor_out}")
        # rank 0 : [0, 3, 6]
        # rank 1 : [9, 12, 15]
        # rank 2 : [18, 21, 24]
    
    if torch.cuda.device_count() > 1:
        destroy_ddp()