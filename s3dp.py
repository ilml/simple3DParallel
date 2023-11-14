import torch
import config
import time

class S3DP:
    def __init__(self, model, config: config):
        self.config = config
        self.model = model
        self.all_module = [(name, module) for name, module in model.named_modules()][0][1]
        # self.all_layers = self.all_module.model.layers
        self.all_layers = self.all_module.transformer.h
        self.init_self()
        self.init_groups()
        self.init_weights()
        
        print("rank:{} has layer: {}".format(self.local_rank, self.layers))
        #time.sleep(10)
        

   
    def init_self(self): 
        torch.distributed.init_process_group(backend='nccl')
        self.rank = torch.distributed.get_rank()
        self.local_rank = self.rank % torch.cuda.device_count()
        

    def init_groups(self):    
        world_size = self.config.world_size 
        self.data_parallel_size = world_size // (self.config.tensor_model_parallel_size *
                                  self.config.pipeline_model_parallel_size) 
        self.num_tensor_model_parallel_groups = world_size // self.config.tensor_model_parallel_size 
        self.num_pipeline_model_parallel_groups = world_size // self.config.pipeline_model_parallel_size 
        self.num_data_parallel_groups = world_size // self.data_parallel_size 
        self.all_data_parallel_group_ranks = self.get_data_parallel_group_ranks()
        self.all_tensor_model_parallel_group_ranks = self.get_tensor_model_parallel_group_ranks()
        self.all_pipeline_model_parallel_group_ranks = self.get_pipeline_model_parallel_group_ranks()
    

    def init_weights(self):
        num_layer_each = len(self.all_layers) // self.config.pipeline_model_parallel_size 
        for idx, rank in enumerate(self.pipeline_model_parallel_group_ranks):
            if rank == self.rank:
                #print("rank:{} get {}".format(rank, list(range(idx * num_layer_each, (idx + 1) * num_layer_each))))
                self.layers = self.all_layers[idx * num_layer_each: (idx + 1) * num_layer_each]
                break
        self.layers.to("cuda:{}".format(self.local_rank))


    def get_data_parallel_group_ranks(self):
        # Build the data-parallel groups.
        all_data_parallel_group_ranks = []
        for i in range(self.config.pipeline_model_parallel_size):
            start_rank = i * self.num_pipeline_model_parallel_groups
            end_rank = (i + 1) * self.num_pipeline_model_parallel_groups
            for j in range(self.config.tensor_model_parallel_size):
                ranks = range(start_rank + j, end_rank,
                              self.config.tensor_model_parallel_size)
                group = torch.distributed.new_group(ranks)
                if self.rank in ranks:
                    self.data_parallel_group = group    
                    self.data_parallel_group_ranks= list(ranks)
                all_data_parallel_group_ranks.append(list(ranks))
        return all_data_parallel_group_ranks


    def get_tensor_model_parallel_group_ranks(self):
        # Build the tensor model-parallel groups.
        all_tensor_model_parallel_group_ranks = []
        for i in range(self.num_tensor_model_parallel_groups):
            ranks = range(i * self.config.tensor_model_parallel_size,
                          (i + 1) * self.config.tensor_model_parallel_size)
            group = torch.distributed.new_group(ranks)
            if self.rank in ranks:
                self.tensor_model_parallel_group= group    
                self.tensor_model_parallel_group_ranks= list(ranks)
            all_tensor_model_parallel_group_ranks.append(list(ranks))
        return all_tensor_model_parallel_group_ranks 
   
        
    def get_pipeline_model_parallel_group_ranks(self):
        # Build the pipeline model-parallel groups and embedding groups
        all_pipeline_model_parallel_group_ranks = []
        for i in range(self.num_pipeline_model_parallel_groups):
            ranks = range(i, self.config.world_size,
                          self.num_pipeline_model_parallel_groups)
            group = torch.distributed.new_group(ranks)
            if self.rank in ranks:
                self.pipeline_model_parallel_group = group    
                self.pipeline_model_parallel_group_ranks= list(ranks)
            all_pipeline_model_parallel_group_ranks.append(list(ranks))
        return all_pipeline_model_parallel_group_ranks