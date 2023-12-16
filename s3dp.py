import torch
import config
import time

class S3DP:
    def __init__(self, model, config: config):
        self.config = config
        self.model = model
        self.world_size = config.world_size
        self.all_module = [(name, module) for name, module in model.named_modules()][0][1]
        # self.all_layers = self.all_module.model.layers

        # by default, we put embedding into the first and last rank in PP
        # drop to fist rank, head to last rank
        self.token_embedding = self.all_module.transformer.wte 
        self.positional_embedding = self.all_module.transformer.wpe
        self.drop = self.all_module.transformer.drop
        
        self.lm_head = self.all_module.lm_head
        self.all_layers = self.all_module.transformer.h
        self.ln_f = self.all_module.transformer.ln_f  # final layer norm 
        
        self.init_self()
        self.init_groups()

        self.is_first_stage = self.local_rank_in_pipeline == 0
        self.is_last_stage = self.local_rank_in_pipeline == config.pipeline_model_parallel_size - 1

        self.init_weights()
        
        # print("rank:{} has layer: {}".format(self.local_rank, self.layers))
        #time.sleep(10)
        
   
    def init_self(self): 
        # torch.distributed.init_process_group(backend='nccl')
        self.rank = torch.distributed.get_rank()
        self.local_rank = self.rank % torch.cuda.device_count()
        

    def init_groups(self):    
        world_size = self.world_size
        self.data_parallel_size = world_size // (self.config.tensor_model_parallel_size *
                                  self.config.pipeline_model_parallel_size) 
        self.num_tensor_model_parallel_groups = world_size // self.config.tensor_model_parallel_size 
        self.num_pipeline_model_parallel_groups = world_size // self.config.pipeline_model_parallel_size 
        self.num_data_parallel_groups = world_size // self.data_parallel_size 
        self.all_data_parallel_group_ranks = self.get_data_parallel_group_ranks()
        self.all_tensor_model_parallel_group_ranks = self.get_tensor_model_parallel_group_ranks()
        self.all_pipeline_model_parallel_group_ranks = self.get_pipeline_model_parallel_group_ranks()
    

    def init_weights(self):
        if self.is_first_stage:
            self.token_embedding.to("cuda:{}".format(self.local_rank))
            self.positional_embedding.to("cuda:{}".format(self.local_rank))
            self.drop.to("cuda:{}".format(self.local_rank))

        if self.is_last_stage:
            self.token_embedding.to("cuda:{}".format(self.local_rank))
            self.ln_f.to("cuda:{}".format(self.local_rank))
            self.lm_head.to("cuda:{}".format(self.local_rank))

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
                self.local_rank_in_pipeline = ranks.index(self.rank)
                self.prev_rank_in_pipeline = ranks[(self.local_rank_in_pipeline - 1) % self.config.pipeline_model_parallel_size]
                self.next_rank_in_pipeline = ranks[(self.local_rank_in_pipeline + 1) % self.config.pipeline_model_parallel_size]
            all_pipeline_model_parallel_group_ranks.append(list(ranks))
        return all_pipeline_model_parallel_group_ranks


    def pre_process_layer(self, input_ids):
        """Everything before Attention blocks"""
        position_ids = torch.arange(0, input_ids.size()[-1], dtype=torch.long, device=self.local_rank)
        position_ids = position_ids.unsqueeze(0)

        inputs_embeds = self.token_embedding(input_ids)
        position_embeds = self.positional_embedding(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)
        return hidden_states

    def post_process_layer(self, hidden_states): 
        """Everything after Attention blocks"""
        hidden_states = self.ln_f(hidden_states)
        hidden_states = self.lm_head(hidden_states)
        return hidden_states