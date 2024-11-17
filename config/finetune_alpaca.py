import pickle
import time

out_dir = 'out-instruction-tuning'
eval_interval = 100 # adjust it to your need
eval_iters = 40
wandb_log = False # feel free to turn on
wandb_project = 'instruction-tuning'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'alpaca-gpt4'
init_from = 'gpt2' # this is the largest GPT-2 model

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
batch_size = 5
gradient_accumulation_steps = 2
max_iters = 5000

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False

# load the data
train_samples = pickle.load(open("data/instruction_tuning/train.pkl", "rb"))
val_samples = pickle.load(open("data/instruction_tuning/val.pkl", "rb"))
END_TOKEN = 50256 # GPT-2's token for "<|endoftext|>"

# experiment setting for part 2
dtype = "float32" # float32, float16, bfloat16. Note that bfloat16 is not supported by P100 and some old GPUs.
optimization_method = "adam" # adam, sgd, (lora or badam)

def get_batch_for_IT(split):
    """i.i.d. sample a batch of data, pad the batch into the same length for instruction tuning
    The pad token is suggested to be END_TOKEN defined above, which is the default choice for GPT-2 model series.
    
    Return:
        x: torch.tensor, shape=(batch_size, block_size)
        y: torch.tensor, shifted x, shape=(batch_size, block_size)
    """
    samples = train_samples if split == 'train' else val_samples
    
    # TODO:
    pass

def query_memory():
    """Query the memory usage of the GPU"""
    allocated_memory = torch.cuda.memory_allocated(device) / 1e9
    reserved_memory = torch.cuda.memory_reserved(device) / 1e9
    max_allocated_memory = torch.cuda.max_memory_allocated(device) / 1e9
    
    print(f"===Memory profile=== Allocated: {allocated_memory:.3f} GB, Max allocated: {max_allocated_memory:.3f} GB, Reserved: {reserved_memory:.3f} GB")
    
    return allocated_memory, reserved_memory, max_allocated_memory