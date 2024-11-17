import pickle
import time

out_dir = 'out-instruction-tuning'
eval_interval = 50
eval_iters = 40
wandb_log = False # feel free to turn on
wandb_project = 'instruction-tuning'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'alpaca-gpt4'
init_from = 'gpt2' # this is the largest GPT-2 model

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
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
PAD_TOKEN = -1


def get_batch_for_IT(split):
    """get batch data for instruction tuning"""
    samples = train_samples if split == 'train' else val_samples
    ix = torch.randint(len(samples), (batch_size,))
    batch = [samples[i] for i in ix]
    
    # pad the samples to the same length
    max_len = max([len(x) for x in batch])
    for i in range(len(batch)):
        batch[i] += [END_TOKEN] * (max_len - len(batch[i]))
        
    batch = torch.tensor(batch, dtype=torch.long).to(device)
    batch = batch[:, :block_size] # crop to block size
    x = batch[:, :-1].contiguous()
    y = batch[:, 1:].contiguous()
    return x, y


def query_memory():
    """Query the memory usage of the GPU"""
    allocated_memory = torch.cuda.memory_allocated(device) / 1e9
    reserved_memory = torch.cuda.memory_reserved(device) / 1e9
    max_allocated_memory = torch.cuda.max_memory_allocated(device) / 1e9
    
    print(f"===Memory profile=== Allocated: {allocated_memory:.3f} GB, Max allocated: {max_allocated_memory:.3f} GB, Reserved: {reserved_memory:.3f} GB")
    
    return allocated_memory, reserved_memory, max_allocated_memory