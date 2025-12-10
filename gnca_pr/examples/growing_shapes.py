import torch
import torch.nn.functional as F
import numpy as np
import os
import gc
from pygsp.graphs import Bunny
from gnca import GNCAModel, StateCache



def get_data():
    graph = Bunny()
    pos = torch.tensor(graph.coords, dtype=torch.float)
    
    # 1. Center everything around the mean
    mean = pos.mean(dim=0)
    pos = pos - mean
    
    # 2. Scale by max norm to make the points fit within a unit sphere
    max_norm = pos.norm(p=2, dim=1).max()
    pos = pos / max_norm
    
    # 3. Get edges
    rows, cols = graph.W.nonzero()
    edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)
    
    return pos, edge_index

def sphericalize_state(target_pos):
    """Init points as a hollow sphere "seed" state"""
    norms = target_pos.norm(p=2, dim=1, keepdim=True)
    norms = torch.where(norms == 0, torch.ones_like(norms), norms)
    return target_pos / norms


def train():
    # keeping memory clean
    gc.collect()
    torch.cuda.empty_cache()
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Note: paper uses batch size 8, but to reduce memory, we're using 1
    # Then we accumulate gradients over 8 steps
    TARGET_BATCH_SIZE = 8
    PHYSICAL_BATCH_SIZE = 1
    ACCUM_STEPS = TARGET_BATCH_SIZE // PHYSICAL_BATCH_SIZE
    
    # training hyperparameters (seem to work well)
    # increasing learning rate makes things unstable
    LR = 0.001
    POOL_SIZE = 1024
    N_EPOCHS = 2000
    
    # data setup
    target_pos, edge_index = get_data() # final 3D target shape
    target_pos = target_pos.to(device)
    edge_index = edge_index.to(device)
    num_nodes = target_pos.size(0)
    # init seed state (normed) $\bar{s}_i = \hat{s}_i / ||\hat{s}_i||$
    seed_state = sphericalize_state(target_pos)

    # batched edge index
    edge_indices_list = []
    for i in range(PHYSICAL_BATCH_SIZE):
        edge_indices_list.append(edge_index + i * num_nodes)
    batched_edge_index = torch.cat(edge_indices_list, dim=1).to(device)

    # setup model (hidden=128 to reduce memory)
    model = GNCAModel(input_channels=3, output_channels=3, hidden_channels=128, output_activation='tanh').to(device)
    
    # Zero-init last layer helps training stability
    # Forces the model to start as "Identity" (Delta=0)
    # The rest of the network uses Xavier init
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
                
    with torch.no_grad():
        model.mlp_post[-1].weight.fill_(0.0)
        model.mlp_post[-1].bias.fill_(0.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # sample pool to store previous states to prevent catastrophic forgetting
    # this is the "cache" from the paper
    pool = StateCache(POOL_SIZE, seed_state) 
    
    print(f"STARTING TRAINING ON {device} (Accumulating {ACCUM_STEPS} steps)...", flush=True)
    
    optimizer.zero_grad()
    
    for epoch in range(N_EPOCHS):
        epoch_loss = 0
        
        # Accumulate Gradients
        for _ in range(ACCUM_STEPS):
            batch_idx, current_states = pool.sample(PHYSICAL_BATCH_SIZE)
            current_states = current_states.to(device)
            # at least one sample in every batch is the original starting seed 
            # (so it doesn't forget how to grow from beginning)
            current_states[0] = seed_state.clone() 
            
            steps = torch.randint(10, 21, (1,)).item()
            x = current_states
            
            # unroll the model for a random number of steps between 10 and 20
            # this is the "growth" part of the model (simulating forward evolution)
            for _ in range(steps):
                x_flat = x.view(-1, 3)
                delta_flat = model(x_flat, batched_edge_index) # prediction
                delta = delta_flat.view(PHYSICAL_BATCH_SIZE, num_nodes, 3)
                x = x + delta
                x = torch.clamp(x, -1.0, 1.0)
                
            loss = F.mse_loss(x, target_pos.unsqueeze(0).expand(PHYSICAL_BATCH_SIZE, -1, -1))
            loss = loss / ACCUM_STEPS 
            loss.backward() # backpropagate the loss through the model (over all steps)
            
            epoch_loss += loss.item()
            pool.update(batch_idx, x.detach())

        # Optimizer Step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        # Logging checkpoints
        if epoch % 50 == 0 or epoch < 10:
            delta_mag = delta.abs().mean().item()
            # keep an eye on delta mean (if it gets too big, it means the model is diverging)
            print(f"Epoch {epoch} | Loss: {epoch_loss:.6f} | Delta Mean: {delta_mag:.6f}", flush=True)
            
            if epoch % 50 == 0:
                os.makedirs('checkpoints', exist_ok=True)
                torch.save(model.state_dict(), 'checkpoints/bunny_gnca.pt')
                
            if epoch % 100 == 0:
                torch.cuda.empty_cache()

    print("TRAINING DONE SUCCESSFULLY")

if __name__ == '__main__':
    train()