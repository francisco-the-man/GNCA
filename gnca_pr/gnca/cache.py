import torch
import numpy as np

class StateCache:
    """
    Stores states of GNCA model for replay memory
    This lets the model learn from previous states and prevents catastrophic forgetting
    """
    def __init__(self, size, initial_state):
        """
        Args:
            size (int): The number of states to keep in the cache.
            initial_state (torch.Tensor): The seed state (e.g., sphere or noise) 
                                          used to initialize and reset the pool.
        """
        self.size = size
        # Store initial_state to be used for resets
        # Ensure we clone it so we don't modify the original reference
        self.init_state = initial_state
        self.cache = [initial_state.clone() for _ in range(size)] 
        
    def sample(self, count):
        """
        Samples a random batch of states from the cache.
        
        Args:
            count (int): Batch size to sample.
            
        Returns:
            idxs (torch.Tensor): The indices of the chosen states.
            batch (torch.Tensor): The stacked states of the chosen indices.
        """
        # Randomly select indices
        idxs = torch.randint(0, len(self.cache), (count,))
        batch = [self.cache[i] for i in idxs]
        
        # Stack them into one tensor for the model
        return idxs, torch.stack(batch)

    def update(self, idxs, new_states):
        """
        Updates the cache with the newly evolved states.
        
        Args:
            idxs (torch.Tensor): The indices of the states to update.
            new_states (torch.Tensor): The new states output by the model.
        """
        # Detach to stop gradient from growing forever
        for i, idx in enumerate(idxs):
            self.cache[idx] = new_states[i].detach().clone()
            
        # Replace one random sample with the initial seed
        # This prevents model from "drifting too far" and forgetting how to grow from beginning
        replace_idx = torch.randint(0, len(self.cache), (1,)).item()
        self.cache[replace_idx] = self.init_state.detach().clone()