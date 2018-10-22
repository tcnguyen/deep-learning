import numpy as np

def get_batches(arr, batch_size, n_steps):
    '''Create a generator that returns batches of size
       batch_size x n_steps from arr.
       
       Arguments
       ---------
       arr: Array you want to make batches from
       batch_size: Batch size, the number of sequences per batch
       n_steps: Number of sequence steps per batch
    '''
    # Get the number of characters per batch and number of batches we can make
    chars_per_batch = batch_size * n_steps
    n_batches = len(arr)//chars_per_batch
    
    # Keep only enough characters to make full batches
    arr = arr[:n_batches * chars_per_batch]
    
    # Reshape into batch_size rows
    arr = arr.reshape((batch_size, -1))
    
    for n in range(0, arr.shape[1], n_steps):
        # The features
        x = arr[:, n:n+n_steps]
        # The targets, shifted by one
        y_temp = arr[:, n+1:n+n_steps+1]
        
        # For the very last batch, y will be one character short at the end of 
        # the sequences which breaks things. To get around this, I'll make an 
        # array of the appropriate size first, of all zeros, then add the targets.
        # This will introduce a small artifact in the last batch, but it won't matter.
        y = np.zeros(x.shape, dtype=x.dtype)
        y[:,:y_temp.shape[1]] = y_temp
        
        yield x, y
