"""
Tiny GPT From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_vocab
def build_vocab(text):
    """Return a sorted list of unique characters in text."""
    # TODO: return a sorted list of every unique character in text
    return sorted(set(text))

# Step 2 - build_stoi
def build_stoi(vocab):
    """Return a dict mapping each character in vocab to its index."""
    # TODO: map each character in vocab to its integer position
    return {char: index for index, char in enumerate(vocab)}

# Step 3 - build_itos
def build_itos(vocab):
    """Return a dict mapping each index 0..len(vocab)-1 to its character."""
    # TODO: build an int-to-string lookup from the vocab list
    return {index: char for index, char in enumerate(vocab)}

# Step 4 - encode_char
def encode_char(ch, stoi):
    """Return the integer token id for a single character ch using stoi."""
    # TODO: look up ch in the stoi mapping and return its id
    return stoi[ch]

# Step 5 - encode_string
def encode_string(text, stoi):
    """Encode a full string into a list of token ids using stoi."""
    # TODO: map each char in text through stoi (via encode_char) into a list of ids
    return [encode_char(ch, stoi) for ch in text]

# Step 6 - decode_int
def decode_int(token_id, itos):
    """Return the single character mapped to token_id by itos."""
    # TODO: look up the character for token_id in the itos dict
    return itos[token_id]

# Step 7 - decode_ids
def decode_ids(ids, itos):
    """Decode a list of token ids into a string using itos."""
    # TODO: map each id through decode_int and join the characters into one string.
    return "".join([decode_int(id, itos) for id in ids])

# Step 8 - make_1d_array
import numpy as np

def make_1d_array(values):
    """Create a 1D NumPy array from a Python list of numbers."""
    # TODO: convert the input list into a 1D numpy ndarray
    return np.array(values)

# Step 9 - get_array_shape
import numpy as np

def get_array_shape(arr):
    """Return the shape tuple of a NumPy array."""
    # TODO: return the shape of arr
    return arr.shape

# Step 10 - get_array_dtype
import numpy as np

def get_array_dtype(arr):
    """Return the dtype of a NumPy array."""
    # TODO: return the dtype attribute of arr
    return arr.dtype

# Step 11 - make_2d_zeros
import numpy as np

def make_2d_zeros(rows, cols):
    """Return a 2D NumPy array of zeros with shape (rows, cols)."""
    # TODO: allocate a (rows, cols) array of zeros and return it
    return np.zeros((rows , cols))

# Step 12 - make_2d_random
import numpy as np

def make_2d_random(rows, cols, seed):
    """Return a (rows, cols) array of uniform floats in [0, 1) seeded by `seed`."""
    # TODO: build a seeded RNG and draw a (rows, cols) uniform sample in [0, 1).
    rng = np.random.default_rng(seed)
    return rng.random((rows, cols))

# Step 13 - index_element
def index_element(arr, i, j):
    """Return the scalar element at position (i, j) of a 2D array."""
    # TODO: return the value at row i, column j of arr
    return arr[i , j]

# Step 14 - slice_row
import numpy as np

def slice_row(arr, i):
    """Return row i of a 2D array as a 1D view."""
    # TODO: return the i-th row of arr as a 1D array of shape (C,)
    return arr[i]

# Step 15 - slice_column
import numpy as np

def slice_column(arr, j):
    """Return column j of a 2D array as a 1D array of length R."""
    # TODO: index into arr to extract the j-th column as a 1D array.
    return arr[:,j]

# Step 16 - slice_subblock
import numpy as np

def slice_subblock(arr, r0, r1, c0, c1):
    """Return the sub-block arr[r0:r1, c0:c1] of a 2D array."""
    # TODO: return the rectangular sub-block of arr bounded by rows [r0,r1) and cols [c0,c1).
    return arr[r0:r1,c0:c1]

# Step 17 - elementwise_add
import numpy as np

def elementwise_add(a, b):
    """Return the elementwise sum of two same-shape arrays."""
    # TODO: return a new array whose entries are the pairwise sums of a and b
    return a+b

# Step 18 - elementwise_multiply
import numpy as np

def elementwise_multiply(a, b):
    """Return the elementwise product of two same-shape arrays."""
    # TODO: compute the elementwise (Hadamard) product of a and b
    return a * b

# Step 19 - scalar_broadcast_add
import numpy as np

def scalar_broadcast_add(arr, scalar):
    """Return a new array equal to arr with scalar added to every element."""
    # TODO: add a Python scalar to every element of an array via broadcasting
    return arr + scalar

# Step 20 - vector_matrix_broadcast_add
import numpy as np

def vector_matrix_broadcast_add(matrix, vector):
    """Add a 1D vector to each row of a 2D matrix via broadcasting."""
    # TODO: return matrix + vector broadcast across rows
    return matrix + vector

# Step 21 - array_exp
import numpy as np

def array_exp(arr):
    """Return the elementwise exponential of arr."""
    # TODO: apply elementwise exponential to arr and return the result
    return np.exp(arr)

# Step 22 - array_log
import numpy as np

def array_log(arr):
    """Return the elementwise natural log of arr (assumes arr > 0)."""
    # TODO: apply elementwise natural log to arr and return the result
    return np.log(arr)

# Step 23 - sum_all
import numpy as np

def sum_all(arr):
    """Return the sum of every element of arr as a scalar."""
    # TODO: collapse every element of arr into a single scalar total
    return np.sum(arr)

# Step 24 - sum_axis0
import numpy as np

def sum_axis0(arr):
    """Sum a 2D array along axis 0, collapsing rows into a 1D vector of column sums."""
    # TODO: reduce the row dimension of arr so the result has shape (C,).
    return np.sum(arr , axis = 0)

# Step 25 - sum_axis1
import numpy as np

def sum_axis1(arr):
    """Sum a 2D array along axis 1, returning a 1D array of row sums."""
    # TODO: collapse the column dimension by summing each row
    return np.sum(arr, axis = 1)

# Step 26 - max_along_axis
import numpy as np

def max_along_axis(arr, axis):
    """Return the maximum of arr along the given axis, with that axis removed."""
    # TODO: compute the maximum value of arr along the given axis
    return np.max(arr , axis = axis)

# Step 27 - matmul
import numpy as np

def matmul(a, b):
    """Return the matrix product a @ b for 2D arrays a (M,K) and b (K,N)."""
    # TODO: compute the matrix product of a and b
    return a@b

# Step 28 - transpose_matrix
def transpose_matrix(arr):
    """Return the transpose of a 2D array."""
    # TODO: return the transpose of arr using the .T attribute
    return arr.T

# Step 29 - sum_keepdims
import numpy as np

def sum_keepdims(arr, axis):
    """Sum along `axis` while keeping that dimension as size 1."""
    # TODO: sum along the given axis preserving the reduced dim as size 1
    return np.sum(arr , axis = axis , keepdims=True)

# Step 30 - naive_softmax_1d
import numpy as np

def naive_softmax_1d(logits):
    """Compute softmax of a 1D logits vector via the direct exp/sum formula."""
    # TODO: exponentiate the logits, then divide by their total sum
    return array_exp(logits) / sum_all(np.exp(logits))

# Step 31 - softmax_overflow_demo
def softmax_overflow_demo(large_value):
    """Show that naive exp overflows on a large logit.

    Return {'naive_exp': float, 'overflowed': bool}.
    """
    # TODO: exponentiate large_value via array_exp and report whether it is inf.
    exp_arr = array_exp([large_value])
    result = exp_arr[0]
    return {'naive_exp': float(result), 'overflowed': bool(np.isinf(result))}

# Step 32 - stable_softmax_1d
import numpy as np

def stable_softmax_1d(logits):
    """Numerically stable softmax over a 1D logits vector."""
    # TODO: subtract the max before exponentiating, then normalize.
    shifted = logits - max_along_axis(logits, axis=0)
    stable_softmax = array_exp(shifted) / sum_all(array_exp(shifted))
    return stable_softmax

# Step 33 - stable_softmax_2d_rowwise
import numpy as np

def stable_softmax_2d_rowwise(logits):
    """Row-wise numerically stable softmax of a 2D logits array."""
    # TODO: turn each row of logits into a probability distribution without overflowing
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    return array_exp(shifted) / sum_keepdims(array_exp(shifted), axis=1)

# Step 34 - read_text_file
def read_text_file(text_blob):
    if not isinstance(text_blob, str):
        raise TypeError("It should be string")
    if not text_blob:
        raise ValueError("Empty")
    return text_blob

# Step 35 - encode_corpus_to_int_array
def encode_corpus_to_int_array(text, stoi):
    result = encode_string(text, stoi)
    return np.array(result, dtype=np.int64)

# Step 36 - pick_split_point
def pick_split_point(n, train_frac):
    result = n * train_frac
    return int(result)

# Step 37 - slice_train_and_val
def slice_train_and_val(data, split_idx):
    """Split a 1D token-id array into (train, val) at split_idx."""
    # TODO: return (data[:split_idx], data[split_idx:])
    return (data[:split_idx],data[split_idx:])

# Step 38 - pick_block_size
def pick_block_size(default_size):
    if default_size < 1:
        return 1
    return default_size

# Step 39 - slice_x_at_offset
import numpy as np

def slice_x_at_offset(data, i, block_size):
    """Return the input window data[i : i + block_size]."""
    # TODO: extract a single input window of length block_size starting at index i
    return data[i:i+block_size]

# Step 40 - slice_y_at_offset
import numpy as np

def slice_y_at_offset(data, i, block_size):
    """Return the target window of length block_size starting at i+1."""
    # TODO: extract the target window Y = data[i+1 : i+1+block_size] shifted by one.
    return data[i + 1 : i + block_size + 1]

# Step 41 - sample_random_batch_offsets
def sample_random_batch_offsets(data_len, block_size, batch_size, rng):
    """Sample batch_size random valid starting offsets for (block_size+1)-windows."""
    return rng.integers(
        low=0,
        high=data_len - block_size,
        size=batch_size
    )

# Step 42 - stack_x_batch
import numpy as np

def stack_x_batch(data, offsets, block_size):
    """Build a batch matrix where each row is a window from data."""
    
    batch = [
        slice_x_at_offset(data, i, block_size)
        for i in offsets
    ]
    
    return np.stack(batch)

# Step 43 - stack_y_batch
import numpy as np

def stack_y_batch(data, offsets, block_size):
    """Stack per-offset Y windows into a 2D (B, block_size) target matrix."""
    # TODO: for each offset, take the length-block_size slice starting at i+1 and stack rows
    batch = [
        slice_y_at_offset(data, i, block_size) 
        for i in offsets 
    ]
    return np.array(batch)

# Step 44 - get_batch
def get_batch(data, block_size, batch_size, rng):
    # TODO: package one training batch (X, Y) of shape (batch_size, block_size) from data using rng.
    random_str = sample_random_batch_offsets(len(data), block_size , batch_size, rng)
    X = stack_x_batch(data , random_str , block_size)
    y = stack_y_batch(data, random_str, block_size)
    return (X,y)

# Step 45 - allocate_count_matrix
import numpy as np

def allocate_count_matrix(vocab_size):
    """Allocate a (V, V) integer zero matrix for bigram counts."""
    # TODO: return a (vocab_size, vocab_size) integer array of zeros.
    return np.zeros((vocab_size, vocab_size), dtype=np.int64)

# Step 46 - loop_fill_counts
import numpy as np

def loop_fill_counts(n_matrix, data):
    """Increment n_matrix[curr, next] for every consecutive pair in data."""
    # we use -1 to stop before the last element like data[3] .... data[3 + 1]
    for t in range(len(data) - 1) :
        pair  = (data[t], data[t+1])
        n_matrix[data[t], data[t+1]] += 1 
    return n_matrix

# Step 47 - vectorize_counts_add_at
import numpy as np

def vectorize_counts_add_at(vocab_size, data):
    """Build (V, V) bigram counts from a 1D id array using vectorized scatter-add."""
    # The key tool here is np.add.at()
    # It adds 1 to every position (rows[i], cols[i]) in the matrix — all at once, no loop!
    allocate_counts = allocate_count_matrix(vocab_size)
    np.add.at(allocate_counts, (data[:-1], data[1:]), 1)
    return allocate_counts

# Step 48 - add_one_smoothing
import numpy as np

def add_one_smoothing(n_matrix):
    """Return n_matrix with every entry incremented by 1 (Laplace smoothing)."""
    # Every bigram needs at least a tiny probability so the model can generate any character, 
    # not get stuck on ones it never saw in training.
    return n_matrix + 1

# Step 49 - row_sums_of_counts
def row_sums_of_counts(n_matrix):
    """Return per-row sums of n_matrix with shape (V, 1)."""
    # TODO: compute per-row sums of the count matrix as a column vector for normalization.
    return sum_keepdims(n_matrix,1)

# Step 50 - normalize_counts_to_probs
def normalize_counts_to_probs(n_matrix):
    """Normalize a (V, V) count matrix into a row-stochastic probability matrix."""
    # you need to divide n_matrix by the row sums 
    return n_matrix / row_sums_of_counts(n_matrix)

# Step 51 - sample_next_token
def sample_next_token(p_matrix, current_id, rng):
    """Sample the next token id from P[current_id] using rng."""
    # You have a row of probabilities like [0.1, 0.6, 0.3] — these are the chances of each next token. 
    #You need to randomly pick one token according to those probabilities.
    return rng.choice(len(p_matrix[current_id]), p=p_matrix[current_id])

# Step 52 - generate_sequence
def generate_sequence(p_matrix, start_id, length, rng):
    """Autoregressively sample `length` token ids from a bigram matrix, starting with `start_id`."""
    # TODO: build a length-L int array starting at start_id, then sample each next id from p_matrix
    lst = [start_id]
    for _ in range(length - 1):
        current_id = sample_next_token(p_matrix,lst[-1],rng)
        lst.append(current_id)
        
    return np.array(lst)

# Step 53 - decode_generated_sequence
def decode_generated_sequence(ids, itos):
    """Decode a generated 1D array/list of token ids into a string via itos."""
    # TODO: turn ids into a readable string using itos
    return decode_ids(ids, itos)

# Step 54 - log_prob_of_pair
def log_prob_of_pair(p_matrix, current_id, next_id):
    """Return the log probability of a single (current, next) bigram."""
    # TODO: pick out P[current_id, next_id] and return its natural log
    rst = index_element(p_matrix,current_id,next_id)
    return array_log(rst)

# Step 55 - sum_negative_log_probs
def sum_negative_log_probs(p_matrix, data):
    # TODO: sum the negative log probabilities of all consecutive bigrams in data
    # Log probabilities are always negative — because probabilities are between 0 and 1
    # so thats why we use -
    total = 0.0 
    for t in range(len(data) - 1):
        total += -log_prob_of_pair(p_matrix, data[t], data[t+1])
    return total

# Step 56 - average_nll
def average_nll(p_matrix, data):
    # TODO: return mean negative log likelihood per bigram over consecutive pairs in data.
    # we used -1 because you pair each token with the next one, so the last token has no pair
    n = len(data) - 1
    return sum_negative_log_probs(p_matrix, data) / n

# Step 57 - initialize_w_random
import numpy as np

def initialize_w_random(vocab_size, rng):
    """Return a (vocab_size, vocab_size) float64 matrix of N(0,1) samples drawn from rng."""
    # TODO: sample a (vocab_size, vocab_size) array of standard normal values using rng
    return rng.standard_normal((vocab_size, vocab_size))
    # or use rng.normal(size=(vocab_size, vocab_size))

# Step 58 - scale_w_small
import numpy as np

def scale_w_small(w_matrix, scale):
    """Return w_matrix scaled by the given small factor."""
    # TODO: return a new array equal to w_matrix multiplied by scale
    return w_matrix * scale

# Step 59 - one_hot_encode_batch
import numpy as np

def one_hot_encode_batch(ids, vocab_size):
    """Convert a 1D array of token ids into a (N, vocab_size) one-hot matrix."""
    # each token gets a row where everything is 0 except one position which is 1. 
    #That position is the token's index:
    zeros = make_2d_zeros(len(ids), vocab_size)
    for i in range(len(ids)):
        zeros[i][ids[i]] = 1.0
    return zeros

# Step 60 - forward_logits_onehot
def forward_logits_onehot(onehot, w_matrix):
    # TODO: compute logits for the neural bigram model as the matrix product of one-hot inputs and W.
    return matmul(onehot, w_matrix)

# Step 61 - observe_lookup_equivalence
import numpy as np

def observe_lookup_equivalence(w, ids):
    """Show that one-hot @ W equals W[ids] for a small example.
    Returns a dict with keys 'onehot_result' and 'index_result'.
    """
    # TODO: compute logits two ways and return both in a dict
    onehot = one_hot_encode_batch(ids, len(w))
    onehot_result = forward_logits_onehot(onehot, w)
    index_result = w[ids]
    return {'onehot_result': onehot_result, 'index_result': index_result}

# Step 62 - forward_logits_lookup
def forward_logits_lookup(w, ids):
    """Return logits (B, V) by gathering rows of w at positions ids."""
    # TODO: return the logits for a batch of token ids by direct row lookup into W.
    return w[ids]

# Step 63 - logits_to_probs_rowwise
def logits_to_probs_rowwise(logits):
    # TODO: convert a (B, V) logits matrix into a row-wise probability matrix
    return stable_softmax_2d_rowwise(logits)

# Step 64 - gather_correct_token_probs
def gather_correct_token_probs(probs, targets):
    """Return probs[i, targets[i]] for each i, shape (B,)."""
    # TODO: pick out the probability assigned to the correct next token for each batch row
    # return probs[range(len(targets)), targets]
    return np.array([probs[i][targets[i]] for i in range(len(targets))])

# Step 65 - cross_entropy_loss
import numpy as np

def cross_entropy_loss(probs, targets):
    """Mean negative log-likelihood over a batch."""
    # TODO: gather correct-token probs, take log, average the negatives
    probas = gather_correct_token_probs(probs, targets)
    _log = array_log(probas)
    loss_per_example = - _log
    loss = np.mean(loss_per_example)
    return loss

# Step 66 - derive_dlogits_on_paper
def derive_dlogits_on_paper():
    """Return a string summarizing the derivation of dL/dlogits for mean cross-entropy."""
    # TODO: return a short written derivation ending in dL/dlogits = (probs - onehot(targets)) / B
    return"""We start from the mean cross-entropy loss...
    Using softmax and negative log-likelihood...
    The gradient simplifies to:
    dL/dlogits = (probs - onehot(targets)) / B"""

# Step 67 - compute_dlogits
def compute_dlogits(probs, targets):
    """Gradient of mean cross-entropy w.r.t. logits. probs: (B,V), targets: (B,)."""
    # TODO: return dL/dlogits of shape (B, V) averaged over the batch.
    B = len(targets)
    one_hot = one_hot_encode_batch(targets, probs.shape[1])
    dlogits = (probs - one_hot) / B 
    return dlogits

# Step 68 - derive_dw_on_paper
def derive_dw_on_paper():
    """Return a short written derivation of dL/dW for the lookup-as-matmul forward."""
    # TODO: return a fixed multi-line string describing the scatter-add gradient.
    return (
        "Forward: logits = onehot(ids) @ W, equivalently logits[b] = W[ids[b]].\n"
        "Shapes: ids (B,), onehot O (B, V), W (V, D), logits (B, D), dlogits (B, D).\n"
        "Chain rule: dL/dW = O.T @ dlogits, shape (V, D).\n"
        "Since O has a single 1 per row at column ids[b], O.T @ dlogits sums rows of dlogits into rows of dW.\n"
        "Row v of dW equals the sum of dlogits[b] over all b with ids[b] == v.\n"
        "Implementation: scatter-add dlogits rows into dW at indices ids."
    )

# Step 69 - compute_dw_scatter_add
import numpy as np

def compute_dw_scatter_add(ids, dlogits, vocab_size):
    """Scatter-add dlogits rows into dW at positions given by ids."""
    # TODO: build a (vocab_size, vocab_size) dW and accumulate dlogits[b] into row ids[b].
    dw = np.zeros((vocab_size, dlogits.shape[1]))  # step 1: zeros matrix
    np.add.at(dw, ids, dlogits)             # step 2: for each id, add dlogits row into dw
    return dw                                # step 3: return

# Step 70 - sgd_update_w
import numpy as np

def sgd_update_w(w, dw, learning_rate):
    """Apply one SGD step: return w - learning_rate * dw as a new array."""
    # TODO: subtract the scaled gradient from the weights and return the new matrix
    return w - learning_rate * dw

# Step 71 - run_one_training_step
def run_one_training_step(w, ids, targets, learning_rate):
    """Run forward, loss, backward, and SGD update once. Return {'w': new_w, 'loss': float}."""
    # TODO: chain the upstream forward/loss/backward/update helpers into one step
    logits = forward_logits_lookup(w, ids)
    probs = logits_to_probs_rowwise(logits)
    loss = cross_entropy_loss(probs, targets)
    dlogits = compute_dlogits(probs, targets)
    dw =  compute_dw_scatter_add(ids, dlogits,  w.shape[0])
    new_w = sgd_update_w(w, dw, learning_rate)
    return {'w': new_w, 'loss': float(loss)}

# Step 72 - train_neural_bigram_loop
def train_neural_bigram_loop(w, data, block_size, batch_size, learning_rate, num_steps, log_every):
    """Run the neural bigram training loop and return {'w', 'loss_history'}."""
    # TODO: repeatedly sample a batch, run one training step, and log loss every log_every steps
    rng = np.random.default_rng()
    loss_history = []
    for i in range(num_steps):
        X, y = get_batch(data, block_size, batch_size, rng)
        ids = X.flatten()
        targets = y.flatten()
        result = run_one_training_step(w, ids, targets, learning_rate)
        w = result['w']
        if i % log_every == 0:
            loss_history.append(result['loss'])
    return {'w': w, 'loss_history': loss_history}

# Step 73 - sample_from_neural_bigram
def sample_from_neural_bigram(w, start_id, num_tokens, itos):
    """Generate a string by repeatedly sampling from softmax of W[id]."""
    # TODO: starting from start_id, sample num_tokens new ids and decode the full sequence...
    ids = [start_id] 
    rng = np.random.default_rng()
    vocab_size = w.shape[0]
    for i in range(num_tokens):
        logits = forward_logits_lookup(w, [ids[-1]])
        probs = logits_to_probs_rowwise(logits)
        next_token = rng.choice(vocab_size, p=probs[0])
        ids.append(next_token)
    return  decode_ids(ids, itos)

# Step 74 - linear_forward
def linear_forward(x, w):
    # TODO: compute Y = X @ W and return {'y': Y, 'cache': {'x': x, 'w': w}}.
    y = x @ w
    return {'y': y, 'cache': {'x': x, 'w': w}}

# Step 75 - derive_dx_on_paper
def derive_dx_on_paper():
    """Return notes deriving dL/dX = dY @ W.T for Y = X @ W."""
    # TODO: return a multi-line string with the derivation and shape check
    return "Y = X @ W\ndL/dX = dY @ W.T\nshapes: X (B, In), W (In, Out), dY (B, Out) -> dL/dX (B, In)"

# Step 76 - derive_linear_dw_on_paper
def derive_linear_dw_on_paper():
    """Return a string with the derivation of dL/dW for Y = X @ W."""
    # TODO: return notes that include the final identity dL/dW = X.T @ dY
    return "Y = X @ W\ndL/dW = X.T @ dY\nshapes: X (B, D_in), dY (B, D_out) -> dL/dW (D_in, D_out)\nX.T has shape (D_in, B) and dY has shape (B, D_out), so their product is (D_in, D_out)."

# Step 77 - linear_backward_dx
def linear_backward_dx(dy, cache):
    # TODO: compute the gradient of the loss w.r.t. the linear layer input X given dy and cache
    
    dL_dX = dy @ cache['w'].T
    return dL_dX

# Step 78 - linear_backward_dw
def linear_backward_dw(dy, cache):
    """Return dL/dW for a linear layer Y = X @ W."""
    # TODO: compute the weight gradient using x from cache and the upstream dy
    dL_dW = cache['x'].T @ dy
    return dL_dW

# Step 79 - bias_add_forward
def bias_add_forward(x, b):
    """Add bias vector b (D,) to every row of x (B, D).

    Returns {'y': ndarray (B, D), 'cache': {'b_shape': tuple}}.
    """
    # TODO: add b to each row of x and cache b's shape for the backward pass

    xxx = vector_matrix_broadcast_add(x, b)
    return {'y': xxx , 'cache': {'b_shape': b.shape}}

# Step 80 - bias_add_backward_db
def bias_add_backward_db(dy, cache):
    """Compute db from upstream gradient dy for y = x + b."""
    # TODO: sum the upstream gradient over the batch dimension to get db of shape (D,)
    db = np.sum(dy, axis=0)
    return db

# Step 81 - relu_forward
def relu_forward(x):
    """Apply elementwise ReLU and cache the input for backward.

    Returns a dict with keys 'y' (activated array) and 'cache' (dict with 'x').
    """
    # TODO: apply elementwise ReLU and cache the input for backward.
    return {'y': np.maximum(0, x), 'cache': {'x': x}}

# Step 82 - relu_backward
def relu_backward(dy, cache):
    """Backward pass for ReLU. cache['x'] holds the original input."""
    # TODO: return dx with gradient zeroed where the cached input was non-positive.
    return dy * (cache['x'] > 0)

# Step 83 - softmax_cross_entropy_backward
def softmax_cross_entropy_backward(probs, targets):
    """Return dL/dlogits for mean cross-entropy with softmax probs."""
    # TODO: produce the (B, V) gradient of mean cross-entropy w.r.t. logits.
    B = len(probs)
    dlogits = probs.copy()
    dlogits[range(B), targets] -= 1
    dlogits /= B
    return dlogits

# Step 84 - layernorm_forward_mean
import numpy as np

def layernorm_forward_mean(x):
    """Return the per-row mean of x with shape (B, 1)."""
    # TODO: compute the per-row mean of x, preserving the reduced axis as size 1
    D = x.shape[-1]
    _mean = sum_keepdims(x, axis=-1)
    return _mean / D

# Step 85 - layernorm_forward_variance
import numpy as np

def layernorm_forward_variance(x, mean):
    """Compute the per-row (biased) variance of x given its per-row mean.

    Args:
        x: ndarray of shape (B, D).
        mean: ndarray of shape (B, 1), the per-row mean of x.

    Returns:
        var: ndarray of shape (B, 1), the per-row variance.
    """
    # TODO: compute per-row variance using mean and return a (B, 1) array
    D = x.shape[-1]

    variance = sum_keepdims((x - mean)**2, axis=-1) / D
    return variance

# Step 86 - layernorm_forward_normalize
import numpy as np

def layernorm_forward_normalize(x, mean, var, eps):
    """Normalize each row of x to zero mean and unit variance."""
    # TODO: subtract the per-row mean and divide by sqrt(var + eps)
    normalized_row = (x - mean) / np.sqrt(var+eps)
    return normalized_row

# Step 87 - layernorm_forward_affine
def layernorm_forward_affine(x, gamma, beta, eps):
    """Run LayerNorm forward over rows of x with affine params gamma, beta."""
    # TODO: normalize each row to zero mean / unit variance, then apply gamma and beta.
    mean = layernorm_forward_mean(x)
    var = layernorm_forward_variance(x, mean)
    x_hat = layernorm_forward_normalize(x, mean , var, eps)
    y = elementwise_multiply(x_hat, gamma)
    map_  = vector_matrix_broadcast_add(y, beta)
    return {
    'y': map_,
    'cache': {
        'x': x,
        'x_hat': x_hat,
        'mean': mean,
        'var': var,
        'gamma': gamma,
        'eps': eps
    }
        }

# Step 88 - layernorm_backward_subtract_mean
import numpy as np

def layernorm_backward_subtract_mean(dy, cache):
    """Gradient through y = x - mean(x, axis=1, keepdims=True).

    dy: (B, D) upstream gradient w.r.t. the centered output.
    cache: dict with keys 'x' (B, D) and 'mean' (B,).
    Returns dx of shape (B, D).
    """
    # TODO: compute the gradient contribution of the subtract-mean op
    #mean = layernorm_forward_mean(x)
    dx = dy - layernorm_forward_mean(dy)
    return dx

# Step 89 - layernorm_backward_divide_std
import numpy as np 

def layernorm_backward_divide_std(dy, cache):
    """Propagate dy through the divide-by-std step of LayerNorm."""
    # TODO: propagate the upstream gradient through the divide-by-std step of LayerNorm
    dx = dy / np.sqrt(cache['var'] + cache['eps'])
    return dx

# Step 90 - layernorm_backward_full
import numpy as np

def layernorm_backward_full(dy, cache):
    """Full LayerNorm backward. Return {'dx', 'dgamma', 'dbeta'}."""
    # TODO: chain rule back through affine, divide-by-std, and subtract-mean.
    x = cache['x']
    x_hat = cache['x_hat']
    mean = cache['mean']
    var = cache['var']
    gamma = cache['gamma']
    eps = cache['eps']

    
    dgamma = sum_axis0(dy * x_hat)
    dbeta = sum_axis0(dy)

    D = x.shape[-1]
    std = np.sqrt(var + eps)
    dx_hat = dy * gamma
    dx = (1/std) * (dx_hat - layernorm_forward_mean(dx_hat) - x_hat * layernorm_forward_mean(dx_hat * x_hat))

    return {'dx': dx, 'dgamma': dgamma, 'dbeta': dbeta}

# Step 91 - layernorm_backward_implementation
def layernorm_backward_implementation(d_out, cache):
    # TODO: return {'dx', 'dgamma', 'dbeta'} gradients for LayerNorm given d_out and the forward cache.
    



    return layernorm_backward_full(d_out, cache)

# Step 92 - create_token_embedding
def create_token_embedding(vocab_size, d_model, scale=0.02):
    """Initialize the token embedding matrix E of shape (vocab_size, d_model)."""
    # TODO: return a (vocab_size, d_model) array of small random values controlled by scale
    return scale_w_small(np.random.randn(vocab_size, d_model), scale)

# Step 93 - token_embedding_forward
def token_embedding_forward(token_ids, embedding_matrix):
    out = embedding_matrix[token_ids]
    cache = {
        'token_ids': token_ids,
        'vocab_size': embedding_matrix.shape[0]
    }
    return out, cache

# Step 94 - token_embedding_backward
import numpy as np

def token_embedding_backward(d_out, cache):
    # TODO: scatter-add d_out into a (vocab_size, d_model) dE using cache['token_ids'].
    vocab_size = cache['vocab_size']
    d_model = d_out.shape[-1]
    token_ids = cache['token_ids']
    dE = np.zeros((vocab_size, d_model)) 
    np.add.at(dE, token_ids, d_out)
    return dE

# Step 95 - create_positional_embedding
def create_positional_embedding(block_size, d_model, scale=0.02):
    """Initialize the learned positional embedding matrix P of shape (block_size, d_model)."""
    # TODO: build a (block_size, d_model) matrix of small random values scaled by `scale`
    P = make_2d_random(block_size, d_model, None)
    P = scale_w_small(P, scale)
    return P

# Step 96 - slice_positional_embedding
import numpy as np

def slice_positional_embedding(positional_matrix, seq_len):
    """Return the first seq_len rows of the positional embedding matrix."""
    # TODO: return the leading seq_len rows of positional_matrix as a (seq_len, d_model) array.
    return P[:seq_len]

# Step 97 - add_token_and_positional_embeddings
def add_token_and_positional_embeddings(token_emb, pos_emb):
    """Sum token embeddings (B,T,d_model) and positional embeddings (T,d_model)."""
    # TODO: combine token and positional embeddings into a single (B,T,d_model) tensor
    return token_emb + pos_emb

# Step 98 - embedding_sum_backward
def embedding_sum_backward(d_out):
    """Backprop through H = token_emb + pos_emb (with broadcasting over batch)."""
    # TODO: route d_out to both branches, reducing over the batch axis for pos_emb.
    return {
        "d_token_emb": d_out,
        "d_pos_emb": np.sum(d_out, axis=0)
    }

# Step 99 - create_qkv_projections
def create_qkv_projections(d_model, d_head, scale=0.02):

    Wq = scale_w_small(
        make_2d_random(d_model, d_head, seed=0),
        scale
    )

    Wk = scale_w_small(
        make_2d_random(d_model, d_head, seed=1),
        scale
    )

    Wv = scale_w_small(
        make_2d_random(d_model, d_head, seed=2),
        scale
    )

    return {
        "Wq": Wq,
        "Wk": Wk,
        "Wv": Wv
    }

# Step 100 - compute_query
import numpy as np

def compute_query(x, w_q):
    return x @ w_q

# Step 101 - compute_key
import numpy as np 

def compute_key(x, w_k):
    return x @ w_k

# Step 102 - compute_value
def compute_value(x, w_v):
    return x @ w_v

# Step 103 - compute_attention_scores
import numpy as np

def compute_attention_scores(q, k):
    return q @ k.transpose(0, 2, 1)

# Step 104 - scale_attention_scores
import numpy as np

def scale_attention_scores(scores, d_head):
    """Rescale (B, T, T) attention scores by a function of d_head."""
    # TODO: rescale the scores so their variance does not grow with d_head.
    return scores / np.sqrt(d_head)

# Step 105 - build_causal_mask
import numpy as np

def build_causal_mask(seq_len):
    """Return a (seq_len, seq_len) boolean lower-triangular mask."""
    # TODO: build a (T, T) boolean mask where True marks allowed (query, key) pairs
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))

# Step 106 - apply_causal_mask
import numpy as np

def apply_causal_mask(scaled_scores, causal_mask):
    """Replace future positions in scaled_scores with -inf using causal_mask."""
    # TODO: return a (B,T,T) array where positions with causal_mask False are -inf...
    return np.where(causal_mask, scaled_scores, -np.inf)

# Step 107 - softmax_attention_weights
import numpy as np

def softmax_attention_weights(masked_scores):
    shifted = masked_scores - np.max(masked_scores, axis=-1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

# Step 108 - attention_weighted_values
import numpy as np

def attention_weighted_values(attn, v):
    """Combine attention weights with values: out = attn @ V.

    attn: (B, T, T) softmaxed attention weights
    v:    (B, T, d_head) value vectors
    returns: (B, T, d_head)
    """
    # TODO: mix the value vectors using the attention weights
    return attn @ v

# Step 109 - apply_output_projection
import numpy as np

def apply_output_projection(attn_out, w_o):
    """Project attention output (B,T,d_head) through Wo (d_head,d_model)."""
    # TODO: return attn_out projected through w_o to shape (B, T, d_model)
    return attn_out @ w_o

# Step 110 - output_projection_backward
def output_projection_backward(d_proj, cache):
    attn_out = cache["attn_out"]
    w_o = cache["w_o"]

    d_attn_out = d_proj @ w_o.T

    dw_o = np.einsum(
        "btd,btm->dm",
        attn_out,
        d_proj
    )

    return {
        "d_attn_out": d_attn_out,
        "dw_o": dw_o
    }

# Step 111 - attention_value_backward
def attention_value_backward(d_attn_out, cache):
    attn = cache["attn"]
    v = cache["v"]

    d_attn = d_attn_out @ v.transpose(0, 2, 1)
    d_v = attn.transpose(0, 2, 1) @ d_attn_out

    return {
        "d_attn": d_attn,
        "d_v": d_v
    }

# Step 112 - masked_softmax_backward
def masked_softmax_backward(d_attn, cache):
    attn = cache["attn"]
    causal_mask = cache["causal_mask"]

    dot = np.sum(
        d_attn * attn,
        axis=-1,
        keepdims=True
    )

    d_scores = attn * (d_attn - dot)

    d_scores = np.where(
        causal_mask,
        d_scores,
        0.0
    )

    return d_scores

# Step 113 - scale_scores_backward
import numpy as np

def scale_scores_backward(d_scaled_scores, d_head):
    return d_scaled_scores / np.sqrt(d_head)

# Step 114 - qk_scores_backward
def qk_scores_backward(d_scores, cache):
    q = cache["q"]
    k = cache["k"]

    d_q = d_scores @ k

    d_k = (
        q.transpose(0,2,1)
        @
        d_scores
    ).transpose(0,2,1)

    return {
        "d_q": d_q,
        "d_k": d_k
    }

# Step 115 - qkv_projection_backward
def qkv_projection_backward(d_q, d_k, d_v, cache):
    x = cache["x"]
    w_q = cache["w_q"]
    w_k = cache["w_k"]
    w_v = cache["w_v"]

    dx = (
        d_q @ w_q.T
        + d_k @ w_k.T
        + d_v @ w_v.T
    )

    dw_q = np.einsum("btd,bth->dh", x, d_q)
    dw_k = np.einsum("btd,bth->dh", x, d_k)
    dw_v = np.einsum("btd,bth->dh", x, d_v)

    return {
        "dx": dx,
        "dw_q": dw_q,
        "dw_k": dw_k,
        "dw_v": dw_v
    }

# Step 116 - choose_attention_head_config
def choose_attention_head_config(d_model, n_heads):
    if d_model % n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")

    return {
        "n_heads": n_heads,
        "d_head": d_model // n_heads,
        "d_model": d_model
    }

# Step 117 - create_multihead_qkv_projections
def create_multihead_qkv_projections(d_model, scale=0.02):
    return {
        "Wq": scale_w_small(
            make_2d_random(d_model, d_model, seed=0),
            scale
        ),
        "Wk": scale_w_small(
            make_2d_random(d_model, d_model, seed=1),
            scale
        ),
        "Wv": scale_w_small(
            make_2d_random(d_model, d_model, seed=2),
            scale
        )
    }

# Step 118 - create_multihead_output_projection
def create_multihead_output_projection(d_model, scale=0.02):
    Wo = make_2d_random(d_model, d_model, seed=0)
    Wo = scale_w_small(Wo, scale)
    return Wo

# Step 119 - reshape_to_heads
def reshape_to_heads(x, n_heads, d_head):
    B, T, _ = x.shape
    return x.reshape(B, T, n_heads, d_head)

# Step 120 - transpose_heads_to_front
import numpy as np 


def transpose_heads_to_front(x):
    return np.ascontiguousarray(x.transpose(0, 2, 1, 3))

# Step 121 - get_multihead_n_heads
def get_multihead_n_heads(config):
    """Return the number of attention heads from a multi-head config."""
    return config["n_heads"]

# Step 122 - get_multihead_sequence_length
def get_multihead_sequence_length(x):
    """Return the sequence length T from an activation tensor."""
    shape = get_array_shape(x)
    return shape[1]

# Step 123 - compute_d_head
def compute_d_head(d_model, n_heads):
    """Return the dimension of each attention head."""

    if d_model % n_heads != 0:
        raise ValueError("n_heads must evenly divide d_model")

    return d_model // n_heads

# Step 124 - multihead_masked_softmax_scores
def multihead_masked_softmax_scores(scores, mask):
    masked_scores = apply_causal_mask(scores, mask)

    weights = masked_scores.copy()

    B, n_heads, T, _ = weights.shape

    for b in range(B):
        for h in range(n_heads):
            weights[b, h] = stable_softmax_2d_rowwise(weights[b, h])

    return weights

# Step 125 - multihead_weighted_sum
import numpy as np

def multihead_weighted_sum(weights, v_heads):
    """
    weights : (B, n_heads, T, T)
    v_heads : (B, n_heads, T, d_head)

    returns:
        (B, n_heads, T, d_head)
    """
    return np.matmul(weights, v_heads)

# Step 126 - transpose_heads_to_back
def transpose_heads_to_back(x):
    return np.transpose(x, (0, 2, 1, 3))

# Step 127 - get_multihead_output_sequence_length
def get_multihead_output_sequence_length(x_heads_back):
    return int(x_heads_back.shape[1])

# Step 128 - merge_heads_to_d_model
import numpy as np

def merge_heads_to_d_model(x):
    B, T, H, D = x.shape
    return x.reshape(B, T, H * D)

# Step 129 - multihead_output_projection_forward
def multihead_output_projection_forward(merged, w_out, b_out):
    linear = linear_forward(merged, w_out)
    biased = bias_add_forward(linear["y"], b_out)

    return {
        "out": biased["y"],
        "cache": {
            "merged": merged,
            "w_out": w_out,
        }
    }

# Step 130 - multihead_reshape_transpose_backward
def multihead_reshape_transpose_backward(d_merged, shape_info):
    x = reshape_to_heads(
        d_merged,
        shape_info["n_heads"],
        shape_info["d_head"],
    )

    return transpose_heads_to_front(x)

# Step 131 - ffn_linear_one_forward
def ffn_linear_one_forward(x, w1, b1):
    linear = linear_forward(x, w1)
    biased = bias_add_forward(linear["y"], b1)

    return {
        "h1": biased["y"],
        "cache": {
            "x": x,
            "w1": w1,
        }
    }

# Step 132 - ffn_activation_forward
def ffn_activation_forward(h1):
    relu = relu_forward(h1)

    return (
        relu["y"],
        {
            "h1": h1
        }
    )

# Step 133 - ffn_linear_two_forward
def ffn_linear_two_forward(a1, w2, b2):
    linear = linear_forward(a1, w2)
    biased = bias_add_forward(linear["y"], b2)

    return {
        "h2": biased["y"],
        "cache": {
            "a1": a1,
            "w2": w2,
        }
    }

# Step 134 - ffn_backward
def ffn_backward(d_out, cache):
    x = cache["x"]
    w1 = cache["w1"]
    h1 = cache["h1"]
    a1 = cache["a1"]
    w2 = cache["w2"]

    x_flat = x.reshape(-1, x.shape[-1])
    h1_flat = h1.reshape(-1, h1.shape[-1])
    a1_flat = a1.reshape(-1, a1.shape[-1])
    d_out_flat = d_out.reshape(-1, d_out.shape[-1])

    linear2_cache = {
        "x": a1_flat,
        "w": w2,
    }

    da1 = linear_backward_dx(d_out_flat, linear2_cache)
    dw2 = linear_backward_dw(d_out_flat, linear2_cache)
    db2 = bias_add_backward_db(d_out_flat, {})

    dh1 = relu_backward(
        da1,
        {"x": h1_flat}
    )

    linear1_cache = {
        "x": x_flat,
        "w": w1,
    }

    dx = linear_backward_dx(dh1, linear1_cache)
    dw1 = linear_backward_dw(dh1, linear1_cache)
    db1 = bias_add_backward_db(dh1, {})

    dx = dx.reshape(x.shape)

    return {
        "dx": dx,
        "dw1": dw1,
        "db1": db1,
        "dw2": dw2,
        "db2": db2,
    }

# Step 135 - residual_forward
def residual_forward(x, sublayer_out):
    """Return x + sublayer_out for a residual connection."""
    # TODO: add the sublayer output to its input to form a residual connection.
    return x + sublayer_out

# Step 136 - residual_backward
def residual_backward(d_y):
    return d_y.copy(), d_y.copy()

# Step 137 - pre_layernorm_sublayer_forward
def pre_layernorm_sublayer_forward(x, ln_params, sublayer_fn, sublayer_params):
    # LayerNorm
    ln_out = layernorm_forward_affine(
        x,
        ln_params["gamma"],
        ln_params["beta"],
        ln_params.get("eps", 1e-5),
    )

    # Sublayer
    sublayer_out = sublayer_fn(ln_out["y"], sublayer_params)

    # Residual
    y = residual_forward(x, sublayer_out["y"])

    cache = {
        "x": x,
        "ln_cache": ln_out["cache"],
        "sublayer_cache": sublayer_out["cache"],
    }

    return {
        "y": y,
        "cache": cache,
    }

# Step 138 - transformer_block_forward
def transformer_block_forward(x, block_params):
    """
    Run one pre-LN Transformer block.

    Structure:
        x
        │
        ├── LayerNorm
        │
        ├── Multi-Head Self-Attention
        │
        └── Residual
             │
             ▼
        h
             │
             ├── LayerNorm
             │
             ├── FFN
             │
             └── Residual
                  │
                  ▼
                  y
    """

    # =========================================================
    # 1. Attention branch
    # =========================================================

    # LayerNorm before Attention
    ln1 = layernorm_forward_affine(
        x,
        block_params["ln1"]["gamma"],
        block_params["ln1"]["beta"],
        block_params["ln1"]["eps"],
    )

    # Multi-Head Self-Attention
    attn = multihead_attention_forward(
        ln1["y"],
        block_params["attn"],
    )

    # Residual connection
    attn_branch = residual_forward(
        x,
        attn["y"],
    )

    h = attn_branch["y"]

    # =========================================================
    # 2. FFN branch
    # =========================================================

    # LayerNorm before FFN
    ln2 = layernorm_forward_affine(
        h,
        block_params["ln2"]["gamma"],
        block_params["ln2"]["beta"],
        block_params["ln2"]["eps"],
    )

    # Feed-Forward Network
    ffn = ffn_forward(
        ln2["y"],
        block_params["ffn"],
    )

    # Residual connection
    ffn_branch = residual_forward(
        h,
        ffn["y"],
    )

    y = ffn_branch["y"]

    # =========================================================
    # 3. Return output + cache
    # =========================================================

    return {
        "y": y,
        "cache": {
            "attn_branch": {
                "ln": ln1,
                "sublayer": attn,
                "residual": attn_branch,
            },
            "ffn_branch": {
                "ln": ln2,
                "sublayer": ffn,
                "residual": ffn_branch,
            },
        },
    }

# Step 139 - transformer_block_backward
def transformer_block_backward(d_y, cache, block_params):
    cache = _complete_block_cache(cache, block_params)

    # -------------------------
    # FFN branch
    # -------------------------

    ffn_branch = cache['ffn_branch']

    d_z2, ffn_grads = _ffn_sublayer_backward(
        d_y,
        ffn_branch['sublayer_cache'],
        block_params['ffn']
    )

    d_h1_from_ffn, d_gamma2, d_beta2 = layernorm_backward_affine(
        d_z2,
        ffn_branch['ln_cache']
    )

    # Residual skip contribution
    d_h1 = d_y + d_h1_from_ffn

    # -------------------------
    # Attention branch
    # -------------------------

    attn_branch = cache['attn_branch']

    d_z1, attn_grads = _attn_sublayer_backward(
        d_h1,
        attn_branch['sublayer_cache'],
        block_params['attn']
    )

    d_x_from_attn, d_gamma1, d_beta1 = layernorm_backward_affine(
        d_z1,
        attn_branch['ln_cache']
    )

    # Residual skip contribution
    d_x = d_h1 + d_x_from_attn

    # -------------------------
    # Assemble gradients
    # -------------------------

    grads = {
        'ln1': {
            'gamma': d_gamma1,
            'beta': d_beta1,
        },
        'ln2': {
            'gamma': d_gamma2,
            'beta': d_beta2,
        },
        'attn': attn_grads,
        'ffn': ffn_grads,
    }

    return d_x, grads

# Step 140 - stack_transformer_blocks (not yet solved)
# TODO: implement

# Step 141 - forward_through_all_blocks (not yet solved)
# TODO: implement

# Step 142 - backward_through_all_blocks (not yet solved)
# TODO: implement

# Step 143 - final_layernorm_forward (not yet solved)
# TODO: implement

# Step 144 - lm_head_linear_forward (not yet solved)
# TODO: implement

# Step 145 - full_model_forward (not yet solved)
# TODO: implement

# Step 146 - full_model_backward (not yet solved)
# TODO: implement

# Step 147 - initialize_adam_moments
def initialize_adam_moments(model_params):
    if isinstance(model_params, dict):
        m = {}
        v = {}

        for key, value in model_params.items():
            m[key], v[key] = initialize_adam_moments(value)

        return m, v

    elif isinstance(model_params, list):
        m = []
        v = []

        for value in model_params:
            m_value, v_value = initialize_adam_moments(value)
            m.append(m_value)
            v.append(v_value)

        return m, v

    elif isinstance(model_params, np.ndarray):
        return np.zeros_like(model_params), np.zeros_like(model_params)

    else:
        return model_params, model_params

# Step 148 - initialize_adam_step_counter
def initialize_adam_step_counter():
    """Return the initial Adam step counter t."""
    # TODO: return the starting value of the Adam time-step counter.
    return 0

# Step 149 - adam_increment_step
def adam_increment_step(t):
    """Return t + 1 so Adam bias correction sees a positive step."""
    # TODO: return the next Adam step counter value
    return t + 1

# Step 150 - adam_update_first_moment
import numpy as np

def adam_update_first_moment(m, grad, beta1):
    """Return the updated Adam first-moment estimate."""
    # TODO: blend previous moment m with current grad using decay beta1
    return beta1 * m + (1 - beta1) * grad

# Step 151 - adam_update_second_moment (not yet solved)
# TODO: implement

# Step 152 - adam_bias_correction (not yet solved)
# TODO: implement

# Step 153 - adam_parameter_update (not yet solved)
# TODO: implement

# Step 154 - wire_full_training_loop (not yet solved)
# TODO: implement

# Step 155 - logging_and_validation_loss (not yet solved)
# TODO: implement

# Step 156 - encode_prompt (not yet solved)
# TODO: implement

# Step 157 - crop_context_to_block_size (not yet solved)
# TODO: implement

# Step 158 - forward_to_get_logits (not yet solved)
# TODO: implement

# Step 159 - take_last_position_logits (not yet solved)
# TODO: implement

# Step 160 - apply_temperature (not yet solved)
# TODO: implement

# Step 161 - top_k_filter (not yet solved)
# TODO: implement

# Step 162 - softmax_to_probs (not yet solved)
# TODO: implement

# Step 163 - sample_one_token (not yet solved)
# TODO: implement

# Step 164 - append_token_to_sequence (not yet solved)
# TODO: implement

# Step 165 - generation_loop_for_n_steps (not yet solved)
# TODO: implement

# Step 166 - decode_final_sequence (not yet solved)
# TODO: implement

