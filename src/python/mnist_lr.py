import jax
import jax.numpy as jnp
import numpy as np
import struct
import time
from os import path

# --- 1. Minimal MNIST Loader (No heavy dependencies) ---
def load_mnist(lbl_path, img_path):
    with open(lbl_path, 'rb') as f:
        magic, n = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    with open(img_path, 'rb') as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, 784)
    return jnp.array(images / 255.0), jnp.array(labels)

# --- 2. Model & Functions ---
def init_mlp(key, sizes):
    keys = jax.random.split(key, len(sizes) - 1)
    return [
        {'w': jax.random.normal(k, (si, so)) * 0.1, 'b': jnp.zeros(so)}
        for k, si, so in zip(keys, sizes[:-1], sizes[1:])
    ]

def forward(params, x):
    hidden = jax.nn.relu(jnp.dot(x, params[0]['w']) + params[0]['b'])
    return jnp.dot(hidden, params[1]['w']) + params[1]['b']

def loss_fn(params, x, y):
    logits = forward(params, x)
    log_probs = logits - jax.nn.logsumexp(logits, axis=1, keepdims=True)
    return -jnp.mean(jnp.sum(log_probs * jax.nn.one_hot(y, 10), axis=1))

def accuracy(params, x, y):
    return jnp.mean(jnp.argmax(forward(params, x), axis=1) == y)

# --- 3. Training Update
@jax.jit
def step(params, x, y, lr=0.05):
    grads = jax.grad(loss_fn)(params, x, y)
    return jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)

# --- 4. Main Execution ---
def main():
    key = jax.random.PRNGKey(42)
    x_train, y_train = load_mnist("examples/data/train-labels-idx1-ubyte", "examples/data/train-images-idx3-ubyte")
    x_test, y_test = load_mnist("examples/data/t10k-labels-idx1-ubyte", "examples/data/t10k-images-idx3-ubyte")
    
    params = init_mlp(key, [784, 128, 10])
    batch_size = 512
    
    num_batches = len(x_train) // batch_size
    limit = num_batches * batch_size
    
    train_batches = list(zip(
        jnp.split(x_train[:limit], num_batches),
        jnp.split(y_train[:limit], num_batches),
    ))

    print(f"Training on {limit} samples ({num_batches} batches)...")

    for epoch in range(100):
        start = time.time()
        
        for x_batch, y_batch in train_batches:
            params = step(params, x_batch, y_batch)

        train_acc = accuracy(params, x_train, y_train)
        test_acc = accuracy(params, x_test, y_test)
        
        print(f"Epoch {epoch}: Test {test_acc:.2%}, Train {train_acc:.2%} ({int((time.time()-start)*1000)}ms)")

if __name__ == "__main__":
    main()