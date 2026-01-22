import jax
import jax.numpy as jnp
import numpy as np
import struct
import time
from os import path

# --- 1. MNIST Loader (Reshaped for CNN) ---
def load_mnist(lbl_path, img_path):
    with open(lbl_path, 'rb') as f:
        magic, n = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    with open(img_path, 'rb') as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        # Reshape to (Batch, Height, Width, Channels) for CNN
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, 28, 28, 1)
    return jnp.array(images / 255.0), jnp.array(labels)

# --- 2. Model & Functions ---
def init_cnn(key):
    k1, k2 = jax.random.split(key)
    
    # Scala: Conv2DLayer.Params (3x3 kernel, 1 in-channel, 16 out-channels)
    # JAX shape: (Kernel H, Kernel W, In, Out)
    conv_w = jax.random.normal(k1, (3, 3, 1, 16)) * 0.1
    conv_b = jnp.zeros((16,))
    
    # Scala: LinearLayer.Params (14*14*16 input -> 10 output)
    # Note: 28x28 with Stride 2 = 14x14 spatial output
    fc_w = jax.random.normal(k2, (14 * 14 * 16, 10)) * 0.1
    fc_b = jnp.zeros((10,))
    
    return {
        'conv': {'w': conv_w, 'b': conv_b},
        'fc':   {'w': fc_w,   'b': fc_b}
    }

def forward(params, x):
    # 1. Convolution
    # dimension_numbers=('NHWC', 'HWIO', 'NHWC') maps:
    # N=Batch, H=Height, W=Width, C=Channel
    # Kernel: H, W, In, Out
    conv_out = jax.lax.conv_general_dilated(
        x, 
        params['conv']['w'], 
        window_strides=(2, 2),  # Stride 2 (matches Scala code)
        padding='SAME',         # Padding SAME (matches Scala code)
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
    
    # Add bias (broadcast to H, W)
    conv_out = conv_out + params['conv']['b'][None, None, None, :]
    
    # 2. Activation
    hidden = jax.nn.relu(conv_out)
    
    # 3. Flatten (Batch, 14*14*16)
    flattened = hidden.reshape(hidden.shape[0], -1)
    
    # 4. Fully Connected
    return jnp.dot(flattened, params['fc']['w']) + params['fc']['b']

def loss_fn(params, x, y):
    logits = forward(params, x)
    log_probs = logits - jax.nn.logsumexp(logits, axis=1, keepdims=True)
    return -jnp.mean(jnp.sum(log_probs * jax.nn.one_hot(y, 10), axis=1))

def accuracy(params, x, y):
    return jnp.mean(jnp.argmax(forward(params, x), axis=1) == y)

# --- 3. Training Update ---
@jax.jit
def step(params, x, y, lr=0.01):
    grads = jax.grad(loss_fn)(params, x, y)
    return jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)

# --- 4. Main Execution ---
def main():
    key = jax.random.PRNGKey(42)
    
    # Paths (adjust as needed)
    x_train, y_train = load_mnist("examples/data/train-labels-idx1-ubyte", "examples/data/train-images-idx3-ubyte")
    x_test, y_test = load_mnist("examples/data/t10k-labels-idx1-ubyte", "examples/data/t10k-images-idx3-ubyte")
    
    params = init_cnn(key)
    
    # Matching Scala hyperparameters roughly
    batch_size = 128 
    
    num_batches = len(x_train) // batch_size
    limit = num_batches * batch_size
    
    train_batches = list(zip(
        jnp.split(x_train[:limit], num_batches),
        jnp.split(y_train[:limit], num_batches),
    ))

    print(f"Training CNN on {limit} samples ({num_batches} batches)...")

    for epoch in range(50): # Matching Scala epoch count
        start = time.time()
        
        for x_batch, y_batch in train_batches:
            params = step(params, x_batch, y_batch, lr=0.01)

        # Evaluate periodically
        if epoch % 1 == 0:
            test_acc = accuracy(params, x_test, y_test)
            # Avoid calculating full train accuracy every step to save time, just like Scala log suggests
            print(f"Epoch {epoch}: Test Acc {test_acc:.2%} ({int((time.time()-start)*1000)}ms)")

if __name__ == "__main__":
    main()