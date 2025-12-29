# JAX helper functions for dimwit library

import jax
import jax.numpy as jnp
from jax import vmap

import builtins
builtins.jax = jax
builtins.jnp = jnp

def vmap(f, dims):
    """
    Applies a function `f` to a tensor using JAX's vmap functionality.
    
    It is wrapped in a Python function to ensure that the function, as otherwise
    jax will crash upon inspection.
    """
                
    # Wrap the ScalaPy function in a pure Python wrapper
    def python_wrapper(x):
        return f(x)
            
    # Create vmap with the wrapper
    return jax.vmap(python_wrapper, in_axes=dims)
            
def apply_over_axes(f, axis):
    """
    Applies a function `f` over specified axes using JAX's vmap functionality.
    
    Args:
        f: Function that takes one argument (x)
        axis: Axis or tuple of axes to map over
    
    It is wrapped in a Python function to ensure that the function, as otherwise
    jax will crash upon inspection.
    """
                
    # Wrap the ScalaPy function in a pure Python wrapper
    def python_wrapper(x):
        return f(x)
            
    # Create vmap with the wrapper
    return jnp.apply_over_axes(python_wrapper, axis)

def vmap2(f, dims):
    """
    Applies a function `f` to two tensors using JAX's vmap functionality.
    
    Args:
        f: Function that takes two arguments (x, y)
        dims: Either an integer (same axis for both inputs) or tuple (axis1, axis2)
    
    It is wrapped in a Python function to ensure that the function, as otherwise
    jax will crash upon inspection.
    """
                
    # Wrap the ScalaPy function in a pure Python wrapper
    def python_wrapper(x, y):
        return f(x, y)
    
    # Handle dims parameter - can be int or tuple
    if isinstance(dims, int):
        # Same axis for both inputs
        in_axes = (dims, dims)
    else:
        # Different axes for each input
        in_axes = dims
            
    # Create vmap with the wrapper
    return jax.vmap(python_wrapper, in_axes=in_axes)



def grad(f):
    """
    Computes the gradient of a function `f` with respect to its arguments.
    
    This is a simple wrapper around JAX's grad function.
    Only works for scalar-output functions.
    """
    from jax import grad as jax_grad
    def python_wrapper(*args):
        # Remove debug print that might cause issues
        return f(*args)
    
    return jax_grad(python_wrapper)

def value_and_grad(f):
    """
    Computes both the value and gradient of a function `f` with respect to its arguments.
    
    This is more efficient than computing value and gradient separately.
    Only works for scalar-output functions.
    """
    from jax import value_and_grad as jax_value_and_grad
    def python_wrapper(*args):
        return f(*args)
    
    return jax_value_and_grad(python_wrapper)

def jacfwd(f):
    """
    Computes the Jacobian of a function `f` using forward-mode differentiation.
    
    Works for vector-output functions. Efficient when output dimension > input dimension.
    """
    from jax import jacfwd as jax_jacfwd
    def python_wrapper(x):
        return f(x)
    
    return jax_jacfwd(python_wrapper)

def jacrev(f):
    """
    Computes the Jacobian of a function `f` using reverse-mode differentiation.
    
    Works for vector-output functions. Efficient when input dimension > output dimension.
    """
    from jax import jacrev as jax_jacrev
    def python_wrapper(x):
        return f(x)
    
    return jax_jacrev(python_wrapper)

def jacobian(f):
    from jax import jacobian as jax_jacobian
    def python_wrapper(x):
        return f(x)
    return jax_jacobian(python_wrapper)

def jit(f):
    """
    Just-in-time compiles a function for faster execution.
    
    The first call will be slower due to compilation, but subsequent calls
    with the same shapes will be much faster.
    """
    from jax import jit as jax_jit
    def python_wrapper(*args):
        return f(*args)
    
    return jax_jit(python_wrapper)

def jit_fn(f):
    """
    Universal JIT wrapper that works with any function.
    Simply wraps the function in a Python wrapper and JIT compiles it.
    
    This is the simplest and most flexible approach - works with:
    - Regular functions
    - vmap'ed functions
    - grad functions
    - Any combination
    
    Args:
        f: Any function to JIT compile
    
    Returns:
        JIT-compiled version of the function
    """
    from jax import jit as jax_jit
    def python_wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    
    return jax_jit(python_wrapper)
