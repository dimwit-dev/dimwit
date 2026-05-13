# JAX helper functions for dimwit library

import jax
import jax.numpy as jnp

import builtins
builtins.jax = jax
builtins.jnp = jnp

def wrap_fn(f):
    """Wrap a ScalaPy callback as a plain Python callable."""
    def python_wrapper(*a, **kw):
        return f(*a, **kw)
    return python_wrapper

def wrap(jax_transform, f, kwargs=None):
    """
    Generic wrapper that shields a ScalaPy function from JAX introspection,
    then applies the given JAX transform.

    Usage:
        wrap(jax.grad, f)
        wrap(jax.vmap, f, kwargs={"in_axes": 0})
        wrap(jax.jit, f, kwargs={"donate_argnums": (0,)})
        wrap(jax.jacfwd, f)
    """
    if kwargs:
        return jax_transform(wrap_fn(f), **kwargs)
    return jax_transform(wrap_fn(f))

def scan(f, init, xs, kwargs=None):
    """
    Wrapper for jax.lax.scan.
    
    Args:
        f: The loop function.
        init: The initial state.
        xs: The sequence to scan over.
        kwargs: Additional arguments like 'length' or 'reverse'.
    """
    if kwargs:
        return jax.lax.scan(wrap_fn(f), init, xs, **kwargs)
    return jax.lax.scan(wrap_fn(f), init, xs)

def vmap(f, dims):
    return wrap(jax.vmap, f, kwargs={"in_axes": dims})
           
def zipvmap(f, dims):
    def python_wrapper(*args):
        return f(args)
    return lambda jax_inputs_tuple: jax.vmap(python_wrapper, in_axes=dims)(*jax_inputs_tuple)

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
    in_axes = (dims, dims) if isinstance(dims, int) else dims
    return wrap(jax.vmap, f, kwargs={"in_axes": in_axes})

def grad(f):
    return wrap(jax.grad, f)

def value_and_grad(f):
    return wrap(jax.value_and_grad, f)

def jacfwd(f):
    return wrap(jax.jacfwd, f)

def jacrev(f):
    return wrap(jax.jacrev, f)

def jacobian(f):
    return wrap(jax.jacobian, f)

def jit(f):
    return wrap(jax.jit, f)

def jit_fn(f, jit_kwargs=None):
    return wrap(jax.jit, f, kwargs=jit_kwargs)
