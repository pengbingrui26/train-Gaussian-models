import numpy
import jax
import jax.numpy as jnp

from train_Gaussian import Gaussian_sampler, make_loss, \
                          make_grad_loss, logp, optimize_sigma


def test_Gaussian_sampler():
    batch = 7
    n = 4
    shape = (batch, n)
    sigma = 2.
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, batch)
    #sampler_vmapped = jax.vmap(Gaussian_sampler, in_axes = (0, None, 0), out_axes = 0)
    #out = sampler_vmapped(shape, sigma, keys)
    out = Gaussian_sampler(shape, sigma, key)
    print(out)
 


def test_logp():
    x = jnp.array( [ 1, 2, 3 ] )    
    sigma = 1.
    print(logp(x, sigma))
    print(logp(1, sigma))
    print(logp(2, sigma))
    print(logp(3, sigma))

def test_make_loss():
    batch = 60
    n = 10
    beta = 4.
    key = jax.random.PRNGKey(42)
    sigma = 1/jnp.sqrt(beta)
    x = Gaussian_sampler((batch, n), sigma, key)
    F_mean = make_loss( x, beta, sigma )
    print('F_mean:', F_mean)
 
def test_make_grad_loss():
    batch = 20
    n = 1000
    beta = 4.
    key = jax.random.PRNGKey(42)
    sigma = 1/jnp.sqrt(beta)
    x = Gaussian_sampler((batch, n), sigma, key)
    grad, loss = make_grad_loss(x, beta, sigma)
    #F_mean = make_loss( x, beta, sigma )
    print('loss:', loss)
 

def test_optimize_sigma():
    batch = 5000
    beta = 4.
    key = jax.random.PRNGKey(42)
    nstep = 100
    learning_rate = 1e-2
    optimize_sigma( batch, beta, key, nstep, learning_rate )


#test_Gaussian_sampler()
#test_logp()
#test_make_loss()
#test_make_grad_loss()
test_optimize_sigma()

