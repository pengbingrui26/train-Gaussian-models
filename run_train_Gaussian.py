import numpy
import jax
import jax.numpy as jnp

from train_Gaussian import Gaussian_sampler, make_loss, \
                          make_grad_loss, logp, optimize_sigma, \
                           Gaussian_sampler_new, scan_sigma

def test_Gaussian_sampler():
    batch = 4
    n = 1000
    shape = (n, )
    sigma = 2.
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, batch)
    sampler_vmapped = jax.vmap(Gaussian_sampler, in_axes = (None, None, 0), out_axes = 0)
    #out = sampler_vmapped(shape, sigma, keys)
    x = Gaussian_sampler(shape, sigma, key)
    #print(x)
    x_square = x**2
    print( x_square.mean() - x.mean()**2 )
 

def test_Gaussian_sampler_new():
    batch = 5
    n = 4000
    shape = (batch, n)
    sigma = 2.
    key = jax.random.PRNGKey(42)
    x = Gaussian_sampler_new(shape, sigma, key)
    #print(x)
    x_square = x**2
    print( x_square.mean() - x.mean()**2 )
 


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
    batch = 5
    beta = 4.
    key = jax.random.PRNGKey(42)
    sigma = 1/jnp.sqrt(beta)
    x = Gaussian_sampler((batch, ), sigma, key)
    grad, loss = make_grad_loss(x, beta, sigma)
    #F_mean = make_loss( x, beta, sigma )
    #print('loss:', loss)
 

def test_optimize_sigma():
    batch = 2000
    beta = 4.
    key = jax.random.PRNGKey(42)
    nstep = 320
    learning_rate = 1e-3
    optimize_sigma( batch, beta, key, nstep, learning_rate )

    #import pickle as pk
    
def test_scan_sigma():
    batch = 5000
    beta = 4.
    key = jax.random.PRNGKey(42)
    scan_sigma( batch, beta, key )



# run test ==========================================================

#test_Gaussian_sampler()
#test_Gaussian_sampler_new()
#test_logp()
#test_make_loss()
#test_make_grad_loss()
#test_optimize_sigma()
test_scan_sigma()

