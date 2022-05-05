import numpy
import jax
import jax.numpy as jnp

from train_Gaussian import Gaussian_sampler, make_loss, \
                          make_grad_loss, logp, optimize_sigma, \
                           Gaussian_sampler_new, scan_sigma


def test_std():
    aa = jnp.array([ [1,2], [3,4] ])
    aa_square = aa**2
    aa_std = jnp.sqrt(aa_square.mean() - aa.mean()**2)
    assert jnp.allclose(aa_std, aa.std())

def test_Gaussian_sampler():
    batch = 10000
    shape = (batch, )
    sigma = 2.
    key = jax.random.PRNGKey(42)
    x = Gaussian_sampler(shape, sigma, key)
    print(x.mean())
    mu = 0.
    assert abs(x.mean() - mu) < 1e-1
    x_square = x**2
    std_square = x_square.mean() - x.mean()**2
    print(std_square)
    assert abs(std_square - sigma**2) < 1e-1
 

def test_gradient_and_std():
    batch = 5000
    beta = 4.
    key = jax.random.PRNGKey(42)
    sigma = 1/jnp.sqrt(beta) 
    x = Gaussian_sampler((batch, ), sigma, key)
    grad, loss, std = make_grad_loss(x, beta, sigma)
    assert abs(grad) < 1e-4
    assert abs(std) < 1e-4

test_std()
#test_Gaussian_sampler()

