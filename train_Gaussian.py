import numpy as np
import jax 
import jax.numpy as jnp


def Gaussian_sampler(shape, sigma, key):
    #print('sigma', sigma)
    x = jax.random.normal(key = key, shape = shape)
    return sigma * x


def Gaussian_sampler_new(shape, sigma, key):
    (batch, n) = shape
    keys = jax.random.split(key, batch)
    sampler_vmapped = jax.vmap(Gaussian_sampler, in_axes = (None, None, 0), out_axes = 0)
    x = sampler_vmapped((n,), sigma, keys)
    x = jnp.reshape(x, (batch*n,))
    return x


def logp(x, sigma):
    p = 1/(np.sqrt(2*jnp.pi)*sigma) * jnp.exp(-x**2/(2*sigma**2))
    return jnp.log(p)
    

def make_loss(x, beta, sigma):
    ln_p = logp(x, sigma)
    F = 1/beta * ln_p + 1/2. * x**2 
    F_mean = F.mean()
    F_std = F.std()
    return F_mean, F_std

def make_grad_loss(x, beta, sigma):
    ln_p = logp(x, sigma)
    F = 1/beta * ln_p + 1/2. * x**2 
    #print('F:')
    #print(F)

    F_mean = F.mean()
    F_std = F.std()
    grad_logp = -1/sigma + x**2/(sigma**3)

    auto_grad_logp = jax.jacrev(logp, argnums = -1)(x, sigma)
    #print('auto_grad_logp:')
    #print(auto_grad_logp)

    grad = jnp.multiply(F, grad_logp)
    auto_grad = jnp.multiply(F - F_mean, auto_grad_logp)
    #auto_grad = jnp.multiply( F-F_mean, auto_grad_logp )
    #print('auto_grad:')
    #print(auto_grad)

    grad_mean = grad.mean()
    auto_grad_mean = auto_grad.mean()

    #assert jnp.allclose(grad_mean, auto_grad_mean)
    #return grad_mean, F_mean
    return auto_grad_mean, F_mean, F_std



def optimize_sigma(batch, beta, key, nstep, learning_rate):
    import optax

    optimizer = optax.adam(learning_rate = learning_rate)
    param = jax.random.uniform( key = jax.random.PRNGKey(42), minval = 0.01, maxval = 1.)
    opt_state = optimizer.init(param)
    
    #x = Gaussian_sampler((batch, ), param, key)
    batch_small = 20
    n = int(batch/batch_small)
    x = Gaussian_sampler_new((batch_small, n), param, key)

    sigma0 = 1/jnp.sqrt(beta)
    F_mean0 = make_loss(x, beta, sigma0)

    import pickle as pk
    fp = open('./out_Gaussian.txt', 'wb')

    #step_out = []
    param_out = []
    param_expect = 1/jnp.sqrt(beta)
    loss_out = []
    loss_expect = []
    loss_plus_std = []
    loss_minus_std = []

    def step(param, opt_state):
        grad, loss, std = make_grad_loss(x, beta, param)
        updates, opt_state = optimizer.update(grad, opt_state, param)
        param = optax.apply_updates(param, updates)
        return param, opt_state, loss, std

    for istep in range(nstep):
        param, opt_state, loss, std = step(param, opt_state)
        #print('istep, sigma, 1/sqrt(beta), F_mean0, loss:')
        print('istep, sigma, 1/sqrt(beta), loss, loss-F_mean0, std, loss - std, loss + std:')
        print(istep, param, 1/jnp.sqrt(beta), loss, loss - F_mean0, std, loss - std, loss + std)
        param_out.append(param)
        loss_out.append(loss)
        loss_expect.append(F_mean0)
        loss_plus_std.append(loss+std)
        loss_minus_std.append(loss-std)
        print('\n') 

    out = {}
    out['batch'] = batch
    out['params'] = param_out
    out['param_expect'] = param_expect
    out['loss'] = loss_out
    out['loss_expect'] = loss_expect
    out['loss_up'] = loss_plus_std
    out['loss_down'] = loss_minus_std

    pk.dump(out, fp)
    fp.close()


def scan_sigma(batch, beta, key):
    batch_small = 20
    n = int(batch/batch_small)
    sigma0 = 1/jnp.sqrt(beta)
    x0 = Gaussian_sampler_new((batch_small, n), sigma0, key)
    F_mean0, F_std0 = make_loss(x0, beta, sigma0)

    import pickle as pk
    fp = open('./out_scan_sigma.txt', 'wb')

    #step_out = []
    #param_out = []
    param_expect = sigma0
    loss_out = []
    loss_expect = []
    loss_plus_std = []
    loss_minus_std = []

    params = jnp.arange(0.01, 0.99, 0.01)
    #print('params:', params)

    for param in params:
        x = Gaussian_sampler_new((batch_small, n), param, key)
        loss, std = make_loss(x, beta, param)
        print('sigma, 1/sqrt(beta), loss, loss-F_mean0, std, loss - std, loss + std:')
        print(param, sigma0, loss, loss - F_mean0, std, loss - std, loss + std)
        loss_out.append(loss)
        loss_expect.append(F_mean0)
        loss_plus_std.append(loss+std)
        loss_minus_std.append(loss-std)
        print('\n') 

    out = {}
    out['batch'] = batch
    out['params'] = params
    out['param_expect'] = param_expect
    out['loss'] = loss_out
    out['loss_expect'] = loss_expect
    out['loss_up'] = loss_plus_std
    out['loss_down'] = loss_minus_std

    pk.dump(out, fp)
    fp.close()


