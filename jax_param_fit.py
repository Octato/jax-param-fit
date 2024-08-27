"""
Example usage of JAX to perform function parameter fitting via gradient descent.
"""

from jax import grad, jit, Array
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt

def gen_sine_data(params: Array, samples: int) -> Array:
    """Generate a sum of two sine waves."""
    x = jnp.linspace(-10, 10, samples)
    y1 = params[0] * jnp.sin(params[1] * x + params[2])
    y2 = params[3] * jnp.sin(params[4] * x + params[5])
    data = jnp.stack([x, y1 + y2])
    return data

def gen_train_data(variance: float, samples: int) -> Array:
    """Generate training data."""
    params = random.normal(random.key(13), [6])
    noise_key = random.key(1)
    data = gen_sine_data(params, samples)
    noise = random.normal(noise_key, [samples]) * variance
    x = data[0]
    y = data[1] + noise
    return jnp.stack([x, y]) 

train_data = gen_train_data(0.1, 1000)

def pred(x: float, params: Array):
    """Evaluate the model."""
    y1 = params[0] * jnp.sin(params[1] * x + params[2])
    y2 = params[3] * jnp.sin(params[4] * x + params[5])
    return y1 + y2

@jit
def loss(params: Array) -> Array:
    """Compute MSE loss for the model."""
    x = train_data[0]
    y_train = train_data[1]
    y_pred = pred(x, params)
    return jnp.mean(jnp.square(y_train - y_pred))

def train() -> Array:
    """Initialize and train the model parameters."""
    search_steps = 500
    train_steps = 2500
    learning_rate = 0.01
    losses = []
    params = None
    keys = random.split(random.key(0), search_steps)
    k = 0
    while k < search_steps:
        new_params = random.normal(keys[k], [6])
        l = loss(new_params)
        if (params is None or l.item() < losses[-1]):
            params = new_params
            losses.append(l.item())
            print(f'[Search] Loss =  {l.item()}')
        k += 1
    i = 0
    while i < train_steps:
        l = loss(params)
        l_grad = grad(loss)
        losses.append(l.item())
        print(f'[Train] Loss = {l.item()}')
        params = params - l_grad(params) * learning_rate
        i += 1
    return params

if __name__ == '__main__':
    trained_params = train()
    pred_data = gen_sine_data(trained_params, 100)
    plt.scatter(train_data[0], train_data[1], c='red', s=1.0)
    plt.plot(pred_data[0], pred_data[1], c='blue')
    plt.show()
