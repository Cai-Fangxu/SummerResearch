import netket as nk
import jax
import jax.numpy as jnp

class myRBM(nk.nn.Module):

    alpha: float = 1

    @nk.nn.compact
    def __call__(self, x):
        x = jnp.atleast_2d(x)
        return jax.vmap(self.single_evaluation, in_axes=(0))(x)

    def single_evaluation(self, x):
        v_bias = self.param("visual_bias", nk.nn.initializers.normal(0.01), (x.shape[-1], ), complex)
        h_bias = self.param("hidden_bias", nk.nn.initializers.normal(0.01), (int(x.shape[-1]*self.alpha), ), complex)
        W = self.param("weight", nk.nn.initializers.normal(0.01), (int(x.shape[-1]*self.alpha), x.shape[-1]), complex)

        x2 = W@x + h_bias
        x3 = nk.nn.activation.logcosh(x2)
        y = jnp.sum(x3) + jnp.dot(x, v_bias)

        return y


class mySymmRBM(nk.nn.Module):
    
    alpha: int = 1
    filter_len: int = None

    @nk.nn.compact
    def __call__(self, x):
        x = jnp.atleast_2d(x)
        return jax.vmap(self.single_evaluation, in_axes=(0))(x)   

    def single_evaluation(self, x):
        a = self.param("a", nk.nn.initializers.normal(0.01), (1, ), complex)
        b = self.param("b", nk.nn.initializers.normal(0.01), (self.alpha,), complex)
        if self.filter_len == None:
            self.filter_len = x.shape[-1]
        W = self.param("weight", nk.nn.initializers.normal(0.5), (self.alpha, self.filter_len, ), complex)
        #W2 = self.param('weight2', nk.nn.initializers.normal(0.01), (self.alpha, ), complex)

        def localFeatures(site=0):
            indices = jnp.arange(self.filter_len)
            indices = (indices+site)% (x.shape[-1])
            sites = x[indices]
            local_features  = jnp.dot(W, sites) + b
            return local_features
            # the shape of local_features is (alpha, ), alpha is the number of features 

        features = jax.vmap(localFeatures)(jnp.arange(x.shape[-1]))
        features = nk.nn.activation.logcosh(features)
        #features = nk.nn.activation.logcosh(jnp.sum(features, axis=0))
        #y = jnp.dot(features, W2) + a[0]*jnp.sum(x)
        y = jnp.sum(features) + a[0]*jnp.sum(x)

        return y
