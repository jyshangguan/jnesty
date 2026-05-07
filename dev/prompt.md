# Start

We are developing a package for nested sampling using JAX GPU acceleration. Please read the development notes in dev/ and summarize the status of the code.


# Organization

I want you to investigate the code structure of dynesty and rearrange jnesty. Please check `https://github.com/joshspeagle/dynesty/tree/8812f7eeae1ef2df656c8d12733859b426bac298/py/dynesty`. For example, `constrained_sampler.py`, `sampler.py`, and `while_loop_sampler.py` are all for the sampler. There should also be a `results.py` module. (I want the sampler to only provide the results in the dynesty format.) Note that we will add more sampling algorithm and bounding algorithm in the future, like dynesty. The code structure should be prepared. Think carefully and give me a plan.


# Result saving

I want you to save all the sampling results