# Start
[Done]
We are developing a package for nested sampling using JAX GPU acceleration. Please read the development notes in dev/ and summarize the status of the code.


# Organization
[Done]
I want you to investigate the code structure of dynesty and rearrange jnesty. Please check `https://github.com/joshspeagle/dynesty/tree/8812f7eeae1ef2df656c8d12733859b426bac298/py/dynesty`. For example, `constrained_sampler.py`, `sampler.py`, and `while_loop_sampler.py` are all for the sampler. There should also be a `results.py` module. (I want the sampler to only provide the results in the dynesty format.) Note that we will add more sampling algorithm and bounding algorithm in the future, like dynesty. The code structure should be prepared. Think carefully and give me a plan.


# Result saving
[Done]
I want you to add a save function to save all the sampling results into a FITS file. I also want a load function to read the FITS results and produce a results object.

===
[Done]
Please update the demos. Save the sampling results of both jnesty and dynesty into FITS. Remove the dev/demo/output_* and reproduce the plots and results (in FITS).


# Installation
[Done]
I want to the package to be installed with `pip install -e .`. Install it in the conda galfits environment. Update the demo scripts so that we do not need to insert the src/ path.


# Link to GitHub
[Done]
I created an empty repo (with license and .gitignore files) in the following github link, git@github.com:jyshangguan/jnesty.git

Please push our package up to GitHub.


# Skills
[Done]
Please add the skills for this package. The skills should include the usage and development information. I hope it is structured and prepared for future developments. Please think carefully and give me a plan.


# Speed with multi-ellipsoid bounding

It seems that our multi-ellipsoid bounding is much slower than the dynesty. Please investigate if there is anything we can accelerate.


# Unit tests

I want you to add unit tests to confirm the robustness and speed of all the key functions. Please check the code carefully and give me a plan.


# Debug

1. Please check why the trace plot of jnesty seems to have much less point than that from dynesty, while the number of samples are not much different (please verify it with the demo 01 results and plots). I understand that jnesty effectively use the same plot function as dynesty. Please help me to understand if there is any bugs.