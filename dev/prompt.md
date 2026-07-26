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


# Speed up

## Parallelization
[Done]
I want you to evaluate how to parallelize the likelihood call in GPU. Make a dev branch and work there. Think carefully and make a plan.

===
[Done]
Please evaluate if there is a default way to determine a suitable batch_size, so that the user do not need to specify. Give me a report first.

===
[Done]
It seems that we only use a single GPU in the sampling when there are more available. Please understand what went wrong. Make a careful check first and give me a report.


## Multi-ellipsoid bounding
[Done]
It seems that our multi-ellipsoid bounding is much slower than the dynesty. Please investigate if there is anything we can accelerate.


# Unit tests
[Done]
I want you to add unit tests to confirm the robustness and speed of all the key functions. Please check the code carefully and give me a plan.


## Annealing
[Abandoned]
Investigated annealed NS (temperature parameter) for peaked likelihoods. All three approaches tested produced biased posteriors — temperature disrupts the prior mass–likelihood relationship that NS relies on. Code rolled back.

# Debug
[Done]
1. Please check why the trace plot of jnesty seems to have much less point than that from dynesty, while the number of samples are not much different (please verify it with the demo 01 results and plots). I understand that jnesty effectively use the same plot function as dynesty. Please help me to understand if there is any bugs.

# Doc

I want you to check the code usage carefully and generate a doc. The doc should include the following pages,

(1) Basic introduction, including the installation and basic usage. 
(2) A page to introduce the basic method. We mainly refer to dynesty for the rwalk and multi-ellipsoid bounding, but this page provides a brief explanation on how our sampling method works.
(3) Convert the demos into four examples in the example page. We just show the JNesty results. No need to show the dynesty results.
(4) A API page that explains all the functions.

Please think carefully and provide me a plan.


# Version

Make the current version 0.1.0. Include this information in the package's __version__ as well as the doc.


# Implement the dynamic sampling

The current JNesty code only have the basic nested sampling method. I want to implement the dynamic nesty sampling. Please investigate the dynesty doc and code, and understand what should be added in JNesty. JNesty should take the full advantage of the GPU acceleration. Please also check this page, https://dynesty.readthedocs.io/en/latest/dynamic.html, to design a clear test so that we can quantitatively check if the code is correctly implementedd. Please think carefully and make a plan. I want you to design a loop so that you can automatically make tests and check to confirm if the code works as expected.

Please work in `/home/shangguan/Softwares/my_modules/JNesty/dev/task_004_dynamic` for this development. Please make a `dev_dynamic` for this task. Please read all the markdown files in `/home/shangguan/Softwares/my_modules/jnesty/dev` to understand the development strategy and keep the rules.


## Check

In the example, `/home/shangguan/Softwares/my_modules/jnesty/dev/demo/05_dynamic_gaussian_jnesty.py`, I want to plot the trace plot for the posterior/evidence split weights to be 80/20, 100/0, and 0/100, to illustrate that the dynamic sampling works.

Sorry, for the traceplot, I mean using `dyplot.runplot` to generate the plot. Please revise the code and re-run it.

I actually want to directly compare the results produced by

```
fig, axes = dyplot.runplot(res, color='black', mark_final_live=False,
                           logplot=True)  # static run
fig, axes = dyplot.runplot(dres, color='red', logplot=True,
                           fig=(fig, axes))  # default dynamic run
fig, axes = dyplot.runplot(dres_p, color='blue', logplot=True,
                           fig=(fig, axes))  # posterior dynamic run
fig, axes = dyplot.runplot(dres_z, color='limegreen', logplot=True,
                           lnz_truth=lnz_truth,  truth_color='orange',
                           fig=(fig, axes))  # evidence dynamic run
fig.tight_layout()
```

in `https://dynesty.readthedocs.io/en/latest/dynamic.html#`. Please try to use the same likelihood and sampling setups and make the same plot, so that I can have one-to-one comparison.

## Debug

I think the result plot is different from the dynesty website. I suggest you to run the same sampling with dynesty and directly compare the results with JNesty. I want quantitative comparison and make sure that JNesty got essentially the same results as dynesty in the test example. As for the result plot, just over plot the sampling results of dynesty and JNesty for a static sampling case and a dynamic sampling case (with posterior/evidence split 80/20).

Good. However, please compare the sample results in more detail. I think the live point history as the function of -lnX (the first panel of the runplot) is very different for dynesty and JNesty for dynamic sampling. The other curves are also noticibly different. The static sampling also show some difference. Please check and try to find the problem.

The results are still quite different. Please investigate the dynesty code for the dynamic sampler part and explain the algorithm step by step. Then, compare your current implementation in JNesty. Make point-to-point comparison and explain the differences. In this way, let's see what may course the difference. Note that our goal is to get almost identical results.

I do not understand why the live point curve (the first panel of the runplot) is very irregular. I expect the injection to be at fixed step while the live point decrease gradually after the injection. The dynesty curve behaves like this while the JNesty curve does not.