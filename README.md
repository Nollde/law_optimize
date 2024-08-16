# Law Optimize
Law Optimize helps to optimize parameterizable law tasks.

## Principle
The optimization is based on a Bayesian Optimization Strategy utilizing a Gaussian Process Estimator.
At first, the Optimizer Preparation task will suggestion `n_initial` points to test.
After these points are tested and feed back is provided to the optimizer instance, the optimizer will have a state.
Now for each optimization step, the optimizer is asked for which new points to test next.
These points are tested by `n_parallel` parallel instances of the `Optimizer` task (so called "Optimization Streams").
Every time an `Optimizer` task finishes, it provides its outcome to the instance, new points are queried, and the process goes on.
The optimization proceeds until `n_iterations` points have been evaluated.

## Example
Setup the law environment
```
source setup.sh
```

The `Evaluation` task has two optimizable parameters:
```
param1 = lopt.SFloatParameter(0.0, 1.0, default=0.5, name="param1")
param2 = lopt.SIntParameter(0, 10, default=5, name="param2")
```

The `SFloatParameter` and `SIntParameter` are recognized by the optimizer as optimizable parameters.
The `Evaluation` task has the (standardized) output "objective", which keeps a file target with exactly one number, the objective number itself.
In this example, the objective is a two dimensional Gauss function:
```
objective = np.exp((self.param1 - 0.5) ** 2 / 1 + (self.param2 - 5) ** 2 / 25)
```

Now we can run the optimizer:
```
law run OptimizerPlot --version example --iterations 200
```

It will query the `Evaluation` task up to `iterations` times.
Then it will generate some beautiful plots of the optimization and cost function:
![objective function](https://github.com/user-attachments/assets/6f214545-86d0-4c0e-ba8a-b3070aff2f64)