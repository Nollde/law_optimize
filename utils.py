# -*- coding: utf-8 -*-

import os
from pathlib import Path
from collections import OrderedDict, defaultdict
from time import sleep
import random

import luigi
import law
from luigi import IntParameter, FloatParameter, ChoiceParameter
from skopt.space import Real, Integer, Categorical
from skopt.plots import plot_objective, plot_evaluations, plot_convergence
import matplotlib.pyplot as plt

from tasks.base import AnalysisTask
from utils.util import NumpyEncoder


law.contrib.load("matplotlib")


class SkoptLuigiParameter(object):
    """Merges Luigi and Skopt parameters
    Redundant keywords are set to the
    """

    optimizable = True

    def __repr__(self):
        return "{}".format(self.__class__.__name__)

    @property
    def skopt_keys(self):
        raise NotImplementedError

    def divide_kwargs(self, kwargs):
        skopt_kwargs = defaultdict()
        for skopt_key in self.skopt_keys:
            if skopt_key in kwargs.keys():
                skopt_kwargs[skopt_key] = kwargs[skopt_key]
                del kwargs[skopt_key]
        if "description" not in kwargs:
            kwargs["description"] = skopt_kwargs["name"]
        return kwargs, skopt_kwargs

    def freeze(self):
        self.optimizable = False
        return self


class SIntParameter(Integer, IntParameter, SkoptLuigiParameter):
    """Implements an integer parameter
    Example:
    a = SIntParameter(
        0,
        2,
        name='min_samples_split',
        default=1,
        description="Number"
    )
    """

    skopt_keys = ["transform", "name"]

    def __init__(self, *args, **kwargs):
        luigi_kwargs, skopt_kwargs = self.divide_kwargs(kwargs)
        IntParameter.__init__(self, **luigi_kwargs)
        Integer.__init__(self, *args, **skopt_kwargs)


class SFloatParameter(Real, FloatParameter, SkoptLuigiParameter):
    """Implements a float parameter
    Example:
    b = SFloatParameter(
        0.0,
        2.0,
        name='max_depth',
        default=1.5,
        description="Number"
    )
    """

    skopt_keys = ["prior", "transform", "name"]

    def __init__(self, *args, **kwargs):
        luigi_kwargs, skopt_kwargs = self.divide_kwargs(kwargs)
        FloatParameter.__init__(self, **luigi_kwargs)
        Real.__init__(self, *args, **skopt_kwargs)


class SChoiceParameter(Categorical, ChoiceParameter, SkoptLuigiParameter):
    """Implements a choice parameter
    Example:
    c = SChoiceParameter(
        [0, 1, 2],
        name="categorical",
        choices=["0", "1", "2"],
        description="categorical"
    )
    """

    skopt_keys = ["prior", "transform", "name"]

    def __init__(self, *args, **kwargs):
        luigi_kwargs, skopt_kwargs = self.divide_kwargs(kwargs)
        choices = args[0]
        ChoiceParameter.__init__(self, choices=choices, **luigi_kwargs)
        Categorical.__init__(self, *args, **skopt_kwargs)


class TargetLock(object):
    def __init__(self, target):
        self.target = target
        self.path = target.path
        self.lock = self.path + ".lock"

    def __enter__(self):
        while True:
            sleep(5 + 10 * random.random())
            if not Path(self.lock).is_file():
                Path(self.lock).touch()
                self.loaded = self.target.load()
                break
        return self.loaded

    def __exit__(self, type, value, traceback):
        self.target.dump(self.loaded)
        os.remove(self.lock)


class Opt:
    opt_version = IntParameter(default=0, description="Version number of the optimizer run")

    def store_parts(self):
        return super().store_parts() + (f"opt_version_{self.opt_version}",)


class Optimizer(Opt, AnalysisTask, law.LocalWorkflow):
    """
    Workflow that runs optimization.
    """

    iterations = luigi.IntParameter(default=4, description="Number of iterations")
    n_parallel = luigi.IntParameter(default=2, description="Number of parallel evaluations")
    objective_key = luigi.Parameter("objective")
    status_frequency = luigi.IntParameter(default=50, description="Frequency to give a status.")

    @property
    def objective(self):
        raise NotImplementedError

    def create_branch_map(self):
        return list(range(self.n_parallel))

    def requires(self):
        return OptimizerPreparation.req(self)

    def output(self):
        return {
            "opt": self.local_target("optimizer.pkl"),
            "conv": self.local_target("convergence.pdf"),
            "obj": self.local_target("objective.pdf"),
        }

    def plot_status(self, opt):

        result = opt.run(None, 0)
        output = self.output()

        plot_convergence(result)
        output["conv"].dump(plt.gcf(), bbox_inches="tight")

        plot_objective(result)
        output["obj"].dump(plt.gcf(), bbox_inches="tight")

        plt.close()

    @property
    def todo(self):
        return self.local_target("todo_{}.json".format(self.branch))

    def obj_req(self, ask):
        return self.objective.req(self, **ask, branch=-1)

    def run(self):
        if self.todo.exists():
            ask = self.todo.load()
            obj = yield self.obj_req(ask)
            y = obj["collection"].targets[0][self.objective_key].load()
            with TargetLock(self.input()["opt"]) as opt:
                opt.tell(list(ask.values()), y)
                self.todo.remove()

        with TargetLock(self.input()["opt"]) as opt, TargetLock(
            self.input()["todos"]
        ) as todos, TargetLock(self.input()["keys"]) as keys:
            iteration = len(opt.Xi)
            if iteration and iteration % self.status_frequency == 0:
                self.plot_status(opt)
            if iteration >= self.iterations:
                self.output()["opt"].dump(opt)
                self.output()["obj"].touch()
                self.output()["conv"].touch()
                return
            print("got new todo", end=", ")
            if len(todos) > 0:
                ask = todos.pop(0)
                print("from todos", end=": ")
            else:
                x = opt.ask()
                ask = dict(zip(keys, x))
                print("by asking", end=": ")
            print(ask)
        self.todo.dump(ask, cls=NumpyEncoder)
        yield self.obj_req(ask)


@luigi.util.inherits(Optimizer)
class OptimizerPreparation(Opt, AnalysisTask):
    """
    Task that prepares the optimizer and draws a todo list.
    """

    n_initial = luigi.IntParameter(
        default=10,
        description="Number of random sampled values \
        before starting optimizations",
    )

    @property
    def n_initial_points(self):
        return max(self.n_initial, self.n_parallel)

    def output(self):
        return {
            "opt": self.local_target("optimizer.pkl"),
            "todos": self.local_target("todos.json"),
            "keys": self.local_target("keys.json"),
        }

    @property
    def objective(self):
        raise NotImplementedError

    def optimizable_parameters(self):
        params = self.objective.get_params()
        opt_params = OrderedDict()
        for name, param in params:
            if isinstance(param, SkoptLuigiParameter):
                if param.optimizable:
                    opt_params[name] = param
        return opt_params

    def run(self):
        import skopt

        opt_params = self.optimizable_parameters()
        optimizer = skopt.Optimizer(
            dimensions=list(opt_params.values()),
            random_state=1,
            n_initial_points=self.n_initial_points,
        )
        x = [optimizer.ask() for i in range(self.n_initial_points)]
        ask = [dict(zip(opt_params.keys(), val)) for val in x]

        with self.output()["opt"].localize("w") as tmp:
            tmp.dump(optimizer)
        with self.output()["todos"].localize("w") as tmp:
            tmp.dump(ask, cls=NumpyEncoder)
        with self.output()["keys"].localize("w") as tmp:
            tmp.dump(list(opt_params.keys()))


@luigi.util.inherits(Optimizer)
class OptimizerDraw(Opt, AnalysisTask):
    """
    Task that prepares the optimizer and draws a todo list.
    """

    n_draw = luigi.IntParameter(
        default=10,
        description="Number of samples \
        drawn for restart of optimization.",
    )

    @property
    def n_initial_points(self):
        return max(self.n_initial, self.n_parallel)

    def output(self):
        return self.local_target("todos.json")

    def run(self):
        import skopt

        opt = self.input()["opt"].load()
        todos = opt.ask(self.n_draw)
        with TargetLock(self.input()["keys"]) as keys:
            todos = [dict(zip(keys, todo)) for todo in todos]
        self.output().dump(todos, cls=NumpyEncoder)


@luigi.util.inherits(Optimizer)
class OptimizerPlot(Opt, AnalysisTask):
    """
    Workflow that runs optimization and plots results.
    """

    def create_branch_map(self):
        return list(range(self.n_parallel))

    plot_objective = luigi.BoolParameter(
        default=True,
        description="Plot objective. \
        Can be expensive to evaluate for high dimensional input",
    )

    def requires(self):
        return Optimizer.req(self, branch=-1)

    def output(self):
        collection = {
            "evaluations": self.local_target("evaluations.pdf"),
            "convergence": self.local_target("convergence.pdf"),
        }

        if self.plot_objective:
            collection["objective"] = self.local_target("objective.pdf")

        return collection

    def run(self):
        result = self.input()["collection"].targets[0]["opt"].load().run(None, 0)
        output = self.output()

        plot_convergence(result)
        output["convergence"].dump(plt.gcf(), bbox_inches="tight")
        plt.close()
        plot_evaluations(result, bins=10)
        output["evaluations"].dump(plt.gcf(), bbox_inches="tight")
        plt.close()
        if self.plot_objective:
            plot_objective(result)
            output["objective"].dump(plt.gcf(), bbox_inches="tight")
            plt.close()
