import os

import luigi
import law
import numpy as np

import utils as lopt


class BaseTask(law.Task):
    """
    Base task which provides some convenience methods to create local file
    and directory targets at the default data path.
    """
    version = law.Parameter() 

    def store_parts(self):
        task_name = self.__class__.__name__
        return (
            os.getenv("OUT_DIR"),
            f"version_{self.version}",
            task_name,
        )

    def local_path(self, *path):
        sp = self.store_parts()
        sp += path
        return os.path.join(*(str(p) for p in sp))

    def local_target(self, *path, **kwargs):
        return law.LocalFileTarget(self.local_path(*path), **kwargs)


class Evaluation(BaseTask):
    param1 = lopt.SFloatParameter(0.0, 1.0, default=0.5, name="param1")
    param2 = lopt.SIntParameter(0, 10, default=5, name="param2")

    def store_parts(self):
        return super().store_parts() + (
            f"param1_{self.param1}",
            f"param2_{self.param2}",
        )

    def output(self):
        return {
            "objective": self.local_target("objective.json"),
        }

    def run(self):
        # dummy cost function
        objective = np.exp((self.param1 - 0.5) ** 2 / 1 + (self.param2 - 5) ** 2 / 25)
        self.output()["objective"].dump(objective)


class OptimizerPreparation(lopt.OptimizerPreparation, BaseTask):
    objective = Evaluation


class Optimizer(lopt.Optimizer, BaseTask):
    objective = Evaluation

    def requires(self):
        return OptimizerPreparation.req(self)

class OptimizerPlot(lopt.OptimizerPlot, BaseTask):
    def requires(self):
        return Optimizer.req(self)
