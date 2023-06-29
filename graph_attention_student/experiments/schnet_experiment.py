from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

__DEBUG__ = True

@Experiment(base_path=folder_path(__file__),
            namespace=file_namespace(__file__),
            glob=globals())
def experiment(e: Experiment):
    e.log("hello world")
    e["VARIABLE"] = 20


experiment.run_if_main()