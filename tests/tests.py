from typing import Optional
import snakemake.common.tests
from snakemake_interface_executor_plugins.settings import ExecutorSettingsBase


# Check out the base classes found here for all possible options and methods:
# https://github.com/snakemake/snakemake/blob/main/snakemake/common/tests/__init__.py
class TestWorkflowsBase(snakemake.common.tests.TestWorkflowsBase):
    __test__ = True

    def get_executor(self) -> str:
        return "pbs"

    def get_executor_settings(self) -> Optional[ExecutorSettingsBase]:
        return None
