import os

from dataclasses import dataclass, field
from typing import List, Generator, Optional
from snakemake_interface_executor_plugins.executors.base import SubmittedJobInfo
from snakemake_interface_executor_plugins.executors.remote import RemoteExecutor
from snakemake_interface_executor_plugins.settings import (
    ExecutorSettingsBase,
    CommonSettings,
)
from snakemake_interface_executor_plugins.jobs import JobExecutorInterface
from snakemake_interface_common.exceptions import WorkflowError  # noqa
import subprocess


def writePBSScript(name, resources, array_size, bash_script_path):
    home = os.path.expanduser("~")
    location = f"{home}/jobs/{name}_job.sh"

    script = "#!/bin/bash\n"
    script += f"#PBS -o {home}/logs/{name}_out.log\n"
    script += f"#PBS -e {home}/logs/{name}_err.log\n"
    script += f"#PBS -l walltime={resources.walltime}\n"

    if hasattr(resources, "ngpus") and resources.ngpus > 0:
        select = f"#PBS -lselect=1:ncpus={resources.ncpus}:ompthreads={resources.ncpus}:mem={resources.mem}gb:ngpus={resources.ngpus}"
        if hasattr(resources, "gpu_type"):
            select += f":gpu_type={resources.gpu_type}"
        select += "\n"
    else:
        select = f"#PBS -lselect=1:ncpus={resources.ncpus}:ompthreads={resources.ncpus}:mem={resources.mem}gb\n"

    script += select
    script += f"#PBS -N {name}\n"
    script += "#PBS -V\n"
    script += f"#PBS -J 0-{array_size-1}\n"  # Array job range
    script += "cd $PBS_O_WORKDIR;\n"
    script += f"{bash_script_path}\n"  # Run the bash script

    with open(location, "w") as f:
        f.write(script)

    return location


def writeBashScript(location, name, commands):

    fileName = f"{location}/{name}.sh"

    with open(fileName, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("cd $PBS_O_WORKDIR;\n")

        if len(commands) == 1:
            # For single jobs, directly write the command
            f.write(f"{commands[0]}\n")
        else:
            # For array jobs, use a case statement to pick the command based on PBS_ARRAYID
            f.write("case $PBS_ARRAYID in\n")
            for index, command in enumerate(commands):
                f.write(f"{index}) {command} ;;\n")
            f.write("esac\n")

        f.write('echo "exit status: $?"\n')  # Exit status for bookkeeping

    subprocess.call(["chmod", "+x", fileName])

    return fileName


def lastline(filename):
    # Assuming filename exists
    return subprocess.check_output(
        f"grep 'exit status' {filename} | tail -1", shell=True
    )


@dataclass
class ExecutorSettings(ExecutorSettingsBase):
    # Example of a custom setting parameter for PBS
    pbs_queue: Optional[str] = field(
        default=None,
        metadata={
            "help": "Queue name for PBS jobs",
            "env_var": False,
            "required": False,
        },
    )


common_settings = CommonSettings(
    non_local_exec=True,
    implies_no_shared_fs=False,
    job_deploy_sources=True,
    pass_default_storage_provider_args=True,
    pass_default_resources_args=True,
    pass_envvar_declarations_to_cmd=True,
    auto_deploy_default_storage_provider=True,
    init_seconds_before_status_checks=0,
)


class Executor(RemoteExecutor):
    def __post_init__(self):
        self.workflow
        self.workflow.executor_settings
        self.job_groups = defaultdict(list)

    def run_job(self, job: JobExecutorInterface):
        # Group jobs by rule name for array job submission
        self.job_groups[job.rule.name].append(job)

    def submit_jobs(self):
        home = os.path.expanduser("~")

        if not os.path.exists(home + "/logs"):
            os.makedirs(home + "/logs")
            os.makedirs(home + "/jobs")

        for rule_name, jobs in self.job_groups.items():
            if len(jobs) > 1:
                # Create an array job
                self.submit_array_job(jobs, rule_name, home)
            else:
                # Submit single job normally
                self.submit_single_job(jobs[0], home)


def submit_array_job(self, jobs, rule_name, home):
    # Create a mapping of array index to job command
    job_commands = []
    array_indices = list(range(len(jobs)))  # Indices will be from 0 to len(jobs)-1

    for index, job in enumerate(jobs):
        name = f"{job.rule.name}.{job.jobid}"
        env = subprocess.check_output("export -p", shell=True).decode("utf-8")
        job_cmd = env + "\n" + self.format_job_exec(job)
        job_commands.append(job_cmd)

    # Create a single bash script that can execute any command based on PBS_ARRAYID
    bashLoc = writeBashScript(home + "/jobs", rule_name, job_commands)

    # Create a single PBS script for the array job
    scriptLoc = writePBSScript(rule_name, jobs[0].resources, len(jobs), bashLoc)

    cmd = f"qsub -t 0-{len(array_indices)-1} {scriptLoc}"

    try:
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True
        )
        job_id = result.stdout.strip()
        for index, job in enumerate(jobs):
            job_info = SubmittedJobInfo(job=job, external_jobid=f"{job_id}[{index}]")
            self.report_job_submission(job_info)
    except subprocess.CalledProcessError as e:
        raise WorkflowError(f"Failed to submit array job: {e.stderr}")

    def submit_single_job(self, job, home):
        name = f"{job.rule.name}.{job.jobid}"
        env = subprocess.check_output("export -p", shell=True).decode("utf-8")
        job_cmd = env + "\n" + self.format_job_exec(job)

        # Use a list with one command for single jobs
        bashLoc = writeBashScript(home + "/jobs", name, [job_cmd])
        scriptLoc = writePBSScript(name, job.resources, 1, bashLoc)

        cmd = f"qsub {scriptLoc}"

        try:
            result = subprocess.run(
                cmd, shell=True, check=True, capture_output=True, text=True
            )
            job_id = result.stdout.strip()
            job_info = SubmittedJobInfo(job=job, external_jobid=job_id)
            self.report_job_submission(job_info)
        except subprocess.CalledProcessError as e:
            raise WorkflowError(f"Failed to submit job: {e.stderr}")

    async def check_active_jobs(
        self, active_jobs: List[SubmittedJobInfo]
    ) -> Generator[SubmittedJobInfo, None, None]:
        async with self.status_rate_limiter:
            for job in active_jobs:
                cmd = f"qstat {job.external_jobid}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

                name = f"{job.job.rule.name}.{job.job.jobid}"

                if result.returncode != 0:  # Job is not found

                    home = os.path.expanduser("~")

                    logFile = f"{home}/logs/{name}_out.log"

                    # Check the logs to get the exit status
                    # (This might be a race condition)
                    if os.path.exists(logFile):
                        statusline = lastline(logFile)
                        exit_status = int(statusline.split()[-1])

                        if exit_status != 0:
                            self.report_job_error(job)
                            continue
                        else:
                            self.report_job_success(job)
                            continue

                    self.report_job_error(job)
                elif " C " in result.stdout:  # PBS marks completed jobs with a " C "
                    self.report_job_success(job)
                else:
                    # Still running
                    yield job

    def cancel_jobs(self, active_jobs: List[SubmittedJobInfo]):
        for job in active_jobs:
            cmd = f"qdel {job.external_jobid}"
            subprocess.run(cmd, shell=True, capture_output=True, text=True)
