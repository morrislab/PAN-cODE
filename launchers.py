import subprocess
import sys
import unicodedata
import getpass
import time

def local_launcher(commands, *args, **kwargs):
    for cmd in commands:
        subprocess.call(cmd, shell=True)
                
def slurm_launcher(commands, max_slurm_jobs, *args, **kwargs):
    for cmd in commands:
        block_until_running(max_slurm_jobs, getpass.getuser())
        subprocess.call(cmd, shell=True)     

def get_slurm_jobs(user):
    # returns a list of jobs IDs for (queued and waiting, running)
    out = subprocess.run(['squeue -u ' + user], shell = True, stdout = subprocess.PIPE).stdout.decode(sys.stdout.encoding)
    a = list(filter(lambda x: len(x) > 0, map(lambda x: x.split(), out.split('\n'))))
    queued, running = [], []
    for i in a:
        if i[0].isnumeric():
            if i[4].strip() == 'PD':
                queued.append(int(i[0]))
            else:
                running.append(int(i[0]))
    return (queued, running)

def block_until_running(n, user):
    while True:
        queued, running = get_slurm_jobs(user)
        if len(queued) + len(running) < n:
            time.sleep(0.2)
            return True
        else:
            time.sleep(10)        
        
REGISTRY = {
    'local': local_launcher,
    'slurm': slurm_launcher
}