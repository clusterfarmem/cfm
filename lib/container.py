import os
from lib import utils
from lib import constants


class Container:
    def __init__(self, name, mem_req, ratio):
        self.name = name
        self.mem_req = mem_req  # in MB
        self.ratio = ratio

    def exists(self):
        """ Returns whether this container still exists """
        return os.path.isdir(self.get_cont_path())

    def delete(self):
        path = self.get_cont_path()
        ret = utils.shell_exec("rmdir {0}".format(path))[0]
        if ret:
            raise RuntimeError("Error deleting {}".format(path))

    def set_memory_limit(self):
        # this is possible if the caller is multithreaded
        # and hasn't realized the container has been deleted
        if not self.exists():
            return

        if self.ratio == 'max':
            memory_limit = 'max'
            print("Setting container memory limit to Max")
        else:
            memory_limit = str(round(self.ratio*self.mem_req)) + 'M'
            print("Setting {} memory limit to "
                  "{}% ({}) of max".format(self.name,
                                           round(self.ratio*100),
                                           memory_limit))

        mem_high_path = self.get_cont_path() + '/memory.high'
        with open(mem_high_path, 'w') as f:
            f.write(memory_limit)

    def set_new_size(self, local_ratio):
        self.ratio = local_ratio
        self.set_memory_limit()

    def get_cont_path(self):
        return "{}/{}".format(constants.CGROUP_PATH, self.name)

    def get_procs_path(self):
        return self.get_cont_path() + '/cgroup.procs'

    def create(self):
        """creates new container as child of CGROUP_PATH"""
        new_cont_path = self.get_cont_path()
        try:
            os.mkdir(new_cont_path)
            assert self.exists()
        except FileExistsError:
            print("container {} already exists, trying to delete".format(self.name))
            self.delete()
            os.mkdir(new_cont_path)
        self.set_memory_limit()

    def get_pids(self):
        try:
            with open(self.get_procs_path(), 'r') as f:
                pids = f.readlines()
                pids = map(lambda p: p.rstrip('\n'), pids)
                pids = tuple(map(int, pids))
                return pids
        except Exception as e:
            print("Exception of type: {}".format(type(e)))
            print("Procs path: {}".format(self.get_procs_path()))
            return ()

def check():
    '''Check that the cgroup path exists and that the memory controller is enabled'''
    if not os.path.isdir(constants.CGROUP_PATH):
        raise RuntimeError("{} does not exist".format(constants.CGROUP_PATH))

    with open(constants.CGROUP_PATH + '/cgroup.subtree_control', 'r') as f:
        content = f.read()
        if 'memory' not in content:
            raise RuntimeError('memory controller not enabled')
