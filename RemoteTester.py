import ray

from Tester import Tester

@ray.remote(scheduling_strategy="SPREAD")
class RemoteTester(Tester):

    def __init__(self, slow=False, print=False, render=False):

        super().__init__(slow=slow, print=print, render=render)
        
