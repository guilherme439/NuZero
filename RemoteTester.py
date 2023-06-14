import ray

from Tester import Tester

@ray.remote
class RemoteTester(Tester):

    def __init__(self, recurrent_iters=2, mcts_simulations=800, pb_c_base=19652, pb_c_init=1.25, use_terminal=False, slow=False):

        super().__init__(
                        recurrent_iters=recurrent_iters,    
                        mcts_simulations=mcts_simulations, 
                        pb_c_base=pb_c_base, 
                        pb_c_init=pb_c_init, 
                        use_terminal=use_terminal,
                        slow=slow
                        )
        
