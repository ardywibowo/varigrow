from torch.optim.optimizer import Optimizer

class MultipleOptimizer(Optimizer):
    def __init__(self, *op):
        self.optimizers = op
        
        params = []
        defaults = []
        for o in self.optimizers:
            params.append(o.params)
            defaults.append(o.defaults)
        
        super().__init__(params, defaults)

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()


# opt = MultipleOptimizer(optimizer1(params1, lr=lr1), 
#                         optimizer2(params2, lr=lr2))

# loss.backward()
# opt.zero_grad()
# opt.step()