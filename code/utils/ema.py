import copy
import torch

class EMA:
    """
    Exponential Moving Average for model parameters.
    
    This class maintains an exponential moving average of model parameters
    for improved validation performance and stability.
    """
    
    def __init__(self, model, decay=0.999):
        """
        Initialize EMA with a copy of the model.
        
        Args:
            model: PyTorch model to track
            decay: EMA decay rate (higher = more smoothing)
        """
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters(): 
            p.requires_grad_(False)
        self.decay = decay
        self._bak = None

    @torch.no_grad()
    def update(self, model):
        """
        Update EMA parameters with current model parameters.
        
        Args:
            model: Current model to update from
        """
        d = self.decay
        for e_p, p in zip(self.ema.parameters(), model.parameters()):
            e_p.data.mul_(d).add_(p.data, alpha=1 - d)

    @torch.no_grad()
    def swap_in(self, model):
        """
        Temporarily swap EMA parameters into the model for evaluation.
        
        Args:
            model: Model to swap EMA parameters into
        """
        self._bak = [p.data.clone() for p in model.parameters()]
        for p, e in zip(model.parameters(), self.ema.parameters()):
            p.data.copy_(e.data)

    @torch.no_grad()
    def restore(self, model):
        """
        Restore original model parameters after EMA evaluation.
        
        Args:
            model: Model to restore original parameters to
        """
        if self._bak is not None:
            for p, b in zip(model.parameters(), self._bak):
                p.data.copy_(b)
            self._bak = None
