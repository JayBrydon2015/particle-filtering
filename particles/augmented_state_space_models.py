"""
Created on Sun Feb 22 2026

@author: Jay Brydon

Augmented state-space models as Python objects.

Overview
========
This module defines:

    1. the `AugmentedStateSpaceModel` class as a subclass of StateSpaceModel,
       which lets you define an augmented state-space model as a Python
       object;

    2. The augmented bootstrap and guided Feynman-Kac models as subclasses of
       the original classes to accomodate the changes in the new Augmented
       SSM class.

The recommended import is:

    from particles import augmented_state_space_models as augssms

See state_space_models.py for more details. This module is simply an extension.

"""


from particles import state_space_models as ssms


class AugmentedStateSpaceModel(ssms.StateSpaceModel):
    """Augmented state-space model class, a subclass of StateSpaceModel."""
    
    def PY(self, t, xp, x, datap=None):
        """Conditional distribution of Y_t, given the states and Y_{t-1}.

        If t = 0, condition only on X_0. No need to pass in datap.
        """
        raise NotImplementedError(self._error_msg("PY"))
    
    def simulate_given_x(self, x):
        lag_x = [None] + x[:-1]
        y = [self.PY(0, lag_x[0], x[0]).rvs(size=1)] # t = 0
        for t in range(1, len(x)): # t >= 1
            y.append(self.PY(t, lag_x[t], x[t], y[-1]).rvs(size=1))
        return y


class AugmentedBootstrap(ssms.Bootstrap):
    """Bootstrap Feynman-Kac formalism for Augmented SSMs."""
    
    def logG(self, t, xp, x):
        if t == 0:
            return self.ssm.PY(t, xp, x).logpdf(self.data[t])
        else:
            return self.ssm.PY(t, xp, x, self.data[t-1]).logpdf(self.data[t])


class AugmentedGuidedPF(ssms.GuidedPF):
    """Guided Feynman-Kac formalism for Augmented SSMs."""
    
    def logG(self, t, xp, x):
        if t == 0:
            return (
                self.ssm.PX0().logpdf(x)
                + self.ssm.PY(0, xp, x).logpdf(self.data[0])
                - self.ssm.proposal0(self.data).logpdf(x)
            )
        else:
            return (
                self.ssm.PX(t, xp).logpdf(x)
                + self.ssm.PY(t, xp, x, self.data[t-1]).logpdf(self.data[t])
                - self.ssm.proposal(t, xp, self.data).logpdf(x)
            )
