"""
Implementation of L_infty Carlini-Wagner attack based on the L2 implementation
in FoolBox v1.9 (with many dependencies on that pakage)
https://github.com/bethgelab/foolbox
"""

from iGAT.foolbox.models import PyTorchModel
from iGAT.foolbox.v1.attacks import Attack
from iGAT.foolbox.v1.attacks.base import call_decorator
from iGAT.foolbox.utils import onehot_like
from iGAT.foolbox.distances import Linf
# call_decorator = lambda x: x
HAVE_FOOLBOX = True
import numpy as np
import torch
import torch.nn as nn

def cw(model,
       X,
       y,
       norm_mean,
       norm_std,
       num_classes,
       binary_search_steps=5,
       max_iterations=100,
       learning_rate=5E-3,
       initial_const=1E-2,
       tau_decrease_factor=0.9,
       eps=8./255.,
       ):
    if not HAVE_FOOLBOX:
        raise ImportError('Could not import FoolBox')
    foolbox_model = PyTorchModel(model, bounds=(0, 1), num_classes=num_classes, device='cuda:0', 
                  preprocessing=(norm_mean, norm_std))

    attack = CarliniWagnerLIAttack(foolbox_model, distance=Linf)
    linf_distances = []
    correct = np.array([0, 0, 0, 0])
    # epses = [8/255, 4/255, 2/255, 1/255]
    eps = 8/255
    total = 0
    for i in range(len(X)):
        image = X[i, :].detach().cpu().numpy()
        label = y[i].cpu().numpy()
        adversarial = attack(image, label,
                             binary_search_steps=binary_search_steps,
                             max_iterations=max_iterations,
                             learning_rate=learning_rate,
                             initial_const=initial_const,
                             tau_decrease_factor=tau_decrease_factor)
        
        if adversarial is None:
            adversarial = image
            adversarial = torch.from_numpy(adversarial).cuda()
            adversarial -= norm_mean.view(3,1,1).cuda()
            adversarial /= norm_std.view(3,1,1).cuda()
            if model(adversarial[None, :]).argmax(axis=-1).cpu().numpy() == label:
                correct += 1
        else:
            for j in range(4):
                adversarial2 = image + np.clip(adversarial - image, -epses[j], epses[j])
                adversarial2 = np.clip(adversarial2, 0., 1.)
                adversarial2 = torch.from_numpy(adversarial2).cuda()
                adversarial2 -= norm_mean.view(3,1,1).cuda()
                adversarial2 /= norm_std.view(3,1,1).cuda()
                if model(adversarial2[None, :]).argmax(axis=-1).cpu().numpy() == label:
                    correct[j] += 1
        total += 1
        print(i, correct, total, correct / total)
    return correct

class CarliniWagnerLIAttack(Attack):
    """The Linf version of the Carlini & Wagner attack.
    This attack is described in [1]_. This implementation
    is based on the reference implementation by Carlini [2]_.
    For bounds \ne (0, 1), it differs from [2]_ because we
    normalize the squared L2 loss with the bounds.
    References
    ----------
    .. [1] Nicholas Carlini, David Wagner: "Towards Evaluating the
           Robustness of Neural Networks", https://arxiv.org/abs/1608.04644
    .. [2] https://github.com/carlini/nn_robust_attacks
    """

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 binary_search_steps=5,
                 tau_decrease_factor=0.9,
                 max_iterations=1000,
                 confidence=0, learning_rate=5e-3,
                 initial_const=1e-2, abort_early=True):

        """The L_infty version of the Carlini & Wagner attack.
        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        binary_search_steps : int
            The number of steps for the binary search used to
            find the optimal tradeoff-constant between distance and confidence.
        max_iterations : int
            The maximum number of iterations. Larger values are more
            accurate; setting it too small will require a large learning rate
            and will produce poor results.
        confidence : int or float
            Confidence of adversarial examples: a higher value produces
            adversarials that are further away, but more strongly classified
            as adversarial.
        learning_rate : float
            The learning rate for the attack algorithm. Smaller values
            produce better results but take longer to converge.
        initial_const : float
            The initial tradeoff-constant to use to tune the relative
            importance of distance and confidence. If `binary_search_steps`
            is large, the initial constant is not important.
        abort_early : bool
            If True, Adam will be aborted if the loss hasn't decreased
            for some time (a tenth of max_iterations).
        """
        norm_std = torch.Tensor([0.2023, 0.1994, 0.2010])
        norm_mean = torch.Tensor([0.4914, 0.4822, 0.4465])
        denorm_std = torch.Tensor([1/0.2023, 1/0.1994, 1/0.2010])
        denorm_mean = torch.Tensor([-0.4914, -0.4822, -0.4465])
        
        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        if not a.has_gradient():
            logging.fatal(
                "Applied gradient-based attack to model that "
                "does not provide gradients."
            )
            return

        min_, max_ = a.bounds()
        def to_attack_space(x):
            # map from [min_, max_] to [-1, +1]
            a = (min_ + max_) / 2
            b = (max_ - min_) / 2
            x = (x - a) / b

            # from [-1, +1] to approx. (-1, +1)
            x = x * 0.999999

            # from (-1, +1) to (-inf, +inf)
            return np.arctanh(x)

        def to_model_space(x):
            """Transforms an input from the attack space
            to the model space. This transformation and
            the returned gradient are elementwise."""

            # from (-inf, +inf) to (-1, +1)
            x = np.tanh(x)

            grad = 1 - np.square(x)

            # map from (-1, +1) to (min_, max_)
            a = (min_ + max_) / 2
            b = (max_ - min_) / 2
            x = x * b + a

            grad = grad * b
            return x, grad

        # variables representing inputs in attack space will be
        # prefixed with att_
        att_original = to_attack_space(a.unperturbed)
        
        # will be close but not identical to a.original_image
        reconstructed_original, _ = to_model_space(att_original)

        # the binary search finds the smallest const for which we
        # find an adversarial
        const = initial_const
        lower_bound = 0
        upper_bound = np.inf
        # pert_image = reconstructed_original - norm_mean.view(1,3,1,1).cuda()
        # pert_image /= norm_std.view(1,3,1,1).cuda()
        true_logits, _ = a.forward_one(reconstructed_original)
        true_label = np.argmax(true_logits)
        # Binary search for constant
        start_tau = 1.0
        for binary_search_step in range(binary_search_steps):
            if binary_search_step == binary_search_steps - 1 and \
                    binary_search_steps >= 10:
                # in the last binary search step, use the upper_bound instead
                # TODO: find out why... it's not obvious why this is useful
                const = min(1e10, upper_bound)

            # found adv with the current const

            att_warmstart = att_original
            tau = start_tau
            while tau > 1. / 256:
                found_adv = False
                att_perturbation = np.zeros_like(att_original)
                # create a new optimizer to minimize the perturbation
                optimizer = AdamOptimizer(att_perturbation.shape)
                loss_at_previous_check = np.inf
                for iteration in range(max_iterations):

                    x, dxdp = to_model_space(att_warmstart + att_perturbation)
                    # pert_image = x - norm_mean.view(1,3,1,1).cuda()
                    # pert_image /= norm_std.view(1,3,1,1).cuda()
                    logits, i = a.forward_one(x)
                    false_label = np.argmax(logits)
                    is_adv = not (false_label == true_label)

                    loss, dldx, adv_loss, distance = self.loss_function(
                        const, tau, a, x, logits, reconstructed_original,
                        confidence, min_, max_)

                    check_loss = logits[true_label] - logits[false_label]

                    # backprop the gradient of the loss w.r.t. x further
                    # to get the gradient of the loss w.r.t. att_perturbation
                    assert dldx.shape == x.shape
                    assert dxdp.shape == x.shape
                    # we can do a simple elementwise multiplication, because
                    # grad_x_wrt_p is a matrix of elementwise derivatives
                    # (i.e. each x[i] w.r.t. p[i] only, for all i) and
                    # grad_loss_wrt_x is a real gradient reshaped as a matrix
                    gradient = dldx * dxdp

                    att_perturbation += optimizer(gradient, learning_rate)
                    x, dxdp = to_model_space(att_warmstart + att_perturbation)

                    if is_adv:
                        # Tau + binary search was successful but continuing opt
                        found_adv = True

                    if abort_early and \
                            iteration % (np.ceil(max_iterations / 10)) == 0:
                        # after each tenth of the iterations, check progress
                        if not (loss_at_previous_check - loss > 0.0001):
                            break  # stop Adam if there has not been progress
                        loss_at_previous_check = loss

                if not found_adv:
                    # This constant is fine, just that we broke out of tau
                    if tau < start_tau:
                        found_adv = True
                    start_tau = tau
                    break

                else:
                    actualtau = np.max(np.abs(x - reconstructed_original))
                    if actualtau < tau:
                        tau = actualtau
                    tau = tau * tau_decrease_factor
                att_warmstart = to_attack_space(x)

            if found_adv:
                upper_bound = const
            else:
                lower_bound = const

            if upper_bound == np.inf:
                # exponential search
                const *= 10
            else:
                # binary search
                const = (lower_bound + upper_bound) / 2

    @classmethod
    def loss_function(cls, const, tau, a, x, logits, reconstructed_original,
                      confidence, min_, max_):
        """Returns the loss and the gradient of the loss w.r.t. x,
        assuming that logits = model(x)."""

        targeted = a.target_class is not None
        if targeted:
            c_minimize = cls.best_other_class(logits, a.target_class)
            c_maximize = a.target_class
        else:
            c_minimize = a.original_class
            c_maximize = cls.best_other_class(logits, a.original_class)

        is_adv_loss = logits[c_minimize] - logits[c_maximize]

        # is_adv is True as soon as the is_adv_loss goes below 0
        # but sometimes we want additional confidence
        is_adv_loss += confidence
        is_adv_loss = max(0, is_adv_loss)

        s = max_ - min_
        # squared_l2_distance = np.sum((x - reconstructed_original)**2) / s**2
        linf_distance = np.sum(
            np.maximum(np.abs(x - reconstructed_original) - tau, 0))

        total_loss = linf_distance + const * is_adv_loss

        # calculate the gradient of total_loss w.r.t. x
        logits_diff_grad = np.zeros_like(logits)
        logits_diff_grad[c_minimize] = 1
        logits_diff_grad[c_maximize] = -1
        is_adv_loss_grad = a.backward_one(logits_diff_grad, x)
        assert is_adv_loss >= 0
        if is_adv_loss == 0:
            is_adv_loss_grad = 0

        squared_l2_distance_grad = (2 / s ** 2) * (x - reconstructed_original)
        linf_distance_grad = np.sign(x - reconstructed_original) * (
                np.abs(x - reconstructed_original) - tau > 0)
        total_loss_grad = linf_distance_grad + const * is_adv_loss_grad
        return total_loss, total_loss_grad, is_adv_loss, linf_distance

    @staticmethod
    def best_other_class(logits, exclude):
        """Returns the index of the largest logit, ignoring the class that
        is passed as `exclude`."""
        other_logits = logits - onehot_like(logits, exclude, value=np.inf)
        return np.argmax(other_logits)


class AdamOptimizer:
    """Basic Adam optimizer implementation that can minimize w.r.t.
    a single variable.
    Parameters
    ----------
    shape : tuple
        shape of the variable w.r.t. which the loss should be minimized
    """

    def __init__(self, shape):
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0

    def __call__(self, gradient, learning_rate,
                 beta1=0.9, beta2=0.999, epsilon=10e-8):
        """Updates internal parameters of the optimizer and returns
        the change that should be applied to the variable.
        Parameters
        ----------
        gradient : `np.ndarray`
            the gradient of the loss w.r.t. to the variable
        learning_rate: float
            the learning rate in the current iteration
        beta1: float
            decay rate for calculating the exponentially
            decaying average of past gradients
        beta2: float
            decay rate for calculating the exponentially
            decaying average of past squared gradients
        epsilon: float
            small value to avoid division by zero
        """
        
        self.t += 1

        self.m = beta1 * self.m + (1 - beta1) * gradient
        self.v = beta2 * self.v + (1 - beta2) * gradient ** 2

        bias_correction_1 = 1 - beta1 ** self.t
        bias_correction_2 = 1 - beta2 ** self.t

        m_hat = self.m / bias_correction_1
        v_hat = self.v / bias_correction_2

        return -learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)