# Portions adapted from lorentz.py by hycoclip (CC BY-NC 4.0)
# Original source: https://github.com/PalAvik/hycoclip.git

"""
Implementation of common operations for the Lorentz model of hyperbolic geometry.
This model represents a hyperbolic space of `d` dimensions on the upper-half of
a two-sheeted hyperboloid in a Euclidean space of `(d+1)` dimensions.

Hyperbolic geometry has a direct connection to the study of special relativity
theory -- implementations in this module borrow some of its terminology. The axis
of symmetry of the Hyperboloid is called the _time dimension_, while all other
axes are collectively called _space dimensions_.

All functions implemented here only input/output the space components, while
while calculating the time component according to the Hyperboloid constraint:

    `x_time = torch.sqrt(1 / curv + torch.norm(x_space) ** 2)`
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as Fs
from torch import Tensor

import functools
from typing import Callable, Any, Union, Tuple


# In hyperbolic methods, we partially use float32 no matter what is defined in the configration file, otherwise it will causes overflow error which result in nan.
def hyperbolic_float32(func: Callable) -> Callable:
    """
    A decorator which changes the precision to float32 during hyperbolic functions，while keeps the data precision of the outputs unchanged.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Union[Tensor, Tuple[Tensor, ...]]:
        # Save original input data precision.
        input_dtypes = []
        for arg in args:
            if isinstance(arg, Tensor):
                input_dtypes.append(arg.dtype)
        
        # Change precision to float32
        float32_args = []
        for arg in args:
            if isinstance(arg, Tensor):
                float32_args.append(arg.float() if arg.dtype != torch.float32 else arg)
            else:
                float32_args.append(arg)
        
        float32_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, Tensor):
                float32_kwargs[key] = value.float() if value.dtype != torch.float32 else value
            else:
                float32_kwargs[key] = value
        
        # Call functions under float32 precision
        result = func(*float32_args, **float32_kwargs)
        
        # Restore to original precision.
        if isinstance(result, tuple):
            restored_result = []
            for i, res in enumerate(result):
                if isinstance(res, Tensor):
                    target_dtype = input_dtypes[0] if input_dtypes else res.dtype
                    restored_result.append(res.to(target_dtype))
                else:
                    restored_result.append(res)
            return tuple(restored_result)
        elif isinstance(result, Tensor):
            target_dtype = input_dtypes[0] if input_dtypes else result.dtype
            return result.to(target_dtype)
        else:
            return result
    
    return wrapper

@hyperbolic_float32
def pairwise_inner(x: Tensor, y: Tensor, curv: float | Tensor = 1.0):
    """
    Compute pairwise Lorentzian inner product between input vectors.

    Args:
        x: Tensor of shape `(B1, D)` giving a space components of a batch
            of vectors on the hyperboloid.
        y: Tensor of shape `(B2, D)` giving a space components of another
            batch of points on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B1, B2)` giving pairwise Lorentzian inner product
        between input vectors.
    """

    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1, keepdim=True))
    xyl = x @ y.T - x_time @ y_time.T
    return xyl

@hyperbolic_float32
def pairwise_dist(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-6
) -> Tensor:
    """
    Compute the pairwise geodesic distance between two batches of points on
    the hyperboloid.

    Args:
        x: Tensor of shape `(B1, D)` giving a space components of a batch
            of point on the hyperboloid.
        y: Tensor of shape `(B2, D)` giving a space components of another
            batch of points on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B1, B2)` giving pairwise distance along the geodesics
        connecting the input points.
    """
    

    # Ensure numerical stability in arc-cosh by clamping input.
    c_xyl = -curv * pairwise_inner(x, y, curv)
    _distance = torch.acosh(torch.clamp(c_xyl, min=1 + eps))
    return _distance / curv**0.5

@hyperbolic_float32
def exp_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-6) -> Tensor:
    """
    Map points from the tangent space at the vertex of hyperboloid, on to the
    hyperboloid. This mapping is done using the exponential map of Lorentz model.

    Args:
        x: Tensor of shape `(B, D)` giving batch of Euclidean vectors to project
            onto the hyperboloid. These vectors are interpreted as velocity
            vectors in the tangent space at the hyperboloid vertex.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid division by zero.

    Returns:
        Tensor of same shape as `x`, giving space components of the mapped
        vectors on the hyperboloid.
    """
    


    rc_xnorm = curv**0.5 * torch.norm(x, dim=-1, keepdim=True)

    # Ensure numerical stability in sinh by clamping input.
    sinh_input = torch.clamp(rc_xnorm, min=eps, max=math.asinh(2**15))
    _output = torch.sinh(sinh_input) * x / torch.clamp(rc_xnorm, min=eps)
    return _output

@hyperbolic_float32
def log_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-6) -> Tensor:
    """
    Inverse of the exponential map: map points from the hyperboloid on to the
    tangent space at the vertex, using the logarithmic map of Lorentz model.

    Args:
        x: Tensor of shape `(B, D)` giving space components of points
            on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid division by zero.

    Returns:
        Tensor of same shape as `x`, giving Euclidean vectors in the tangent
        space of the hyperboloid vertex.
    """
    

    # Calculate distance of vectors to the hyperboloid vertex.
    rc_x_time = torch.sqrt(1 + curv * torch.sum(x**2, dim=-1, keepdim=True))
    _distance0 = torch.acosh(torch.clamp(rc_x_time, min=1 + eps))

    rc_xnorm = curv**0.5 * torch.norm(x, dim=-1, keepdim=True)
    _output = _distance0 * x / torch.clamp(rc_xnorm, min=eps)
    return _output

@hyperbolic_float32
def half_aperture(
    x: Tensor, curv: float | Tensor = 1.0, min_radius: float = 0.1, eps: float = 1e-6
) -> Tensor:
    """
    Compute the half aperture angle of the entailment cone formed by vectors on
    the hyperboloid. The given vector would meet the apex of this cone, and the
    cone itself extends outwards to infinity.

    Args:
        x: Tensor of shape `(B, D)` giving a batch of space components of
            vectors on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        min_radius: Radius of a small neighborhood around vertex of the hyperboloid
            where cone aperture is left undefined. Input vectors lying inside this
            neighborhood (having smaller norm) will be projected on the boundary.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B, )` giving the half-aperture of entailment cones
        formed by input vectors. Values of this tensor lie in `(0, pi/2)`.
    """
    

    # Ensure numerical stability in arc-sin by clamping input.
    asin_input = 2 * min_radius / (torch.norm(x, dim=-1) * curv**0.5 + eps)
    _half_aperture = torch.asin(torch.clamp(asin_input, min=-1 + eps, max=1 - eps))

    return _half_aperture

@hyperbolic_float32
def oxy_angle(x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-6):
    """
    Given two vectors `x` and `y` on the hyperboloid, compute the exterior
    angle at `x` in the hyperbolic triangle `Oxy` where `O` is the origin
    of the hyperboloid.

    This expression is derived using the Hyperbolic law of cosines.

    Args:
        x: Tensor of shape `(B, D)` giving a batch of space components of
            vectors on the hyperboloid.
        y: Tensor of same shape as `x` giving another batch of vectors.
        curv: Positive scalar denoting negative hyperboloid curvature.

    Returns:
        Tensor of shape `(B, )` giving the required angle. Values of this
        tensor lie in `(0, pi)`.
    """
    

    # Calculate time components of inputs (multiplied with `sqrt(curv)`):
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1))

    # Calculate lorentzian inner product multiplied with curvature. We do not use
    # the `pairwise_inner` implementation to save some operations (since we only
    # need the diagonal elements).
    c_xyl = curv * (torch.sum(x * y, dim=-1) - x_time * y_time)

    # Make the numerator and denominator for input to arc-cosh, shape: (B, )
    acos_numer = y_time + c_xyl * x_time
    acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))

    acos_input = acos_numer / (torch.norm(x, dim=-1) * acos_denom + eps)
    _angle = torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))

    return _angle

def entailment_loss(fathers, sons, curv):
    """"
    Compute the entailment constraint loss induced by hyperbolic cones.

    This loss function measures the hierarchical relationship between parent and child vectors
    in hyperbolic space using cone-based constraints.

    Args:
        fathers: A PyTorch tensor with shape (..., dim), where each vector resides at a 
                 shallower level in the hierarchy.
        sons: A PyTorch tensor with shape (..., dim), where each vector resides at a 
              deeper level in the hierarchy.
        curv: The curvature parameter of the Lorentz model.

    Returns:
        A tensor containing the entailment constraint loss value induced by hyperbolic cones.
    """

    fathers_shape, sons_shape = fathers.shape, sons.shape
    assert fathers_shape == sons_shape, f"fathers_shape{fathers_shape} is inconsistent with sons_shape{sons_shape}"
    fathers = fathers.reshape(-1, fathers_shape[-1])
    sons = sons.reshape(-1, sons_shape[-1])
    _angle = oxy_angle(fathers, sons, curv=curv)
    _aperture = half_aperture(fathers, curv=curv)
    loss = torch.clamp(_angle - _aperture, min=0).mean()
    return loss

def tree_aligning_loss(text_tree_nodes, image_tree_nodes, curv_i, curv_t, curv_m, scaling_factor_i=1., scaling_factor_t=1.):
    """
    Computes the alignment loss between corresponding paths in text and image tree structures.

    This loss measures the degree of misalignment between hierarchical representations extracted from
    text and image modalities by comparing their projected paths in hyperbolic space.

    Args:
        text_tree_nodes: A tensor of shape `(bsz, path_len, dim)` representing a path from the root to a target node
                         in the text hierarchy tree (root node excluded).
        image_tree_nodes: A tensor of shape `(bsz, path_len, dim)` representing the corresponding path in the image
                          tree, extracted using the text tree nodes as queries.
        curv_i: Curvature parameter of the image manifold.
        curv_t: Curvature parameter of the text manifold.
        curv_m: Curvature parameter of the intermediate manifold used for alignment.
        scaling_factor_i: Scaling factor applied to image features before hyperbolic projection.
        scaling_factor_t: Scaling factor applied to text features before hyperbolic projection.

    Returns:
        A scalar tensor containing the computed tree alignment loss value.
    """

    shape = text_tree_nodes.shape
    text_tree_nodes_hyp = exp_map0(scaling_factor_t * text_tree_nodes.reshape(-1, shape[-1]), curv=curv_t).reshape(shape)
    image_tree_nodes_hyp = exp_map0(scaling_factor_i * image_tree_nodes.reshape(-1, shape[-1]), curv=curv_i).reshape(shape)
    hier_path_consistency_loss = entailment_loss(image_tree_nodes_hyp[:, :-1, :], image_tree_nodes_hyp[:, 1:, :], curv=curv_i) + entailment_loss(text_tree_nodes_hyp[:, :-1, :], text_tree_nodes_hyp[:, 1:, :], curv=curv_t)
    text_tree_nodes_commonhyp = exp_map0(scaling_factor_t * text_tree_nodes.reshape(-1, shape[-1]), curv=curv_m).reshape(shape)
    image_tree_nodes_commonhyp = exp_map0(scaling_factor_i * image_tree_nodes.reshape(-1, shape[-1]), curv=curv_m).reshape(shape)
    image_text_consistency_loss = entailment_loss(text_tree_nodes_commonhyp, image_tree_nodes_commonhyp, curv=curv_m)

    return hier_path_consistency_loss + image_text_consistency_loss
    


class OptimalC3Function(torch.autograd.Function):
    scaling_factor = 1
    @staticmethod
    def forward(ctx, c1, c2, r1, r2):
        with torch.no_grad():
            c3_optimal = solve_c3_no_grad(c1, c2, r1, r2)
        
        ctx.save_for_backward(c1, c2, c3_optimal, r1, r2)
        return c3_optimal
    
    @staticmethod
    def backward(ctx, grad_output):
        verbose = False
        boundary_optimal = False
        c1, c2, c3_optimal, r1, r2 = ctx.saved_tensors
        to_ret = [None, None, None, None]
        if c1.item() < c2.item():
            minc, maxc = c1, c2
            minpos, maxpos = 0, 1
        else:
            minc, maxc = c2, c2
            minpos, maxpos = 1, 0

        if c3_optimal - minc < 1e-7:
            boundary_optimal = True
            to_ret[minpos] = torch.tensor(1.)
        elif maxc - c3_optimal < 1e-7:
            boundary_optimal = True
            to_ret[maxpos] = torch.tensor(1.)       
        if boundary_optimal:
            # if accidentally get an boundary solution(e.g. due to precision or some other problems), we shouldn't use the implicity function to get the gradient.
            if verbose:
                print('warning, optimal at boundary point!')
            ret1, ret2, ret3, ret4 = to_ret
            return ret1, ret2, ret3, ret4


        # Calculate the graident using implicit function therom.
        with torch.enable_grad():
            c1_temp = c1.detach().requires_grad_(True)
            c2_temp = c2.detach().requires_grad_(True)
            c3_temp = c3_optimal.detach().requires_grad_(True)
            
            partial_val = partial_f_partial_c3(c3_temp, c1_temp, c2_temp, r1, r2)
            
            df_dc3 = torch.autograd.grad(partial_val, c3_temp, retain_graph=True)[0]
            df_dc1 = torch.autograd.grad(partial_val, c1_temp, retain_graph=True)[0]
            df_dc2 = torch.autograd.grad(partial_val, c2_temp)[0]
            
            # implicit function therom: dc3/dc1 = -df/dc1 / df/dc3
            dc3_dc1 = -df_dc1 / (df_dc3 + 1e-12)
            dc3_dc2 = -df_dc2 / (df_dc3 + 1e-12)
        
        grad_c1 = grad_output * OptimalC3Function.scaling_factor * dc3_dc1
        grad_c2 = grad_output * OptimalC3Function.scaling_factor * dc3_dc2
        return grad_c1, grad_c2, None, None  # r doesn't need a gradient.

def d(c3, ci, r, eps=1e-12):
    sqrt_ci = torch.sqrt(ci + eps)
    sqrt_c3 = torch.sqrt(c3 + eps)
    arg = (sqrt_c3 - sqrt_ci) * r
    numerator = -sqrt_ci + 2 * sqrt_c3 * torch.cosh(arg)
    denominator = 2 * sqrt_ci * c3
    return numerator / (denominator + eps)

def f(c3, c1, c2, r1, r2):
    return d(c3, c1, r1) + d(c3, c2, r2)

def partial_d_partial_c3(c3, c_i, r_i):
    '''
    Calculate the partial derivates of  d(c3, c_i; r) with respect to c3
    '''
    
    sqrt_c3 = torch.sqrt(c3)
    sqrt_ci = torch.sqrt(c_i)
    
    delta = sqrt_c3 - sqrt_ci
    arg = delta * r_i
    
    sinh_val = torch.sinh(arg)
    cosh_val = torch.cosh(arg)
    
    numerator = sqrt_ci - sqrt_c3 * cosh_val + c3 * r_i * sinh_val
    denominator = 2 * sqrt_ci * (c3 ** 2)
    
    return numerator / denominator

def partial_f_partial_c3(c3, c1, c2, r1, r2):
    """
    Calculate the partial derivates of f(c3; c1, c2, r1, r2) with respect to c3
    
    Based on:
    f(c3; c1, c2, r1, r2) = d(c3, c1; r1) + d(c3, c2; r2)
    """
    return (partial_d_partial_c3(c3, c1, r1) + 
            partial_d_partial_c3(c3, c2, r2))

def solve_c3_no_grad(c1, c2, r1, r2):
    """Numerical solve with gradient untracked"""
    eps_val = 1e-12
    max_iter = 100
    tol = 1e-6
    
    # search interval
    min_c = torch.min(c1, c2)
    max_c = torch.max(c1, c2)
    a = min_c * 0.9
    b = max_c * 1.1
    
    # harmonic mean as the init value
    c3 = 2 * c1 * c2 / (c1 + c2 + eps_val)
    c3 = torch.clamp(c3, min=a, max=b)
    
    # golden section search
    golden_ratio = (torch.sqrt(torch.tensor(5.0)) - 1) / 2
    
    for i in range(max_iter):
        c = a + (1 - golden_ratio) * (b - a)
        d = a + golden_ratio * (b - a)
        
        fc = f(c, c1, c2, r1, r2)
        fd = f(d, c1, c2, r1, r2)
        
        if fc < fd:
            b = d
            c3 = c
        else:
            a = c
            c3 = d
        
        # check convergence
        if (b - a) < tol:
            break
    
    # return optimal value
    c3 = (a + b) / 2
    return c3

def find_optimal_c3(c1, c2, r1, r2):
    """
    This function calls a customized autograd fucntion and allows different constants r1, r2 for two manifolds.
    In practice, we sample a batch and calculate the corresponding r (but don;t record gradients). see train_one_epoch in enginee.py for more details.
    """
    return OptimalC3Function.apply(c1, c2, r1, r2)

def test_gradient_flow():
    """ Not used. Just a test case whether our algorithm still works when values of the two curvautes are very close."""
    c1 = torch.tensor(0.446171, requires_grad=True)
    c2 = torch.tensor(0.454308, requires_grad=True)
    r = torch.tensor(10.)
    r1 = torch.tensor(10)
    r2 = torch.tensor(50)
    
    print("=== Test1: Only calculate c3 ===")
    c3_optimal = find_optimal_c3(c1, c2, r1, r2)
    print(f"c3_optimal: {c3_optimal.item():.6f}")
    print(f"c1.grad: {c1.grad}")  # Should be None.
    print(f"c2.grad: {c2.grad}")  # Should be None.
    
    print("\n=== Test2: Backward. ===")
    if c1.grad is not None:
        c1.grad.zero_()
    if c2.grad is not None:
        c2.grad.zero_()
    
    c3_optimal = find_optimal_c3(c1, c2, r1, r2)
    
    loss = c3_optimal.sum()  # 
    loss.backward()
    print(f"c1:{c1}, c2:{c2}, r1:{r1}, r2:{r2}")
    print(f"c3_optimal: {c3_optimal.item():.6f}")
    print(f"f(c1): {f(c1,c1,c2,r1, r2):.6f}")
    print(f"f(c2): {f(c2,c1,c2,r1,r2):.6f}")
    print(f"f(c3_optimal): {f(c3_optimal,c1,c2,r1,r2):.6f}")
    print(f"loss: {loss.item():.6f}")
    if c1.grad is not None:
        print(f"c1.grad: {c1.grad.item():.6f}") 
    else:
        print(f"c1.grad: None")
    if c2.grad is not None:
        print(f"c2.grad: {c2.grad.item():.6f}")
    else:
        print(f"c2.grad: None")

if __name__ == "__main__":
    test_gradient_flow()

