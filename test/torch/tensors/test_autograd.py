import torch
import syft

from syft.frameworks.torch.tensors.interpreters import AutogradTensor


def test_wrap(hook):
    """
    Test the .on() wrap functionality for AutogradTensor
    """

    x_tensor = torch.Tensor([1, 2, 3])
    x = AutogradTensor().on(x_tensor)

    assert isinstance(x, torch.Tensor)
    assert isinstance(x.child, AutogradTensor)
    assert isinstance(x.child.child, torch.Tensor)


def test_add_backwards(workers):
    alice = workers['alice']
    a = torch.tensor([3, 2., 0], requires_grad=True)
    b = torch.tensor([1, 2., 3], requires_grad=True)

    ptr_a = a.send(alice, local_autograd=True)
    ptr_b = b.send(alice, local_autograd=True)

    a_torch = torch.tensor([3, 2.0, 0], requires_grad=True)
    b_torch = torch.tensor([1, 2.0, 3], requires_grad=True)

    c = ptr_a + ptr_b
    c_torch = a_torch + b_torch

    c.backward(torch.ones(c.shape).send(alice))
    c_torch.backward(torch.ones(c_torch.shape))

    assert (ptr_a.grad.get() == a_torch.grad).all()
    assert (ptr_b.grad.get() == b_torch.grad).all()


def test_mul_backwards(workers):
    alice = workers['alice']
    a = torch.tensor([3, 2., 0], requires_grad=True)
    b = torch.tensor([1, 2., 3], requires_grad=True)

    ptr_a = a.send(alice, local_autograd=True)
    ptr_b = b.send(alice, local_autograd=True)

    a_torch = torch.tensor([3, 2.0, 0], requires_grad=True)
    b_torch = torch.tensor([1, 2.0, 3], requires_grad=True)

    c = ptr_a * ptr_b
    c_torch = a_torch * b_torch
    
    c.backward(torch.ones(c.shape).send(alice))
    c_torch.backward(torch.ones(c_torch.shape))

    assert (ptr_a.grad.get() == a_torch.grad).all()
    assert (ptr_b.grad.get() == b_torch.grad).all()


def test_sqrt_backwards(workers):
    alice = workers['alice']
    
    a = torch.tensor([3, 2., 0], requires_grad=True)
    ptr_a = a.send(alice, local_autograd=True)
    
    a_torch = torch.tensor([3, 2.0, 0], requires_grad=True)

    c = ptr_a.sqrt()
    c_torch = a_torch.sqrt()

    c.backward(torch.ones(c.shape).send(alice))
    c_torch.backward(torch.ones_like(c_torch))

    # Have to do .child.grad here because .grad doesn't work on Wrappers yet
    assert (ptr_a.grad.get() == a_torch.grad).all()


def test_asin_backwards(workers):
    alice = workers['alice']
    
    a = torch.tensor([0.3, 0.2, 0], requires_grad=True)
    ptr_a = a.send(alice, local_autograd=True)
    
    a_torch = torch.tensor([0.3, 0.2, 0], requires_grad=True)

    c = ptr_a.asin()
    c_torch = a_torch.asin()

    c.backward(torch.ones(c.shape).send(alice))
    c_torch.backward(torch.ones_like(c_torch))

    # Have to do .child.grad here because .grad doesn't work on Wrappers yet
    assert (ptr_a.grad.get() == a_torch.grad).all()


def test_sin_backwards(workers):
    alice = workers['alice']
    
    a = torch.tensor([0.3, 0.2, 0], requires_grad=True)
    ptr_a = a.send(alice, local_autograd=True)
    
    a_torch = torch.tensor([0.3, 0.2, 0], requires_grad=True)

    c = ptr_a.sin()
    c_torch = a_torch.sin()

    c.backward(torch.ones(c.shape).send(alice))
    c_torch.backward(torch.ones_like(c_torch))

    # Have to do .child.grad here because .grad doesn't work on Wrappers yet
    assert (ptr_a.grad.get() == a_torch.grad).all()


def test_sinh_backwards(workers):
    alice = workers['alice']
    
    a = torch.tensor([0.3, 0.2, 0], requires_grad=True)
    ptr_a = a.send(alice, local_autograd=True)
    
    a_torch = torch.tensor([0.3, 0.2, 0], requires_grad=True)

    c = ptr_a.sinh()
    c_torch = a_torch.sinh()

    c.backward(torch.ones(c.shape).send(alice))
    c_torch.backward(torch.ones_like(c_torch))

    # Have to do .child.grad here because .grad doesn't work on Wrappers yet
    assert (ptr_a.grad.get() == a_torch.grad).all()