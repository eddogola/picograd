"""
simple autograd engine
"""

import math


class Val:  # scalar
    """
    A scalar value with a backpropagation gradient.
    """

    def __init__(self, val: float, parents=None):
        parents = [] if not parents else parents
        self.value = val
        self.parents = parents
        self.grad = 0.0

    def __str__(self) -> str:
        return f"Var(v={self.value}, grad={self.grad})"

    def __add__(self: "Val", other: "Val") -> "Val":
        # parent: (parent, gradient)
        return Val(self.value + other.value, parents=[(self, 1.0), (other, 1.0)])

    def __mul__(self: "Val", other: "Val") -> "Val":
        return Val(
            self.value * other.value,
            parents=[(self, other.value), (other, self.value)],
        )

    def __pow__(self: "Val", power: float):
        return Val(
            self.value**power,
            parents=[(self, power * self.value ** (power - 1))],
        )

    def __neg__(self: "Val") -> "Val":
        return Val(-1 * self.value)

    def __sub__(self: "Val", other: "Val") -> "Val":
        return self + (-other)

    def __truediv__(self: "Val", other: "Val") -> "Val":
        return self * other ** (-1)

    def tanh(self: "Val") -> "Val":
        """
        returns the tanh of the value, and the gradient
        """
        return Val(
            math.tanh(self.value), parents=[(self, 1 - math.tanh(self.value) ** 2)]
        )

    def backward(self):
        """
        Initiate backpropagation by backpropagating a gradient of 1.0
        """
        self.backprop(1.0)

    def backprop(self, gradient: float):
        """
        Backpropagate gradients through the graph.
        """
        self.grad += gradient
        if not self.parents:
            return
        for parent, grad in self.parents:
            parent.backprop(grad * gradient)  # chain rule


# forward mode autodiff


class DualNumber:
    """
    Dual number class
    """

    def __init__(self, val, dual=0.0):
        self.val = val
        self.dual = dual

    def __add__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.val + other.val, self.dual + other.dual)
        return DualNumber(self.val + other, self.dual)

    def __mul__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(
                self.val * other.val, self.dual * other.val + self.val * other.dual
            )
        return DualNumber(self.val * other, self.dual * other)

    def __pow__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(
                self.val**other.val,
                self.val**other.val
                * (other.dual * math.log(self.val) + other.val * self.dual / self.val),
            )

        return DualNumber(
            self.val**other, other * self.val ** (other - 1) * self.dual
        )


def forward_autodiff(func, value):
    """
    convert x into dual number, and evaluate func

    func: function to evaluate
    value(`int` or `float`): value to evaluate at
    """
    seed = DualNumber(value, 1.0)

    return func(seed)


if __name__ == "__main__":
    a = Val(3.0)
    b = Val(5.0)
    c = Val(1.0)

    # 3x^2 + 5x + 9; let x = 1
    exp1 = a * c**2 + b * c + Val(9)
    exp1.backward()

    for v in [c, exp1]:
        print(v)

    # forward mode autodiff
    print("forward mode autodiff")

    def some_func(val):
        """
        f(x) = x^2 + 2x + 1
        """
        return val**2 + val * 2 + 1

    result = forward_autodiff(some_func, 3)

    print(f"eval_result: {result.val}, dual: {result.dual}")
