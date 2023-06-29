import math


class Val:  # scalar
    def __init__(self, val: float, parents=None):
        parents = [] if not parents else parents
        self.value = val
        self.parents = parents
        self.grad = 0.0

    def __str__(self) -> str:
        return f"Var(v={self.value}, grad={self.grad})"

    def __add__(self: "Val", other: "Val") -> "Val":
        # parent: (parent, gradient)
        return Val(self.value + other.value, parents=[(self, 1.0),
                                                      (other, 1.0)])

    def __mul__(self: "Val", other: "Val") -> "Val":
        return Val(self.value * other.value, parents=[(self, other.value),
                                                      (other, self.value)])

    def __pow__(self: "Val", power: float):
        return Val(self.value**power, parents=[(self, power *
                                                self.value**(power-1))])

    def __neg__(self: "Val") -> "Val":
        return Val(-1*self.value)

    def __sub__(self: "Val", other: "Val") -> "Val":
        return self + (-other)

    def __truediv__(self: "Val", other: "Val") -> "Val":
        return self * other**(-1)

    def tanh(self: "Val") -> "Val":
        return Val(math.tanh(self.value),
                   parents=[(self, 1 - math.tanh(self.value)**2)])

    def backward(self):
        self.backprop(1.0)

    def backprop(self, gradient: float):
        self.grad += gradient
        if not self.parents:
            return
        for parent, grad in self.parents:
            parent.backprop(grad * gradient)  # chain rule


if __name__ == "__main__":
    a = Val(3.0)
    b = Val(5.0)
    x = Val(1.0)

    exp1 = (
        a * x ** 2 + b * x + Val(9)
    )  # 3x^2 + 5x + 9; let x = 1
    exp1.backward()

    for v in [x, exp1]:
        print(v)
