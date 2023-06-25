class Val():  # scalar
    def __init__(self, val: float):
        self.value = val

    def __str__(self) -> str:
        return f"{self.value}"

    def __add__(self, other):
        return Val(self.value + other.value)

    def __radd__(self, other):
        return Val(self.value + other.value)

    def __mul__(self, other):
        return Val(self.value * other.value)

    def __rmul__(self, other):
        return Val(self.value * other.value)

    def __pow__(self, other):
        return Val(self.value ** other.value)

    def __rpow__(self, other):
        return Val(other.value ** self.value)

    def diff(self):
        pass


if __name__ == "__main__":
    x = Val(3.0)
    y = Val(5.0)

    exp1 = Val(3)*Val(1)**Val(2) + Val(5)*Val(1) + Val(9)  # 3x^2 + 5x + 9; let x = 1

    print(y**Val(2))
    print(exp1)
