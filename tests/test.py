class Test:
    x = 0
    y = 1

    def __iter__(self):
        return iter((self.x, self.y))


t = Test()

x, y = t

print(x, y)
