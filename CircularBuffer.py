class CircularBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = [0] * size
        self.index = 0
        self.full = False
        # print(f"CircularBuffer initialized with size {size}")

    def append(self, value):
        self.buffer[self.index] = value
        self.index = (self.index + 1) % self.size
        if self.index == 0:
            self.full = True
        #print(f"Appended value {value} to buffer at index {self.index}")

    def get(self):
        if self.full:
            return self.buffer[self.index:] + self.buffer[:self.index]
        else:
            return self.buffer[:self.index]
