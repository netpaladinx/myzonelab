

size = 10
stride = 2
factor = 2

class Item:
    def __init__(self, value, step):
        self.value = value
        self.step = step

class Memory:
    def __init__(self, size, stride, factor):
        self.size = size
        self.stride = stride
        self.factor0 = factor
        self.factor = factor
        
        self.queue = []
        self.pointer = 0
        self._n_deleted_one_scan = 0

    def push(self, a):
        self.queue.append(Item(value=a, step=-1))
        if len(self.queue) > 1:
            self.queue[-2].step = 1
        
        self._check_queue()
        self.print()
            
    def _check_queue(self):
        while len(self.queue) > size:
            cur_step = self.queue[self.pointer].step
            if self._is_removable(self.pointer, cur_step):
                next_step = self.queue[self.pointer+1].step
                if self._is_removable(self.pointer+1, next_step) and self.pointer+3 < len(self.queue):
                    next2_step = self.queue[self.pointer+2].step
                    if cur_step < next_step + next2_step:
                        new_step = cur_step + next_step
                        self.queue[self.pointer].step = new_step
                        del self.queue[self.pointer+1]
                        self._n_deleted_one_scan += 1
                else:
                    if next_step == -1:
                        new_step = -1
                    else:
                        new_step = cur_step + next_step
                    self.queue[self.pointer].step = new_step
                    del self.queue[self.pointer+1]
                    self._n_deleted_one_scan += 1
            
            self.pointer += 1
            
            if self.pointer + 2 >= len(self.queue):
                self._reset_pointer()
            
    def _is_removable(self, index, step):
        return step <= self.factor ** (max(0, size - index) // stride)
    
    def _reset_pointer(self):
        if self._n_deleted_one_scan == 0:
            self.factor += 1
            
        self.pointer = 0
        self._n_deleted_one_scan = 0
    
    def print(self):
        print(f'Size [{len(self.queue)}] Factor [{self.factor}]: ' + ', '.join([f'{item.value} [{item.step}]' for item in self.queue]))
    

memory = Memory(size, stride, factor)
for i in range(1000):
    memory.push(i)