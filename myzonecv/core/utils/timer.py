import time
from collections import OrderedDict


class TimerError(Exception):
    def __init__(self, msg):
        self.msg = msg
        super().__init__(msg)


class Timer:
    def __init__(self, start=False, print_tmpl='{:.3f}'):
        self.running = False
        self.print_tmp = print_tmpl
        self.start_time = None
        self.last_time = None
        self.last_var = None
        self.mark_time = OrderedDict()
        self.mark_var = OrderedDict()

        if start:
            self.start()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.stop()

    def start(self, keep_var=None, mark=None):
        if not self.running:
            self.start_time = time.time()
            self.running = True
        self.last_time = time.time()
        self.last_var = keep_var

        if mark is not None:
            self.mark_time[mark] = self.last_time
            self.mark_var[mark] = self.last_var

    def restart(self, keep_var=None, mark=None):
        if self.running:
            self.stop()
        self.start(keep_var=keep_var, mark=mark)

    def stop(self):
        # print(self.print_tmp.format(self.since_last_check()))
        self.running = False
        self.clear_mark()

    def clear_mark(self):
        self.mark_time.clear()
        self.mark_var.clear()

    def since_start(self, keep_var=None, mark=None):
        if not self.running:
            raise TimerError('timer is not running')
        self.last_time = time.time()
        self.last_var = keep_var

        if mark is not None:
            self.mark_time[mark] = self.last_time
            self.mark_var[mark] = self.last_var

        return self.last_time - self.start_time

    def since_last_check(self, keep_var=None, mark=None, return_var=False):
        if not self.running:
            raise TimerError('timer is not running')
        prev_time = self.last_time
        ret_var = self.last_var
        self.last_time = time.time()
        self.last_var = keep_var

        if mark is not None:
            self.mark_time[mark] = self.last_time
            self.mark_var[mark] = self.last_var

        return (self.last_time - prev_time, ret_var) if return_var else self.last_time - prev_time

    def since_mark(self, mk, keep_var=None, mark=None, return_var=False):
        if not self.running:
            raise TimerError('timer is not running')
        if not isinstance(mk, (tuple, list)):
            mk = [mk]
        for m in mk:
            if m not in self.mark_time:
                raise TimerError(f'mark {m} is not found')

        prev_time = [self.mark_time[m] for m in mk]
        ret_val = [self.mark_var[m] for m in mk]
        self.last_time = time.time()
        self.last_var = keep_var

        if mark is not None:
            self.mark_time[mark] = self.last_time
            self.mark_var[mark] = self.last_var

        ret = [(self.last_time - pt, rv) if return_var else self.last_time - pt for pt, rv in zip(prev_time, ret_val)]
        if len(ret) == 1:
            ret = ret[0]
        return ret
