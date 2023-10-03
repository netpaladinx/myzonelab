from ..registry import RUNS, RUN_SCHEDULERS
from ..utils import get_logger
from ..context import DummyContext, Context
from .base_run import BaseRun

logger = get_logger('station_thread_loop')


class Thread:
    WAITING = 'waiting'
    RUNNING = 'running'
    CLOSED = 'closed'
    FINISHED = 'finished'

    def __init__(self,
                 max_outer_rounds=None,
                 max_inner_rounds=None,
                 start_outer_round=0,
                 end_outer_round=None,
                 skip_outer_rounds=[],
                 outer_round_interval=1,
                 start_inner_round=0,
                 end_inner_round=None,
                 skip_inner_rounds=[],
                 inner_round_interval=1,
                 terminate_inner_loop_when_closed=False,
                 will_terminate_inner_loop_when_closed=False,
                 close_station_when_finished=False,
                 terminate_outer_loop_when_finished=False,
                 will_terminate_outer_loop_when_finished=False,
                 disable=False,
                 name=None,
                 **kwargs):
        self.start_outer_round = start_outer_round
        self.end_outer_round = max_outer_rounds if end_outer_round is None else end_outer_round
        self.skip_outer_rounds = skip_outer_rounds
        self.outer_round_interval = outer_round_interval
        self.start_inner_round = start_inner_round
        self.end_inner_round = max_inner_rounds if end_inner_round is None else end_inner_round
        self.skip_inner_rounds = skip_inner_rounds
        self.inner_round_interval = inner_round_interval
        self.terminate_inner_loop_when_closed = terminate_inner_loop_when_closed
        self.will_terminate_inner_loop_when_closed = will_terminate_inner_loop_when_closed
        self.close_station_when_finished = close_station_when_finished
        self.terminate_outer_loop_when_finished = terminate_outer_loop_when_finished
        self.will_terminate_outer_loop_when_finished = will_terminate_outer_loop_when_finished
        self.disable = disable
        self.name = name
        self.state = Thread.WAITING
        self.ctx = Context(name=name, **kwargs)
        self.reenter = False

    def check_if_run(self, outer_loop=None, inner_loop=None):
        if self.disable:
            return False

        assert outer_loop is not None or inner_loop is not None
        if outer_loop is not None:
            # called before inner loop
            outer_round = outer_loop.outer_round
            if outer_round < self.start_outer_round:
                return False
            if self.end_outer_round is not None and outer_round >= self.end_outer_round:
                return False
            if outer_round in self.skip_outer_rounds:
                return False
            if (outer_round - self.start_outer_round) % self.outer_round_interval != 0:
                return False
            return True
        else:
            # called in a thread
            inner_round = inner_loop.inner_round
            if inner_round < self.start_inner_round:
                return False
            if self.end_inner_round is not None and inner_round >= self.end_inner_round:
                return False
            if inner_round in self.skip_inner_rounds:
                return False
            if (inner_round - self.start_inner_round) % self.inner_round_interval != 0:
                return False
            return True

    def check_if_close(self, inner_loop):
        if self.end_inner_round is not None and inner_loop.inner_round >= self.end_inner_round:
            return True
        return False

    def check_if_finish(self, outer_loop):
        if self.end_outer_round is not None and outer_loop.outer_round >= self.end_outer_round:
            return True
        return False

    @property
    def is_waiting(self):
        return self.state == Thread.WAITING

    @property
    def is_running(self):
        return self.state == Thread.RUNNING

    @property
    def is_closed(self):
        return self.state == Thread.CLOSED

    @property
    def is_finished(self):
        return self.state == Thread.FINISHED

    def wait(self):
        self.state = Thread.WAITING

    def run(self):
        self.state = Thread.RUNNING

    def close(self):
        self.state = Thread.CLOSED

    def finish(self):
        self.state = Thread.FINISHED


class InnerLoop:
    def __init__(self,
                 max_outer_rounds=None,
                 threads=[],
                 max_inner_rounds=None):
        self.max_inner_rounds = max_inner_rounds
        self.terminate_inner_loop = False
        self.will_terminate_inner_loop = False
        self.inner_round = 0
        self.threads = [Thread(max_outer_rounds=max_outer_rounds, max_inner_rounds=max_inner_rounds, **thread_cfg)
                        for thread_cfg in threads]

    def reset(self):
        self.terminate_inner_loop = False
        self.will_terminate_inner_loop = False
        self.inner_round = 0

    def terminate(self):
        self.terminate_inner_loop = True

    def will_terminate(self):
        self.will_terminate_inner_loop = True

    @property
    def is_terminated(self):
        return self.terminate_inner_loop

    @property
    def will_be_terminated(self):
        return self.will_terminate_inner_loop

    @property
    def max_rounds_reached(self):
        return self.max_inner_rounds is not None and self.inner_round >= self.max_inner_rounds

    def check_if_terminate(self, thread):
        if thread.terminate_inner_loop_when_closed and (thread.is_closed or thread.is_finished):
            return True
        return False

    def check_if_will_terminate(self, thread):
        if thread.will_terminate_inner_loop_when_closed and (thread.is_closed or thread.is_finished):
            return True
        return False


class Station:
    WAITING = 'waiting'
    RUNNING = 'running'
    CLOSED = 'closed'

    def __init__(self,
                 max_outer_rounds=None,
                 threads=[],
                 max_inner_rounds=None,
                 start_outer_round=0,
                 end_outer_round=None,
                 skip_outer_rounds=[],
                 outer_round_interval=1,
                 terminate_outer_loop_when_closed=False,
                 will_terminate_outer_loop_when_closed=False,
                 disable=False,
                 name=None,
                 **kwargs):
        self.start_outer_round = start_outer_round
        self.end_outer_round = max_outer_rounds if end_outer_round is None else end_outer_round
        self.skip_outer_rounds = skip_outer_rounds
        self.outer_round_interval = outer_round_interval
        self.terminate_outer_loop_when_closed = terminate_outer_loop_when_closed
        self.will_terminate_outer_loop_when_closed = will_terminate_outer_loop_when_closed
        self.disable = disable
        self.name = name
        self.inner_loop = InnerLoop(max_outer_rounds=max_outer_rounds, threads=threads, max_inner_rounds=max_inner_rounds)
        self.state = Station.WAITING
        self.ctx = Context(name=name, **kwargs)

    def check_if_run(self, outer_loop):
        if self.disable:
            return False

        outer_round = outer_loop.outer_round
        if outer_round < self.start_outer_round:
            return False
        if self.end_outer_round is not None and outer_round >= self.end_outer_round:
            return False
        if outer_round in self.skip_outer_rounds:
            return False
        if (outer_round - self.start_outer_round) % self.outer_round_interval != 0:
            return False
        return True

    def check_if_close(self, outer_loop=None, thread=None):
        assert outer_loop is not None or thread is not None
        if outer_loop is not None:
            # called at a station
            if self.end_outer_round is not None and outer_loop.outer_round >= self.end_outer_round:
                return True
            return False
        else:
            # called in a thread
            if thread.close_station_when_finished and thread.is_finished:
                return True
            return False

    def get_threads(self):
        return self.inner_loop.threads

    @property
    def is_waiting(self):
        return self.state == Station.WAITING

    @property
    def is_running(self):
        return self.state == Station.RUNNING

    @property
    def is_closed(self):
        return self.state == Station.CLOSED

    def wait(self):
        self.state = Station.WAITING

    def run(self):
        self.state = Station.RUNNING

    def close(self):
        self.state = Station.CLOSED


class OuterLoop:
    def __init__(self, stations=[], max_outer_rounds=None):
        self.max_outer_rounds = max_outer_rounds
        self.terminate_outer_loop = False
        self.will_terminate_outer_loop = False
        self.outer_round = 0
        self.stations = [Station(max_outer_rounds=max_outer_rounds, **station_cfg)
                         for station_cfg in stations]

    def terminate(self):
        self.terminate_outer_loop = True

    def will_terminate(self):
        self.will_terminate_outer_loop = True

    @property
    def is_terminated(self):
        return self.terminate_outer_loop

    @property
    def will_be_terminated(self):
        return self.will_terminate_outer_loop

    @property
    def max_rounds_reached(self):
        return self.max_outer_rounds is not None and self.outer_round >= self.max_outer_rounds

    def check_if_terminate(self, station=None, thread=None):
        assert station is not None or thread is not None
        if station is not None:
            # called at a station
            if station.terminate_outer_loop_when_closed and station.is_closed:
                return True
            return False
        else:
            # called in a thread
            if thread.terminate_outer_loop_when_finished and thread.is_finished:
                return True
            return False

    def check_if_will_terminate(self, station=None, thread=None):
        assert station is not None or thread is not None
        if station is not None:
            # called at a station
            if station.will_terminate_outer_loop_when_closed and station.is_closed:
                return True
            return False
        else:
            # called in a thread
            if thread.will_terminate_outer_loop_when_finished and thread.is_finished:
                return True
            return False


@RUN_SCHEDULERS.register_class('station_thread_loop_scheduler')
class StationThreadLoopScheduler:
    def __init__(self, **outer_loop_cfg):
        self.outer_loop = OuterLoop(**outer_loop_cfg)

    def get_stations(self):
        return self.outer_loop.stations

    def get_threads(self):
        return [thread for station in self.outer_loop.stations for thread in station.get_threads()]


@RUNS.register_class('station_thread_loop')
class StationThreadLoop(BaseRun):
    def __init__(self, name='station_thread_loop', stage_names=None):
        super().__init__(name, stage_names)

    def __call__(self, ctx):
        self.call_stage('run_begin', ctx)  # custom stage

        if isinstance(ctx, DummyContext):
            self.call_stage('enter_station', ctx)
            self.call_stage('enter_thread', ctx)
            self.call_stage('execute_thread', ctx)
            self.call_stage('exit_thread', ctx)
            self.call_stage('exit_station', ctx)
        else:
            assert ctx.has('run_scheduler') and isinstance(ctx.run_scheduler, StationThreadLoopScheduler), \
                "You must create a station-thread-loop scheduler in order to use station-thread loop"

            logger.info("Start station-thread-loop run")
            self._run_outer_loop(ctx.run_scheduler.outer_loop, ctx, prefix='-')
            logger.info("End station-thread-loop run")

        self.call_stage('run_end', ctx)  # custom stage

    def _run_outer_loop(self, outer_loop: OuterLoop, ctx, prefix=''):
        ctx.outer_loop = outer_loop

        while not outer_loop.is_terminated:
            logger.info(f"{prefix}Start outer loop round = {outer_loop.outer_round}")

            num_unclosed_stations = 0
            for station in outer_loop.stations:
                if station.is_closed:
                    continue

                num_unclosed_stations += 1
                self._process_station(station, outer_loop, ctx, prefix=f'{prefix}-')
                if outer_loop.is_terminated:
                    break

            logger.info(f"{prefix}End outer loop round = {outer_loop.outer_round}: {num_unclosed_stations} unclosed stations")

            if num_unclosed_stations > 0:
                outer_loop.outer_round += 1
            else:
                outer_loop.terminate()
                logger.info(f"{prefix}Outer loop terminated due to num_unclosed_stations = 0")

            if outer_loop.max_rounds_reached:
                outer_loop.terminate()
                logger.info(f"{prefix}Outer loop terminated due to max_rounds_reached is True")

            if outer_loop.will_be_terminated:
                outer_loop.terminate()
                logger.info(f"{prefix}Outer loop terminated due to will_be_terminated is True")

        del ctx.outer_loop

    def _process_station(self, station: Station, outer_loop: OuterLoop, ctx, prefix=''):
        ctx.station = station

        # pre-station-enter check
        assert station.state not in (Station.RUNNING, Station.CLOSED)
        if station.check_if_run(outer_loop):
            station.run()

        if station.is_running:
            logger.info(f"{prefix}Enter station {station.name}")
            self.call_stage('enter_station', ctx)  # custom stage

            self._run_inner_loop(station.inner_loop, station, outer_loop, ctx, prefix=f'{prefix}-')

            logger.info(f"{prefix}Exit station {station.name}")
            self.call_stage('exit_station', ctx)  # custom stage

        # post-station-exit check
        if station.is_running:
            station.wait()
        if not station.is_closed and station.check_if_close(outer_loop=outer_loop):
            station.close()
            logger.info(f"{prefix}Station {station.name} closed")
        if not outer_loop.will_be_terminated and outer_loop.check_if_will_terminate(station=station):
            outer_loop.will_terminate()
            logger.info(f"{prefix}Outer loop will be terminated at station {station.name}")
        if not outer_loop.is_terminated and outer_loop.check_if_terminate(station=station):
            logger.info(f"{prefix}Outer loop terminated at station {station.name}")
            outer_loop.terminate()

        del ctx.station

    def _run_inner_loop(self, inner_loop: InnerLoop, station: Station, outer_loop: OuterLoop, ctx, prefix=''):
        ctx.inner_loop = inner_loop

        # pre-inner-loop check
        inner_loop.reset()
        num_unclosed_threads = 0
        num_unfinished_threads = 0
        for thread in inner_loop.threads:
            assert not thread.is_running
            if thread.is_finished:
                continue
            num_unfinished_threads += 1
            if thread.check_if_run(outer_loop=outer_loop):
                thread.wait()
                num_unclosed_threads += 1
            else:
                thread.close()
                logger.info(f"{prefix}Thead {thread.name} closed before inner loop")

        if num_unfinished_threads == 0:
            station.close()
            logger.info(f"{prefix}Station {station.name} closed due to num_unfinished_threads = 0")
        if num_unclosed_threads == 0:
            inner_loop.terminate()
            logger.info(f"{prefix}Inner loop terminated due to num_unclosed_threads = 0")

        # inner loop
        while not inner_loop.is_terminated:
            num_unclosed_threads = 0
            for thread in inner_loop.threads:
                if thread.state in (Thread.CLOSED, Thread.FINISHED):
                    continue

                num_unclosed_threads += 1
                self._process_thread(thread, inner_loop, station, outer_loop, ctx, prefix=f'{prefix}-')
                if inner_loop.is_terminated:
                    break

            if num_unclosed_threads > 0:
                inner_loop.inner_round += 1
            else:
                inner_loop.terminate()
                logger.info(f"{prefix}Inner loop terminated due to num_unclosed_threads = 0")

            if inner_loop.max_rounds_reached:
                inner_loop.terminate()
                logger.info(f"{prefix}Inner loop terminated due to max_rounds_reached is True")

            if inner_loop.will_be_terminated:
                inner_loop.terminate()
                logger.info(f"{prefix}Inner loop terminated due to will_be_terminated is True")

        del ctx.inner_loop

    def _process_thread(self, thread: Thread, inner_loop: InnerLoop, station: Station, outer_loop: OuterLoop, ctx, prefix=''):
        ctx.thread = thread

        # pre-thread-enter check
        assert thread.state not in (Thread.RUNNING, Thread.CLOSED, Thread.FINISHED)
        if thread.check_if_run(inner_loop=inner_loop):
            thread.run()

        if thread.is_running:
            while True:
                thread.reenter = False

                self.call_stage('enter_thread', ctx)  # custom stage

                self.call_stage('execute_thread', ctx)  # custom stage

                self.call_stage('exit_thread', ctx)  # custom stage

                if not thread.reenter:
                    break

        # post-thread-exit check
        if thread.is_running:
            thread.wait()
        if not thread.is_finished:
            if thread.check_if_finish(outer_loop):
                thread.finish()
                logger.info(f"{prefix}Thead {thread.name} finished")
            elif not thread.is_closed and thread.check_if_close(inner_loop):
                thread.close()
                logger.info(f"{prefix}Thead {thread.name} closed")

        if not inner_loop.will_be_terminated and inner_loop.check_if_will_terminate(thread):
            inner_loop.will_terminate()
            logger.info(f"{prefix}Inner loop will be terminated at thread {thread.name}")
        if not outer_loop.will_be_terminated and outer_loop.check_if_will_terminate(thread=thread):
            outer_loop.will_terminate()
            logger.info(f"{prefix}Outer loop will be terminated at thread {thread.name}")
        if not inner_loop.is_terminated and inner_loop.check_if_terminate(thread):
            inner_loop.terminate()
            logger.info(f"{prefix}Inner loop terminated at thread {thread.name}")
        if outer_loop.is_terminated:
            inner_loop.terminate()
            logger.info(f"{prefix}Inner loop terminated at thread {thread.name} due to outer loop termined")
        elif outer_loop.check_if_terminate(thread=thread):
            outer_loop.terminate()
            inner_loop.terminate()
            logger.info(f"{prefix}Both inner and outer loop terminated at thread {thread.name}")
        if not station.is_closed and station.check_if_close(thread=thread):
            station.close()
            logger.info(f"{prefix}Station {station.name} closed at thread {thread.name}")

        del ctx.thread
