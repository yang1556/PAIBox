import numpy as np
import pytest

import paibox as pb


@pytest.mark.parametrize("Process", [pb.simulator.UniformGen, pb.simulator.Constant])
def test_Processes_run(Process: pb.base.Process):
    duration = 10

    # 1. Output shape is a scalar
    proc = Process(shape_out=100)
    output = proc.run(10)

    assert output.shape == (duration, 100)

    # 2. Output shape is 1-dimension
    proc = Process(shape_out=(100,))
    output = proc.run(10)

    assert output.shape == (duration, 100)

    # 3. Output shape is 2-dimension with keep_size = False
    proc = Process(shape_out=(10, 10), keep_size=False)
    output = proc.run(10)

    assert output.shape == (duration, 100)

    # 4. Output shape is 2-dimension with keep_size = True
    proc = Process(shape_out=(10, 10), keep_size=True)
    output = proc.run(10)

    assert output.shape == (duration, 10, 10)


def test_User_Process_run():
    duration = 10

    class Simple_User_Process(pb.base.Process):
        def __init__(self, shape_out, **kwargs):
            super().__init__(shape_out, **kwargs)

        def update(self, *args, **kwargs):
            return np.random.randint(2, size=self.varshape)

    # 1. Output shape is a scalar
    proc = Simple_User_Process(shape_out=100)
    output = proc.run(duration)

    assert output.shape == (duration, 100)

    # 2. Output shape is 1-dimension
    proc = Simple_User_Process(shape_out=(100,))
    output = proc.run(duration)

    assert output.shape == (duration, 100)

    # 3. Output shape is 2-dimension with keep_size = False
    proc = Simple_User_Process(shape_out=(10, 10))
    output = proc.run(duration)

    assert output.shape == (duration, 100)

    # 4. Output shape is 2-dimension with keep_size = True
    proc = Simple_User_Process(shape_out=(10, 10), keep_size=True)
    output = proc.run(duration)

    assert output.shape == (duration, 10, 10)

    class MyProcess1_with_timestep(pb.base.Process):
        def __init__(self, shape_out, **kwargs):
            super().__init__(shape_out, **kwargs)

        def update(self, t):
            return np.random.randint(2, size=self.varshape) * t

    # 1. Output shape is a scalar
    proc = MyProcess1_with_timestep(shape_out=100)
    output = proc.run(duration)

    assert output.shape == (duration, 100)

    # 2. Output shape is 1-dimension
    proc = MyProcess1_with_timestep(shape_out=(100,))
    output = proc.run(duration)

    assert output.shape == (duration, 100)

    # 3. Output shape is 2-dimension with keep_size = False
    proc = MyProcess1_with_timestep(shape_out=(10, 10))
    output = proc.run(duration)

    assert output.shape == (duration, 100)

    # 4. Output shape is 2-dimension with keep_size = True
    proc = MyProcess1_with_timestep(shape_out=(10, 10), keep_size=True)
    output = proc.run(duration)

    assert output.shape == (duration, 10, 10)


def test_User_Process_run_with_args():
    duration = 10

    class MyProcess1_with_args(pb.base.Process):
        def __init__(self, shape_out, **kwargs):
            super().__init__(shape_out, **kwargs)

        def update(self, t, bias, *args, **kwargs):
            if "Hello" in kwargs:
                bias += 1

            return np.random.randint(2, size=self.varshape) * t + bias

    # 1. Output shape is a scalar
    proc = MyProcess1_with_args(shape_out=100)
    output = proc.run(duration, bias=1, extra1=1, extra2="Hello")

    assert output.shape == (duration, 100)

    # 2. Output shape is 1-dimension
    proc = MyProcess1_with_args(shape_out=(100,))
    output = proc.run(duration, bias=2)

    assert output.shape == (duration, 100)

    # 3. Output shape is 2-dimension with keep_size = False
    proc = MyProcess1_with_args(shape_out=(10, 10))
    output = proc.run(duration, bias=0)

    assert output.shape == (duration, 100)

    # 4. Output shape is 2-dimension with keep_size = True
    proc = MyProcess1_with_args(shape_out=(10, 10), keep_size=True)
    output = proc.run(duration, bias=2)

    assert output.shape == (duration, 10, 10)
