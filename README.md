# vector-quantize-tinygrad

A small [tinygrad](https://github.com/tinygrad/tinygrad) port of the LFQ module from [lucidrains/vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch).

Only LFQ is ported so far. The broader `vector-quantize-pytorch` API, including the other quantizers, is not implemented here yet.

## Install

```bash
pip install git+https://github.com/sign/vector-quantize-tinygrad.git
```

## Basic Usage

```python
from tinygrad import Tensor
from vector_quantize_tinygrad import LFQ

lfq = LFQ(dim=10, codebook_size=1024)
x = Tensor.randn(2, 128, 10)

ret = lfq(x)
quantized, indices, entropy_aux_loss = ret
```

For training:

```python
from tinygrad import Tensor

with Tensor.train():
    ret = lfq(x, inv_temperature=5.0)
    loss = ret.quantized.square().mean() + ret.entropy_aux_loss
```

## Tests

The compatibility tests compare the tinygrad LFQ implementation against the PyTorch LFQ from `vector-quantize-pytorch`.

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install vector-quantize-pytorch
python -m pip install git+https://github.com/tinygrad/tinygrad.git
python -m pip install -e .
python tests/test_lfq.py
```

See [tests/test_lfq.py](tests/test_lfq.py) for the exact correctness checks and benchmark configurations.

## Current Performance

Forward/backward CUDA timings from the benchmark in [tests/test_lfq.py](tests/test_lfq.py) on a DGX Spark:

| config | torch fwd/bwd | tinygrad fwd/bwd |
| --- | ---: | ---: |
| single_codebook_1024 | 0.942s / 4.123s | 0.601s / 1.143s |
| two_codebooks_512 | 0.956s / 4.141s | 0.566s / 1.063s |
| projected_2048 | 0.948s / 2.662s | 0.620s / 1.325s |

With `BEAM=2`:

| config | torch fwd/bwd | tinygrad fwd/bwd |
| --- | ---: | ---: |
| single_codebook_1024 | 0.946s / 4.132s | 0.408s / 0.682s |
| two_codebooks_512 | 0.959s / 4.142s | 0.555s / 0.787s |
| projected_2048 | 0.943s / 2.667s | 0.475s / 0.747s |

## CI

GitHub Actions runs the LFQ compatibility test on every push and pull request. The workflow installs CPU PyTorch, `vector-quantize-pytorch`, tinygrad from source, this package, and then runs `python tests/test_lfq.py`. Configure GitHub branch protection to require the workflow before merging pull requests.

## License

MIT. Contributions are welcome.
