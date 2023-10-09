# Demo

To run our demo, you need to prepare *Dromedary-2* checkpoints locally. Please follow the [instructions here](https://github.com/IBM/SALMON#model-weights). This demo is adpted from [Guanaco demo notebook](https://colab.research.google.com/drive/17XEqL1JcmVWjHkT-WczdYkJlNINacwG7?usp=sharing). The code is tested on a single A100-80GB GPU, with a peak GPU memory usage < 48GB.

```bash
python -u demo.py \
    --model-name "/path/to/llama-2-70b-hf" \
    --adapters-name "/path/to/qlora_adapter_models"
```
