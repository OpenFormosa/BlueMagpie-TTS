# Apple Silicon MLX 加速


在 Apple Silicon（M 系列）上，可改用 **MLX** 原生路徑，直接在 Apple GPU（Metal、統一記憶體）上推論，通常比 PyTorch 的 MPS 後端更快。這是**選用**功能，核心仍維持 torch-only：

```bash
pip install -e .[mlx]
```

```python
from bluemagpie import BlueMagpieModel
from bluemagpie.mlx import BlueMagpieMLX, mlx_generate

model = BlueMagpieModel.from_local(model_dir, tokenizer=tokenizer, device="cpu")
mlx_model = BlueMagpieMLX(model)          # 轉換權重（只需一次）

import soundfile as sf
audio = mlx_generate(model, mlx_model, "今天天氣真好。", seed=0)   # 48 kHz 波形
sf.write("output.wav", audio.numpy(), model.sample_rate)
```

- 整條推論路徑（Barbet、RALM、LocEnc、LocDiT／CFM、**AudioVAE 解碼器**、AR 迴圈）皆以 MLX 重寫，並逐模組對 PyTorch 做數值 parity 驗證，生成過程可完全不經 PyTorch（僅斷詞與參考音檔編碼仍用 torch）。
- decode 採用快取式單步（cached step），逐步推進、不重算整個序列。
- `mlx_generate` 支援與 `generate` 相同的四種輸入模式（一般合成、語音接續、參考音檔、語者向量）。
- 在**真實 7.75GB 模型**上端對端 **RTF 0.77**（比即時還快），比 torch-MPS 快約 **1.45×**、比 torch-CPU 快約 **3.27×**（fp32，`scripts/bench_rtf.py`）。設計與限制詳見 [`src/bluemagpie/mlx/DESIGN.md`](../src/bluemagpie/mlx/DESIGN.md)。

[回到 README](../README.md)
