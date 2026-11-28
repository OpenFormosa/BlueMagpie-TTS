# 批次推論引擎


若你要同時處理多筆合成請求、追求更高吞吐量，可改用內建的批次推論引擎 `BlueMagpieEngine`。它採用**連續批次**（continuous batching）：多筆請求會一起批次解碼，新請求能在解碼途中加入，彼此互不影響。

引擎的特點：

- **不需額外相依套件**：只用到 `torch`，不必安裝 vLLM、flash-attn 等套件。
- **跨裝置**：CUDA、Apple Silicon（MPS）、CPU 共用同一套程式碼；CUDA 專屬的最佳化會自動偵測並啟用，其餘裝置自動略過。
- **與單筆 `generate` 數值一致**：在 batch=1 時，輸出與 `model.generate` 逐值相同（`model.generate` 始終是對照基準）。

## 基本用法

```python
import soundfile as sf
from bluemagpie.serving import BlueMagpieEngine, EngineConfig, Request

# model 與 tokenizer 的載入方式同前（from_local）
engine = BlueMagpieEngine(model, EngineConfig(max_num_seqs=16))

engine.add_request(Request(target_text="今天天氣真好。", seed=0))
engine.add_request(Request(target_text="第二句話。", reference_wav_path="speaker.wav"))

for out in engine.run():            # 依請求加入順序回傳
    # out.audio：48 kHz 波形（已掛載 AudioVAE 時）；out.latents：潛在表徵 [T, p, d]
    sf.write(f"output_{out.request_id}.wav", out.audio.numpy(), out.sample_rate)
```

`Request` 支援與 `generate` 相同的四種輸入模式（一般合成、語音接續、參考音檔、語者向量），欄位對應 `target_text`、`prompt_text`、`prompt_wav_path`、`reference_wav_path`、`speaker_centroid`、`cfg_value`、`inference_timesteps` 等。每筆請求可給定 `seed`，使該請求的輸出與同批其他請求的數量、加入順序無關。

## 串流輸出

`engine.stream()` 是一個產生器，會在每一步逐筆回傳各請求的區塊：

```python
for chunk in engine.stream():
    # chunk.request_id、chunk.latents、chunk.audio、chunk.finished
    play_or_write(chunk)
```

> 一般合成、參考音檔、語者向量這三種模式會回傳串流音訊（`chunk.audio`）；語音接續模式目前只回傳 `latents`，需要音訊時請改用 `run()`。

## 設定

`EngineConfig` 常用參數：

| 參數 | 預設值 | 說明 |
|---|---|---|
| `max_num_seqs` | `16` | 同時批次處理的最大請求數 |
| `max_model_len` | `2048` | 每筆序列的最大長度（提示＋生成） |
| `inference_timesteps` | `10` | 取樣步數 |
| `cfg_value` | `2.0` | 引導強度 |
| `enforce_eager` | `True` | 維持與單筆 `generate` 數值一致的路徑 |
| `compile` | `False` | 啟用 `torch.compile`（僅 CUDA 有效，其餘裝置自動略過） |

> 引擎的設計、取捨與已知限制詳見 [`src/bluemagpie/serving/DESIGN.md`](../src/bluemagpie/serving/DESIGN.md)。

## 加速原理：為什麼不是直接套 vLLM？

很多人期待「用 vLLM 之類的框架就能加速」，但對 BlueMagpie 來說，直接套 vLLM 並不可行，原因有二：

1. **真正的運算瓶頸不在語言模型，而在擴散式解碼器。** 每生成一個音訊單元，DiT（擴散式解碼器 LocDiT／CFM）要被呼叫「取樣步數 × 無條件／有條件兩路」次，以預設 10 步為例就是約 20 次，而語言模型（Barbet、RALM）各只跑一次。vLLM 是**文字語言模型**的推論框架，它根本不處理擴散式解碼器，就算把語言模型搬到 vLLM，主要運算量仍是 eager 執行，端對端不會明顯變快。
2. **vLLM 不支援 Barbet 的混合架構。** BlueMagpie 的語意語言模型 Barbet 是 Mamba2 與注意力的混合模型，vLLM（以及 nano-vllm、vllm-omni）對這種混合 TSLM 是零支援，得自行實作一個 first-class 混合模型才跑得起來，工程量大且僅限 CUDA。

因此本引擎改採**借用 vLLM 的架構技術、但不依賴它的 CUDA 套件**的做法：

- **連續批次**處理多請求（吞吐量的主要來源），跨請求共用批次運算。
- 以 **padded KV cache + SDPA + 遮罩**取代 vLLM 的 PagedAttention／FlashAttention，換取跨裝置、零依賴（代價是單一運算略慢、記憶體較不精省）。
- Barbet 的 Mamba 狀態以**純 PyTorch 單步遞迴**處理，不需融合 kernel。
- 可選的 `compile=True` 透過 `torch.compile`（內部即 CUDA graphs）加速 **DiT 與 LocEnc**，也就是真正的熱點，而這正是直接套 vLLM 不會幫你做的部分。

> 一句話總結：我們不追求單一運算比 vLLM 快，而是用 vLLM 級的**批次調度**搭配**針對 DiT 瓶頸的最佳化**，在零額外依賴、跨裝置的前提下提升整體吞吐量。

[回到 README](../README.md)
