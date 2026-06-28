# BlueMagpie-TTS

<p align="center">
  <img src="assets/icon.png" alt="OpenFormosa Blue Magpie TTS" width="260">
</p>

BlueMagpie-TTS是一套文字轉語音（TTS）模型，能把文字合成為自然的語音。它支援三種使用情境：

- **一般語音合成** —— 直接把文字唸出來。
- **聲音複製** —— 提供一段參考音檔，讓輸出模仿該語者的音色。
- **指定語者** —— 用事先準備好的語者向量來控制音色。

同時也提供**串流輸出**，適合需要邊合成邊播放的應用。

🔊 **線上試玩**：[BlueMagpie-TTS Demo（Hugging Face Space）](https://huggingface.co/spaces/voidful/BlueMagpie-TTS-Demo)

## 命名由來

專案全名是 **OpenFormosa Blue Magpie TTS**。「藍鵲」取自**台灣藍鵲**（Taiwan Blue Magpie，學名 *Urocissa caerulea*）。選牠作為 TTS 的識別有幾層用意：
- **會發聲、辨識度高**：台灣藍鵲本來就是叫聲響亮、容易辨認的鳥，恰好呼應 TTS「把文字變成聲音」的核心。
- **長尾的動態感**：標誌性的長尾巴帶來流動、延展的視覺意象，比一般的喇叭 icon 更有記憶點與品牌個性。
- **立足台灣**：OpenFormosa（福爾摩沙）點出專案面向台灣華語、在地化語音的定位。

## 安裝

先把專案 clone 下來，再以可編輯模式安裝：

```bash
git clone https://github.com/OpenFormosa/BlueMagpie-TTS
cd BlueMagpie-TTS
pip install -e .
```

安裝過程會自動從 GitHub 取得相依的 [`barbet`](https://github.com/OpenFormosa/Barbet) 套件（負責文字語意的語言模型）。語音合成所需的聲學模組已內含於專案中（位於 `bluemagpie/_vendor/`，原始碼取自 [VoxCPM](https://github.com/OpenBMB/VoxCPM)，採 Apache-2.0 授權），不需另外安裝。

若要儲存合成出來的音檔，建議另外安裝 `soundfile`：

```bash
pip install soundfile
```

## 載入模型

### 從 Hugging Face 下載

```python
import os
from huggingface_hub import snapshot_download
from transformers import PreTrainedTokenizerFast
from bluemagpie import BlueMagpieModel

model_dir = snapshot_download("OpenFormosa/BlueMagpie-TTS", token=True)
# 直接從 tokenizer.json 載入 tokenizer，相容較新版 transformers（5.x）
tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_dir, "tokenizer.json"))
model = BlueMagpieModel.from_local(model_dir, tokenizer=tokenizer, training=False, device="cuda")
```

### 從本機目錄載入

如果你已經有一份模型檔案，直接指向該目錄即可：

```python
import os
from transformers import PreTrainedTokenizerFast
from bluemagpie import BlueMagpieModel

model_dir = "checkpoints/bluemagpie"
tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_dir, "tokenizer.json"))
model = BlueMagpieModel.from_local(model_dir, tokenizer=tokenizer, training=False, device="cuda")
```

- `device` 可填 `"cuda"` 或 `"cpu"`，不指定時會自動選擇。
- 推論時請固定使用 `training=False`。

## 基本使用：文字轉語音

`generate` 會回傳一段語音波形（`torch.Tensor`），搭配 `soundfile` 即可存成 `.wav`。輸出的取樣率可由 `model.sample_rate` 取得：

```python
import soundfile as sf

audio = model.generate(
    target_text="今天天氣真好。",
    cfg_value=2.0,
)

sf.write("output.wav", audio.squeeze().cpu().numpy(), model.sample_rate)
```

## 聲音複製：模仿參考語者的音色

有兩種做法。

**A. 語者向量（`speaker_centroid`）** —— 從參考音檔抽出語者向量再合成（免逐字稿）：

```bash
pip install -e ".[clone]"   # 抽取需要 speechbrain（ECAPA-TDNN）
python scripts/extract_speaker_centroid.py --audio reference.wav --out my_voice.pt
# 多段同一語者更穩定：--audio a.wav b.wav c.wav
```

```python
import torch

centroid = torch.load("my_voice.pt", weights_only=True)   # [192] 語者向量
audio = model.generate(
    target_text="今天天氣真好。",
    speaker_centroid=centroid,
    cfg_value=2.8,
)

# 也可以在程式中直接抽取（不寫檔）：
from bluemagpie import extract_speaker_centroid
centroid = extract_speaker_centroid("reference.wav")      # [192]
```

**B. 參考音檔（`reference_wav_path`）** —— 直接提供一段參考音檔：

```python
audio = model.generate(
    target_text="今天天氣真好。",
    reference_wav_path="reference.wav",
    cfg_value=2.8,
)
```

## 指定語者：以語者向量控制音色

模型自帶一個**多語者向量表** `checkpoints/speaker_centroids.pt`，目前內含兩個語者：

| 語者 ID | 說明 | 建議 `cfg_value` |
| --- | --- | --- |
| `hung_yi_lee` | 李宏毅（Hung-yi Lee）老師的語者向量（已取得本人授權；官方最佳參數即針對此語者調校） | 2.0–2.8 |
| `female_voice` | 一個通用女聲語者向量 | 2.0–2.8 |

向量表的格式為 `{"speaker_ids": [...], "centroids": tensor[N, 192], "dim": 192}`。以 `torch.load` 載入後，**依語者 ID 取出該語者的 `[192]` 向量**，再透過 `speaker_centroid` 指定音色：

```python
import os
import torch

table = torch.load(
    os.path.join(model_dir, "checkpoints", "speaker_centroids.pt"),
    map_location="cpu",
    weights_only=True,
)
print(table["speaker_ids"])          # ['hung_yi_lee', 'female_voice']

# 切換語者只要改這一行（"hung_yi_lee" 或 "female_voice"）
speaker_id = "female_voice"
speaker_centroid = table["centroids"][table["speaker_ids"].index(speaker_id)]   # [192]

audio = model.generate(
    target_text="今天天氣真好。",
    speaker_centroid=speaker_centroid,   # 也可以直接傳入你自己已取得授權的語者向量
    cfg_value=2.0,
)
```

只用模型 ID（尚未先 `snapshot_download` 整個模型）時，可單獨抓向量表這一個檔：

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download("OpenFormosa/BlueMagpie-TTS", "checkpoints/speaker_centroids.pt")
table = torch.load(path, map_location="cpu", weights_only=True)
```

> 想新增更多語者，用上方〈聲音複製〉的 `extract_speaker_centroid` 抽出你自己（已取得授權）的 `[192]` 向量即可，傳法完全相同。早期僅含李宏毅單一語者的 `checkpoints/hung_yi_lee_speaker_centroids.pt`（格式相同）仍保留可用。

## 串流輸出

需要邊合成邊播放時，改用 `generate_streaming`。它是一個產生器，會一段一段地回傳音訊區塊：

```python
chunks = []
for chunk in model.generate_streaming(target_text="今天天氣真好。"):
    chunks.append(chunk)
    # 這裡可以即時播放或寫出 chunk
```

> 注意：串流模式下不支援自動重試（`retry_badcase`）。

## 四種輸入模式

模型支援四種輸入組合，皆透過同一個 `generate` 介面切換：

| 模式 | 需要的參數 | 用途 |
|---|---|---|
| 一般合成 | `target_text` | 直接把文字唸出來 |
| 語音接續 | `target_text`、`prompt_text`、`prompt_wav_path` | 從一段已有的語音與其文字接著往下唸 |
| 參考音檔 | `target_text`、`reference_wav_path` | 模仿參考音檔的語者音色 |
| 語者向量 | `target_text`、`speaker_centroid` | 以語者向量複製音色 |

## `generate` 常用參數

| 參數 | 預設值 | 說明 |
|---|---|---|
| `target_text` | （必填） | 要合成的文字 |
| `prompt_text` | `""` | 提示文字，搭配 `prompt_wav_path` 做語音接續 |
| `prompt_wav_path` | `""` | 提示音檔路徑，用於語音接續 |
| `reference_wav_path` | `""` | 參考音檔路徑，用於聲音複製 |
| `speaker_centroid` | `None` | 語者向量，用於指定音色 |
| `cfg_value` | `2.0` | 引導強度，數值越大越貼合條件、但可能較不自然 |
| `inference_timesteps` | `10` | 取樣步數，越多通常品質越好、速度越慢 |
| `min_len` / `max_len` | `2` / `2000` | 輸出長度的下限與上限 |
| `retry_badcase` | `False` | 偵測到異常輸出時自動重試（串流模式不支援） |

## 批次推論引擎（多請求加速）

若你要同時處理多筆合成請求、追求更高吞吐量，可改用內建的批次推論引擎 `BlueMagpieEngine`。它採用**連續批次**（continuous batching）：多筆請求會一起批次解碼，新請求能在解碼途中加入，彼此互不影響。

引擎的特點：

- **不需額外相依套件**：只用到 `torch`，不必安裝 vLLM、flash-attn 等套件。
- **跨裝置**：CUDA、Apple Silicon（MPS）、CPU 共用同一套程式碼；CUDA 專屬的最佳化會自動偵測並啟用，其餘裝置自動略過。
- **與單筆 `generate` 數值一致**：在 batch=1 時，輸出與 `model.generate` 逐值相同（`model.generate` 始終是對照基準）。

### 基本用法

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

### 串流輸出

`engine.stream()` 是一個產生器，會在每一步逐筆回傳各請求的區塊：

```python
for chunk in engine.stream():
    # chunk.request_id、chunk.latents、chunk.audio、chunk.finished
    play_or_write(chunk)
```

> 一般合成、參考音檔、語者向量這三種模式會回傳串流音訊（`chunk.audio`）；語音接續模式目前只回傳 `latents`，需要音訊時請改用 `run()`。

### 設定

`EngineConfig` 常用參數：

| 參數 | 預設值 | 說明 |
|---|---|---|
| `max_num_seqs` | `16` | 同時批次處理的最大請求數 |
| `max_model_len` | `2048` | 每筆序列的最大長度（提示＋生成） |
| `inference_timesteps` | `9` | 取樣步數 |
| `cfg_value` | `2.8` | 引導強度 |
| `enforce_eager` | `True` | 維持與單筆 `generate` 數值一致的路徑 |
| `compile` | `False` | 啟用 `torch.compile`（僅 CUDA 有效，其餘裝置自動略過） |

> 引擎的設計、取捨與已知限制詳見 [`src/bluemagpie/serving/DESIGN.md`](src/bluemagpie/serving/DESIGN.md)。

### 加速原理：為什麼不是直接套 vLLM？

很多人期待「用 vLLM 之類的框架就能加速」，但對 BlueMagpie 來說，直接套 vLLM 並不可行，原因有二：

1. **真正的運算瓶頸不在語言模型，而在擴散式解碼器。** 每生成一個音訊單元，DiT（擴散式解碼器 LocDiT／CFM）要被呼叫約 16–18 次（取樣步數 × 無條件／有條件兩路），而語言模型（Barbet、RALM）各只跑一次。vLLM 是**文字語言模型**的推論框架，它根本不處理擴散式解碼器——就算把語言模型搬到 vLLM，主要運算量仍是 eager 執行，端到端不會明顯變快。
2. **vLLM 不支援 Barbet 的混合架構。** BlueMagpie 的語意語言模型 Barbet 是 Mamba2 與注意力的混合模型，vLLM（以及 nano-vllm、vllm-omni）對這種混合 TSLM 是零支援，得自行實作一個 first-class 混合模型才跑得起來，工程量大且僅限 CUDA。

因此本引擎改採**借用 vLLM 的架構技術、但不依賴它的 CUDA 套件**的做法：

- **連續批次**處理多請求（吞吐量的主要來源），跨請求共用批次運算。
- 以 **padded KV cache + SDPA + 遮罩**取代 vLLM 的 PagedAttention／FlashAttention，換取跨裝置、零依賴（代價是單一運算略慢、記憶體較不精省）。
- Barbet 的 Mamba 狀態以**純 PyTorch 單步遞迴**處理，不需融合 kernel。
- 可選的 `compile=True` 透過 `torch.compile`（內部即 CUDA graphs）加速 **DiT 與 LocEnc**——也就是真正的熱點，而這正是直接套 vLLM 不會幫你做的部分。

> 一句話總結：我們不追求單一運算比 vLLM 快，而是用 vLLM 級的**批次調度**搭配**針對 DiT 瓶頸的最佳化**，在零額外依賴、跨裝置的前提下提升整體吞吐量。

## Apple Silicon MLX 加速（選用）

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

- 整條推論路徑（Barbet、RALM、LocEnc、LocDiT／CFM、**AudioVAE 解碼器**、AR 迴圈）皆以 MLX 重寫，並逐模組對 PyTorch 做數值 parity 驗證——生成過程可完全不經 PyTorch（僅斷詞與參考音檔編碼仍用 torch）。
- decode 採用快取式單步（cached step），逐步推進、不重算整個序列。
- `mlx_generate` 支援與 `generate` 相同的四種輸入模式（一般合成、語音接續、參考音檔、語者向量）。
- 在**真實 7.75GB 模型**上端到端 **RTF 0.77**（比即時還快）——比 torch-MPS 快約 **1.45×**、比 torch-CPU 快約 **3.27×**（fp32，`scripts/bench_rtf.py`）。設計與限制詳見 [`src/bluemagpie/mlx/DESIGN.md`](src/bluemagpie/mlx/DESIGN.md)。

## 注意事項

- 上面範例都直接從 `tokenizer.json` 載入 tokenizer 再傳給 `from_local`，在較新版 transformers（5.x）也能穩定運作；背後原因見〈疑難排解〉。
- 沒有 GPU 也可以執行：把 `device` 設為 `"cpu"` 即可（速度較慢，但短句合成只需數十秒）。輸出為 48 kHz 單聲道。
- 模型內附的 `hung_yi_lee` 語者向量已取得本人授權，可直接作為範例使用；指定其他語者或進行聲音複製時，請只使用你已取得授權的參考音檔或語者向量。
- 請妥善保管語者向量表與合成出來的音檔，未經授權前不要對外散布。

## 疑難排解

**較新版 transformers（5.x）的 tokenizer 載入**

上面的範例都直接從 `tokenizer.json` 載入 tokenizer 再傳給 `from_local`，因此在 transformers 5.x 也能正常運作，不需額外處理（模型只用到 tokenizer 的 `encode`）。

若你改用 `from_local` 的自動載入（不傳入 `tokenizer`），在 transformers 5.x 可能會失敗 —— 解析 `tokenizer_config.json` 時出現

```
TypeError: ..._patch_mistral_regex() got multiple values for keyword argument 'fix_mistral_regex'
```

或載入看似成功、但呼叫 `generate()` 時才報 `ValueError: No tokenizer attached to BlueMagpieModel`。遇到時改回上面範例的明確載入方式即可。
