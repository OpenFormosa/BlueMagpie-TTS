# BlueMagpie-TTS

<p align="center">
  <img src="assets/icon.png" alt="OpenFormosa Blue Magpie TTS" width="260">
</p>

**BlueMagpie-TTS** 是一套針對**台灣華語**與**中英混合（code-switching）**情境打造的開源文字轉語音（TTS）模型：像「這個 feature 明天上線」這種台灣日常語句，不必改寫就能自然唸出，輸出 48 kHz 高品質語音。在同一份內部測試集上，它的字元錯誤率（CER）為 7.4%，比參考基準系統的 11.4% **相對降低約 35%**——每合成 100 個字，唸錯的字約從 11 個降到 7 個（數字的意義見〈[效能數據](#效能數據)〉）。

- 🔊 **線上試玩**：[BlueMagpie-TTS Demo（Hugging Face Space）](https://huggingface.co/spaces/voidful/BlueMagpie-TTS-Demo)——不用安裝，開瀏覽器就能聽。
- 📦 **模型下載**：[OpenFormosa/BlueMagpie-TTS（Hugging Face）](https://huggingface.co/OpenFormosa/BlueMagpie-TTS)——公開模型，不需申請、不需 token。

## 它能做什麼

| 功能 | 說明 | 章節 |
| --- | --- | --- |
| 一般語音合成 | 直接把文字唸出來 | 〈[基本使用](#基本使用文字轉語音)〉 |
| 聲音複製（**免逐字稿**） | 給一段 3 秒以上的參考音檔，模仿該語者的音色；不需要參考音檔的文字稿 | 〈[聲音複製](#聲音複製模仿參考語者的音色)〉 |
| 指定語者 | 用內附（已授權）或自建的語者向量控制音色 | 〈[指定語者](#指定語者以語者向量控制音色)〉 |
| 語音接續 | 從一段既有語音與其文字接著唸下去 | 〈[四種輸入模式](#四種輸入模式)〉 |
| 串流輸出 | 邊合成邊播放 | 〈[串流輸出](#串流輸出)〉 |
| 批次推論引擎 | 多請求連續批次解碼，提升吞吐量（torch-only、跨裝置） | 〈[批次推論引擎](#批次推論引擎多請求加速)〉 |
| Apple Silicon 加速 | MLX 原生推論，M 系列晶片上比即時更快 | 〈[MLX 加速](#apple-silicon-mlx-加速選用)〉 |

## 支援環境

| 項目 | 需求 |
| --- | --- |
| Python | **3.10 – 3.12**（3.13 尚未支援） |
| 作業系統 | Linux、macOS（Windows 未經測試） |
| 主要相依套件 | PyTorch ≥ 2.1、transformers ≥ 4.44（5.x 亦可）、numpy ≥ 1.26 且 < 2.4 |
| 模型下載量 | 約 8 GB（首次執行自動下載） |
| Hugging Face 帳號 | **不需要**（模型公開、非 gated） |

| 硬體 | 支援 | 用法 | 速度參考* |
| --- | :-: | --- | --- |
| NVIDIA GPU（CUDA） | ✅ | `device="cuda"` | 最快 |
| Apple Silicon（MLX，選用） | ✅ | `pip install -e .[mlx]`，見〈[MLX 加速](#apple-silicon-mlx-加速選用)〉 | RTF ≈ 0.77（比即時快） |
| Apple Silicon（MPS） | ✅ | `device="mps"` | RTF ≈ 1.1 |
| CPU | ✅ | `device="cpu"`，見〈[在 CPU 上執行](#在-cpu-上執行)〉 | RTF ≈ 2.5 |

<sub>* RTF（real-time factor）＝合成 1 秒語音所需的運算秒數，小於 1 代表比即時快。數值為 `scripts/bench_rtf.py` 在 Apple M 系列、fp32 下實測，不同機器會有差異。</sub>

各介面的功能支援：

| 功能 | 單筆 `generate` | 串流 `generate_streaming` | 批次引擎 | MLX |
| --- | :-: | :-: | :-: | :-: |
| 一般合成 | ✅ | ✅ | ✅ | ✅ |
| 聲音複製（參考音檔） | ✅ | ✅ | ✅ | ✅ |
| 聲音複製／指定語者（語者向量） | ✅ | ✅ | ✅ | ✅ |
| 語音接續（prompt） | ✅ | ✅ | ✅（串流時僅回傳 latents） | ✅ |
| 自動重試（`retry_badcase`） | ✅ | — | — | — |

## 快速上手

**1. 安裝**（需 Python 3.10–3.12）：

```bash
git clone https://github.com/OpenFormosa/BlueMagpie-TTS
cd BlueMagpie-TTS
pip install -e .
pip install soundfile        # 存 .wav 檔用
```

安裝過程會自動從 GitHub 取得相依的 [`barbet`](https://github.com/OpenFormosa/Barbet) 套件（負責文字語意的語言模型）。語音合成所需的聲學模組已內含於專案中（位於 `bluemagpie/_vendor/`，原始碼取自 [VoxCPM](https://github.com/OpenBMB/VoxCPM)，採 Apache-2.0 授權），不需另外安裝。

**2. 合成第一句語音**（首次執行會自動下載模型，約 8 GB）：

```python
import os
import torch
import soundfile as sf
from huggingface_hub import snapshot_download
from transformers import PreTrainedTokenizerFast
from bluemagpie import BlueMagpieModel

model_dir = snapshot_download("OpenFormosa/BlueMagpie-TTS")

# 直接從 tokenizer.json 載入 tokenizer，相容較新版 transformers（5.x）
tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_dir, "tokenizer.json"))

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model = BlueMagpieModel.from_local(model_dir, tokenizer=tokenizer, training=False, device=device)

# 用內附（已取得授權）的語者向量，第一次就能聽到穩定的音色
table = torch.load(os.path.join(model_dir, "checkpoints", "speaker_centroids.pt"),
                   map_location="cpu", weights_only=True)
centroid = table["centroids"][table["speaker_ids"].index("hung_yi_lee")]

audio = model.generate(
    target_text="你好，這是 BlueMagpie 合成的第一句話。",
    speaker_centroid=centroid,
    cfg_value=2.0,
)
sf.write("hello.wav", audio.squeeze().cpu().numpy(), model.sample_rate)
print("已寫出 hello.wav")
```

**3. 打開 `hello.wav` 聽結果。** 遇到問題先看〈[疑難排解](#疑難排解)〉。

## 效能數據

以下數字來自模型的內部保留測試集，隨 checkpoint 發佈於 [`release_metadata.json`](https://huggingface.co/OpenFormosa/BlueMagpie-TTS/blob/main/release_metadata.json)。評測方式：把合成語音用 ASR 轉寫回文字，再與原文比對。

| 系統 | 條件 | CER ↓ | WER ↓ |
| --- | --- | ---: | ---: |
| **BlueMagpie-TTS** | 語者向量（centroid） | **7.44%** | **8.57%** |
| **BlueMagpie-TTS** | 參考音檔（免逐字稿） | 8.99% | 11.77% |
| 參考基準系統 | 同一測試集 | 11.45% | 14.83% |

**怎麼讀這些數字**：

- **CER（字元錯誤率）**＝合成語音轉寫回文字後，與原文不一致（唸錯、漏唸、多唸）的字元比例；**WER** 是詞層級的同一概念。越低越好。
- 以語者向量條件為例：CER 從基準的 11.45% 降到 7.44%，**相對降低 35%**——直觀來說，每合成 100 個字，唸錯的字從約 11 個降到 7 個。
- 對想導入的團隊而言：同樣的素材，需要人工重聽、重錄的比例大約可以少三分之一。

長文合成診斷（同一 checkpoint）：

| 指標 | 數值 | 說明 |
| --- | ---: | --- |
| 語者向量 CER／WER | 7.51% | 長文逐段合成後的整體評測 |
| 參考音檔 CER／WER | 8.92% | 同上 |
| 語速 | 約 4.0 字/秒 | 接近自然朗讀速度 |
| 跨段落語者相似度下降 | 0.109 | 參考音檔模式下，長文各段音色一致性的損失 |

> 這是內部模型選擇用的評測、並非公開基準；換用不同的 ASR 或測試集，絕對數字會不同，請以「同條件下的相對比較」為主。

## 載入模型

### 從 Hugging Face 下載

```python
import os
from huggingface_hub import snapshot_download
from transformers import PreTrainedTokenizerFast
from bluemagpie import BlueMagpieModel

model_dir = snapshot_download("OpenFormosa/BlueMagpie-TTS")
# 直接從 tokenizer.json 載入 tokenizer，相容較新版 transformers（5.x）
tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_dir, "tokenizer.json"))
model = BlueMagpieModel.from_local(model_dir, tokenizer=tokenizer, training=False, device="cuda")
```

模型是公開的，下載不需要 token 或申請權限。

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

- `device` 可填 `"cuda"`、`"mps"` 或 `"cpu"`，不指定時會自動選擇。
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

兩種做法，**都不需要**參考音檔的逐字稿：

**A. 參考音檔（`reference_wav_path`）** —— 最簡單：直接給一段 3 秒以上的乾淨語音：

```python
audio = model.generate(
    target_text="今天天氣真好。",
    reference_wav_path="reference.wav",
    cfg_value=2.0,
)
```

> 此路徑自 checkpoint `step_0006000`（2026-07 發佈）起正式支援（內部評測 CER 8.99%）。**更早的 checkpoint 不支援這條路徑，會產生內容錯誤的語音**；請重新執行 `snapshot_download` 取得最新模型。

**B. 語者向量（`speaker_centroid`）** —— 品質最穩定（內部評測 CER 7.44%）：先從參考音檔抽出語者向量再合成，多段參考音檔可平均出更穩的音色：

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
    cfg_value=2.0,
)

# 也可以在程式中直接抽取（不寫檔）：
from bluemagpie import extract_speaker_centroid
centroid = extract_speaker_centroid("reference.wav")      # [192]
```

**怎麼選**：只有一小段音檔、想快速試用 → 用 **A**；要長期使用、追求最穩定的音色一致性 → 用 **B**（多段參考平均）。

> ⚠️ 請只使用你已取得授權的聲音；請勿在未經本人同意下複製他人聲音。

## 指定語者：以語者向量控制音色

模型自帶一個**多語者向量表** `checkpoints/speaker_centroids.pt`，目前內含兩個語者：

| 語者 ID | 說明 | 建議 `cfg_value` |
| --- | --- | --- |
| `hung_yi_lee` | 李宏毅（Hung-yi Lee）老師的語者向量（已取得本人授權） | 2.0–2.8 |
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
| 參考音檔 | `target_text`、`reference_wav_path` | 模仿參考音檔的語者音色（免逐字稿） |
| 語者向量 | `target_text`、`speaker_centroid` | 以語者向量複製音色 |

## `generate` 常用參數

| 參數 | 預設值 | 說明 |
|---|---|---|
| `target_text` | （必填） | 要合成的文字 |
| `prompt_text` | `""` | 提示文字，搭配 `prompt_wav_path` 做語音接續 |
| `prompt_wav_path` | `""` | 提示音檔路徑，用於語音接續 |
| `reference_wav_path` | `""` | 參考音檔路徑，用於聲音複製 |
| `speaker_centroid` | `None` | 語者向量，用於指定音色 |
| `cfg_value` | `2.0` | 引導強度，數值越大越貼合條件、但可能較不自然（建議 2.0–2.8） |
| `inference_timesteps` | `10` | 取樣步數，越多通常品質越好、速度越慢 |
| `min_len` / `max_len` | `2` / `2000` | 輸出長度的下限與上限 |
| `retry_badcase` | `False` | 偵測到異常輸出時自動重試（串流模式不支援；離線批次合成建議開啟） |

## 在 CPU 上執行

沒有 GPU 也能跑：載入時把 `device` 設為 `"cpu"`，其餘程式碼完全相同：

```python
model = BlueMagpieModel.from_local(model_dir, tokenizer=tokenizer, training=False, device="cpu")
```

- **速度參考**：RTF 約 2.5——合成 1 秒語音約需 2.5 秒運算（Apple M 系列實測；x86 依核心數而異）。一句十多個字的話，約十幾秒可完成。
- 長文建議切句逐段合成（搭配〈串流輸出〉），第一句能更快出聲。
- Apple Silicon 使用者建議改用〈[MLX 加速](#apple-silicon-mlx-加速選用)〉，比 CPU 快約 3.3 倍、比即時更快。

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
| `inference_timesteps` | `10` | 取樣步數 |
| `cfg_value` | `2.0` | 引導強度 |
| `enforce_eager` | `True` | 維持與單筆 `generate` 數值一致的路徑 |
| `compile` | `False` | 啟用 `torch.compile`（僅 CUDA 有效，其餘裝置自動略過） |

> 引擎的設計、取捨與已知限制詳見 [`src/bluemagpie/serving/DESIGN.md`](src/bluemagpie/serving/DESIGN.md)。

### 加速原理：為什麼不是直接套 vLLM？

很多人期待「用 vLLM 之類的框架就能加速」，但對 BlueMagpie 來說，直接套 vLLM 並不可行，原因有二：

1. **真正的運算瓶頸不在語言模型，而在擴散式解碼器。** 每生成一個音訊單元，DiT（擴散式解碼器 LocDiT／CFM）要被呼叫「取樣步數 × 無條件／有條件兩路」次——以預設 10 步為例就是約 20 次，而語言模型（Barbet、RALM）各只跑一次。vLLM 是**文字語言模型**的推論框架，它根本不處理擴散式解碼器——就算把語言模型搬到 vLLM，主要運算量仍是 eager 執行，端到端不會明顯變快。
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

## 應用場景與合作

BlueMagpie-TTS 適合需要「台灣的聲音」的場景：

- **有聲內容**：新聞、部落格、電子報轉成有聲版；長文可批次或串流合成。
- **客服與語音介面**：IVR 提示音、對話系統回覆；中英夾雜的專有名詞（產品名、英文縮寫）不必改寫就能唸。
- **無障礙應用**：以自然的台灣華語朗讀網頁與文件。
- **研究與教學**：推論程式碼完整開源（含批次引擎與 MLX 移植），適合作為語音合成研究與課程的基礎。

專案由 [OpenFormosa](https://github.com/OpenFormosa) 社群維護，正在尋找以下合作：

- **語音資料與語者授權**：擴充台灣在地語者，以及台語、客語方向的延伸。
- **部署試點**：有實際場景想導入 TTS 的團隊，歡迎交流需求與使用回饋。
- **開發貢獻**：效能最佳化、硬體支援、文件改進。

聯繫方式：[GitHub Issues](https://github.com/OpenFormosa/BlueMagpie-TTS/issues) 或 [Hugging Face 討論區](https://huggingface.co/OpenFormosa/BlueMagpie-TTS/discussions)。

## 命名由來

專案全名是 **OpenFormosa Blue Magpie TTS**。「藍鵲」取自**台灣藍鵲**（Taiwan Blue Magpie，學名 *Urocissa caerulea*）。選牠作為 TTS 的識別有幾層用意：
- **會發聲、辨識度高**：台灣藍鵲本來就是叫聲響亮、容易辨認的鳥，恰好呼應 TTS「把文字變成聲音」的核心。
- **長尾的動態感**：標誌性的長尾巴帶來流動、延展的視覺意象，比一般的喇叭 icon 更有記憶點與品牌個性。
- **立足台灣**：OpenFormosa（福爾摩沙）點出專案面向台灣華語、在地化語音的定位。

## 注意事項

- 上面範例都直接從 `tokenizer.json` 載入 tokenizer 再傳給 `from_local`，在較新版 transformers（5.x）也能穩定運作；背後原因見〈疑難排解〉。
- 模型內附的 `hung_yi_lee` 語者向量已取得本人授權，可直接作為範例使用；指定其他語者或進行聲音複製時，請只使用你已取得授權的參考音檔或語者向量。
- 請妥善保管語者向量表與合成出來的音檔，未經授權前不要對外散布。
- 合成的語音可能不完美；正式使用前請人工檢視。

## 疑難排解

**下載模型時出現 `Token is required (token=True), but no token found`**

模型是公開的，下載**不需要** token。若你參考的範例含有 `snapshot_download(..., token=True)`，把 `token=True` 拿掉即可；或先執行 `huggingface-cli login` 登入。

**`pip install -e .` 安裝失敗**

先確認 Python 版本：本套件支援 **3.10–3.12**，Python 3.13（含以上）目前不支援。另外相依要求 `numpy>=1.26,<2.4`，若既有環境的 numpy 版本衝突，建議建立乾淨的虛擬環境安裝。

**聲音複製（`reference_wav_path`）的輸出內容錯誤、像在亂講話**

請確認使用的是 `step_0006000`（2026-07）之後的 checkpoint——更早的 checkpoint 不支援這條路徑。重新執行 `snapshot_download("OpenFormosa/BlueMagpie-TTS")` 會自動更新到最新版。另外，參考音檔請提供 3 秒以上的乾淨人聲。

**較新版 transformers（5.x）的 tokenizer 載入**

上面的範例都直接從 `tokenizer.json` 載入 tokenizer 再傳給 `from_local`，因此在 transformers 5.x 也能正常運作，不需額外處理（模型只用到 tokenizer 的 `encode`）。

若你改用 `from_local` 的自動載入（不傳入 `tokenizer`），在 transformers 5.x 可能會失敗 —— 解析 `tokenizer_config.json` 時出現

```
TypeError: ..._patch_mistral_regex() got multiple values for keyword argument 'fix_mistral_regex'
```

或載入看似成功、但呼叫 `generate()` 時才報 `ValueError: No tokenizer attached to BlueMagpieModel`。遇到時改回上面範例的明確載入方式即可。
