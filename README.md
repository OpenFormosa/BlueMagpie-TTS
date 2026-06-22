# BlueMagpie-TTS

BlueMagpie-TTS 是一套文字轉語音（TTS）模型，能把文字合成為自然的語音。它支援三種使用情境：

- **一般語音合成** —— 直接把文字唸出來。
- **聲音克隆** —— 提供一段參考音檔，讓輸出模仿該語者的音色。
- **指定語者** —— 用事先準備好的語者向量來控制音色。

同時也提供**串流輸出**，適合需要邊合成邊播放的應用。

> 本文件只說明如何安裝與使用模型。

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
from huggingface_hub import snapshot_download
from bluemagpie import BlueMagpieModel

model_dir = snapshot_download("OpenFormosa/BlueMagpie-TTS", token=True)
model = BlueMagpieModel.from_local(model_dir, training=False, device="cuda")
```

### 從本機目錄載入

如果你已經有一份模型檔案，直接指向該目錄即可：

```python
from bluemagpie import BlueMagpieModel

model = BlueMagpieModel.from_local("checkpoints/bluemagpie", training=False, device="cuda")
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

## 聲音克隆：以參考音檔模仿語者

提供一段 `reference_wav_path`，輸出就會模仿該段音檔的音色：

```python
audio = model.generate(
    target_text="今天天氣真好。",
    reference_wav_path="speaker.wav",
    cfg_value=2.0,
)
```

## 指定語者：以語者向量控制音色

如果你手上有事先算好的語者向量（speaker centroid），可以透過 `speaker_centroid` 指定音色。向量表通常以 `torch.load` 載入後，依語者 ID 取出對應的向量：

```python
import torch

centroids = torch.load("speaker_centroids.pt")
table = {sid: centroids["centroids"][i] for i, sid in enumerate(centroids["speaker_ids"])}

audio = model.generate(
    target_text="今天天氣真好。",
    speaker_centroid=table["<speaker_id>"],   # 也可以直接傳入一段新的語者向量
    cfg_value=2.0,
)
```

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
| 參考音檔＋接續 | 以上參數合併使用 | 同時指定音色並接續語音 |

## `generate` 常用參數

| 參數 | 預設值 | 說明 |
|---|---|---|
| `target_text` | （必填） | 要合成的文字 |
| `prompt_text` | `""` | 提示文字，搭配 `prompt_wav_path` 做語音接續 |
| `prompt_wav_path` | `""` | 提示音檔路徑，用於語音接續 |
| `reference_wav_path` | `""` | 參考音檔路徑，用於聲音克隆 |
| `speaker_centroid` | `None` | 語者向量，用於指定音色 |
| `cfg_value` | `2.0` | 引導強度，數值越大越貼合條件、但可能較不自然 |
| `inference_timesteps` | `10` | 取樣步數，越多通常品質越好、速度越慢 |
| `min_len` / `max_len` | `2` / `2000` | 輸出長度的下限與上限 |
| `retry_badcase` | `False` | 偵測到異常輸出時自動重試（串流模式不支援） |

## 注意事項

- 載入模型時若沒有一併附上 tokenizer，`from_local` 會嘗試從模型目錄自動載入；請確認模型目錄內含對應的 tokenizer 檔案。
- 沒有 GPU 也可以執行：把 `device` 設為 `"cpu"` 即可（速度較慢，但短句合成只需數十秒）。輸出為 48 kHz 單聲道。
- 進行聲音克隆或指定語者合成時，請只使用已取得授權的參考音檔或語者向量。
- 請妥善保管語者向量表與合成出來的音檔，未經授權前不要對外散布。

## 疑難排解

**tokenizer 自動載入失敗（較新版 transformers）**

若 `from_local` 自動載入 tokenizer 時出現類似

```
TypeError: ..._patch_mistral_regex() got multiple values for keyword argument 'fix_mistral_regex'
```

這是 transformers 5.x 在解析模型 `tokenizer_config.json` 時的已知問題。改為自行從 `tokenizer.json` 載入再傳入即可（模型只用到 tokenizer 的 `encode`）：

```python
import os
from transformers import PreTrainedTokenizerFast
from bluemagpie import BlueMagpieModel

tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_dir, "tokenizer.json"))
model = BlueMagpieModel.from_local(model_dir, tokenizer=tok, training=False, device="cpu")
```
