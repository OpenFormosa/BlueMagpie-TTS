# 從 Hugging Face 使用模型

本文件說明如何從 Hugging Face 下載 BlueMagpie-TTS 模型並載入使用。

## 下載並載入

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

- 模型是**公開**的，下載不需要 token 或申請權限。若舊範例中的 `token=True` 造成
  `Token is required ... no token found` 錯誤，把該參數拿掉即可。
- 範例直接從 `tokenizer.json` 載入 tokenizer 並傳給 `from_local`，在較新版 transformers（5.x）也能正常運作。
- 載入後即可呼叫 `model.generate(...)` 合成語音，詳細用法請見 [README](../README.md)。
- 若要同時處理多筆請求、追求更高吞吐量，可使用批次推論引擎 `bluemagpie.serving.BlueMagpieEngine`，詳見 README 的〈批次推論引擎〉一節。

## 模型目錄包含的檔案

下載下來的模型目錄會包含以下檔案：

- `pytorch_model.bin` —— 模型權重
- `audiovae.pth` —— 音訊解碼器權重
- `config.json` —— 模型設定
- `tokenizer.json`、`tokenizer_config.json` —— 斷詞器
- `README.md`、`USAGE.md` —— 說明文件
- `release_metadata.json` —— 釋出資訊（checkpoint 版本與內部評測數據）
- `checkpoints/speaker_centroids.pt` —— 內附的多語者向量表（`hung_yi_lee`、`female_voice` 兩個語者），用 `speaker_centroid` 指定音色
- `checkpoints/hung_yi_lee_speaker_centroids.pt` —— 早期的單語者向量表（格式相同，仍可使用）

`from_local` 會自動讀取這些檔案，你不需要手動處理。

## 確認 checkpoint 版本

聲音複製（`reference_wav_path`，免逐字稿）自 checkpoint `step_0006000`（2026-07）起才正式支援。
可以這樣確認手上的版本：

```python
import json, os
meta = json.load(open(os.path.join(model_dir, "release_metadata.json")))
print(meta["checkpoint"])   # 應為 "step_0006000" 或更新
```

若版本較舊，重新執行 `snapshot_download("OpenFormosa/BlueMagpie-TTS")` 即會更新到最新版。

## 指定語者

模型內附 `checkpoints/speaker_centroids.pt`（含 `hung_yi_lee`、`female_voice` 兩個語者）。
依語者 ID 取出 `[192]` 向量、用 `speaker_centroid` 指定音色，完整程式碼見
[README 的〈指定語者〉](../README.md#指定語者以語者向量控制音色) /
[USAGE](../USAGE.md#speaker-selection-control-timbre-with-a-speaker-vector)。

## 使用提醒

進行聲音複製或指定語者合成時，請只使用已取得授權的參考音檔或語者向量。
