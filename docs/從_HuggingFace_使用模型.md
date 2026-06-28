# 從 Hugging Face 使用模型

本文件說明如何從 Hugging Face 下載 BlueMagpie-TTS 模型並載入使用。

## 下載並載入

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

- `token=True` 會使用你本機已登入的 Hugging Face 權杖；私有模型必須先登入並具備存取權限。
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
- `release_metadata.json` —— 釋出資訊
- `checkpoints/speaker_centroids.pt` —— 內附的多語者向量表（`hung_yi_lee`、`female_voice` 兩個語者），用 `speaker_centroid` 指定音色

`from_local` 會自動讀取這些檔案，你不需要手動處理。

## 指定語者

模型內附 `checkpoints/speaker_centroids.pt`（含 `hung_yi_lee`、`female_voice` 兩個語者）。
依語者 ID 取出 `[192]` 向量、用 `speaker_centroid` 指定音色，完整程式碼見
[README 的〈指定語者〉](../README.md#指定語者以語者向量控制音色) /
[USAGE](../USAGE.md#speaker-selection-control-timbre-with-a-speaker-vector)。

## 使用提醒

進行聲音複製或指定語者合成時，請只使用已取得授權的參考音檔或語者向量。
