# 從 Hugging Face 使用模型

本文件說明如何從 Hugging Face 下載 BlueMagpie-TTS 模型並載入使用。

## 下載並載入

```python
from huggingface_hub import snapshot_download
from bluemagpie import BlueMagpieModel

model_dir = snapshot_download("OpenFormosa/BlueMagpie-TTS", token=True)
model = BlueMagpieModel.from_local(model_dir, training=False, device="cuda")
```

- `token=True` 會使用你本機已登入的 Hugging Face 權杖；私有模型必須先登入並具備存取權限。
- 載入後即可呼叫 `model.generate(...)` 合成語音，詳細用法請見 [README](../README.md)。

## 模型目錄包含的檔案

下載下來的模型目錄會包含以下檔案：

- `pytorch_model.bin` —— 模型權重
- `audiovae.pth` —— 音訊解碼器權重
- `config.json` —— 模型設定
- `tokenizer.json`、`tokenizer_config.json` —— 斷詞器
- `README.md`、`USAGE.md` —— 說明文件
- `release_metadata.json` —— 釋出資訊

`from_local` 會自動讀取這些檔案，你不需要手動處理。

## 使用提醒

進行聲音克隆或指定語者合成時，請只使用已取得授權的參考音檔或語者向量。
