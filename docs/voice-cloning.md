# 聲音複製與指定語者

BlueMagpie-TTS 可以模仿參考語者的音色，也可以用語者向量指定音色。請只使用你已取得授權的聲音。

## 聲音複製：模仿參考語者的音色


兩種做法，**都不需要**參考音檔的逐字稿：

**A. 參考音檔（`reference_wav_path`）**：最簡單的做法。直接給一段 3 秒以上的乾淨語音：

```python
audio = model.generate(
    target_text="今天天氣真好。",
    reference_wav_path="reference.wav",
    cfg_value=2.0,
)
```

> 此路徑自 checkpoint `step_0006000`（2026-07 發佈）起正式支援（內部評測 CER 8.99%）。**更早的 checkpoint 不支援這條路徑，會產生內容錯誤的語音**；請重新執行 `snapshot_download` 取得最新模型。

**B. 語者向量（`speaker_centroid`）**：品質最穩定（內部評測 CER 7.44%）。先從參考音檔抽出語者向量再合成，多段參考音檔可平均出更穩的音色：

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

[回到 README](../README.md)
