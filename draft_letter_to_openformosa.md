# 【洽詢】商用授權 + MLX Apple Silicon 優化成果分享 + 上游加速建議

您好，我是 BlueMagpie-TTS 的使用者。最近在一台 **Mac mini M4（16GB，base 版 10 核心 GPU）**上完成了完整的部署與優化，過程中做出一些研究成果，想回饋給社群；同時也有商用授權與上游技術方向的想法想跟你們討論。

---

## 一、我們做出的 MLX 優化（可整理成 PR 回饋）

你們的 MLX port 品質很好（parity 測試幫了大忙），但在 16GB 入門機器上直接跑會遇到幾個牆。我們逐一解掉了：

### 1. fp32 / int4 pickle 快取（解決「載入即爆記憶體」）
原路徑 `BlueMagpieMLX(model)` 會同時持有 torch 模型（7.75GB fp32）與 MLX 權重副本（~15GB 峰值），16GB 機器直接窒息。我們做了：
- `bluemagpie/mlx/lite.py`：一次性轉換 → pickle 快取，之後啟動**完全不載 torch 主模型**（輸入組裝只需要 `audiovae.pth` 的 AudioVAE encoder 做輕量代理）
- 支援 fp32 / fp16 / **int4（MLX grouped-affine 4-bit, group 64）** 三種精度，int4 用 fused `mx.quantized_matmul` kernel，權重維持 packed 不膨脹
- int4 checkpoint 含量化 side-car，pickle 後可直接以 `compute="int4-cached"` 秒載（含一個踩過的坑：走訪器必須跳過 side-car，否則 scales 會被二次量化弄壞）

| 配置 | 權重 RAM | 載入 | RTF (M4 base) |
|---|---|---|---|
| 原始 MLX fp32 | ~15 GB 峰值 | 40s+ | 2.84 |
| fp32 pickle | 7.75 GB | 4.8s | 2.0 |
| **int4 cached** | **~2 GB** | **1.5s** | **0.83–0.92** |

### 2. Fast prefill（整段 forward + cache warm，取代逐 token step）
原 `prefill` 是逐 token 跑 cached step（RALM prefill 吃掉 6 秒）。改成整段 full forward（barbet 12×、ralm 也大幅加速），同時把 KV cache 與 mamba conv/SSM state 從 full forward 一併 warm 好，AR 迴圈無縫接續。parity 對齊 ~2e-5，且這其實更貼近你們 torch 參考實作的路徑。

### 3. Streaming（MLX 端 stateful AudioVAE decode）
`mlx_generate_streaming`：AR loop 分塊 yield latents + AudioVAE causal-conv 帶狀態分塊解碼（語義對齊 torch `StreamingVAEDecoder`）。與非串流輸出 corr = 1.0000，**首段音訊 TTFB ~1s**。

### 4. 量化的實測數據（M4 base，可能對你們有用）
- **fp16 GEMM 比 fp32 慢約 70%**（MLX eager 下）— fp16 只適合當儲存格式
- **mxfp4 比 affine 4-bit 慢**（1.35x vs 1.64x vs fp32）— mxfp4 是新晶片硬體路徑，M4 base 反而吃虧
- group size 64 > 32
- 「保留 fp32 權重讓 prefill 走 plain GEMM」在 16GB 機器是負收益（記憶體壓力拖慢全域），≥32GB 才值得
- **品質注意**：4-bit 讓 FSQ 邊界翻轉，內容會變（corr 0.06）；但聽感上我們的使用者可接受，且 cfg=2.8 下表現最佳 — 這點想聽聽你們的看法

以上（lite.py、streaming、fast prefill、bug fix 如新版 MLX 的 `group_for_head` 需要 `dtype=mx.int32`）都可以整理成 PR，歡迎告訴我偏好的拆法。

---

## 二、上游加速建議：DiT 一致性蒸餾

我們把推理端榨乾了（RTF 0.92），剩餘瓶頸的時間分佈很清楚：

- DiT CFG 採樣：**89ms/patch（佔熱迴圈約七成）**
- barbet.step + ralm.step：23ms/patch
- AudioVAE 解碼：85ms/patch（一次性）

**推理端已無能為力的部分**，需要訓練端：

1. **Consistency Distillation for LocDiT**（最想要）：把 9-step Euler CFG 蒸餾成 1–2 步（LCM/一致性模型路線）。估算 RTF 可從 0.92 → **~0.35**，所有 Mac 用戶受惠。
2. **Guidance Distillation**：把 CFG 摺進權重，DiT 每次 forward 的 batch 直接減半（成本砍半）。
3. （選配）MTP 式多 patch 預測 head。

附帶一提，為什麼推理端的 speculative decoding 在這個架構行不通（我們評估過）：每步成本由 DiT 主導，「草稿-驗證」的驗證本身就要跑完整 DiT，等於沒省；且連續 latent 經 FSQ 離散化後對微小偏差極敏感（timesteps 9→7 就讓輸出內容完全分岔，corr 0.16），投機候選幾乎必被否決。所以這題只能靠你們在訓練端解。

---

## 三、商用授權洽詢

我們有實際場景想導入（台灣華語 TTS 服務），需要釐清三件事：

1. **模型權重授權**：HF 模型卡目前標注 research/evaluation 用途。我們理解並尊重這個定位，想洽談正式部署的商用授權條件 — README 提到正在尋找部署試點夥伴，我們正好是。
2. **`female_voice` 語者向量**：來源與授權範圍為何？若範圍不含商業使用，我們會改用自有授權語者錄製參考音檔（這條路我們已確認可行）。
3. **合成音檔的散布**：在上述授權釐清後的範圍確認。

---

## 聯絡方式

請回覆此討論區，或告知偏好的聯絡方式（email）。優化代碼隨時可以整理提交。

謝謝你們做出這麼好的台灣華語 TTS 開源專案！
