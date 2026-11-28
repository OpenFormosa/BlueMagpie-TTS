# 疑難排解


**下載模型時出現 `Token is required (token=True), but no token found`**

模型是公開的，下載**不需要** token。若你參考的範例含有 `snapshot_download(..., token=True)`，把 `token=True` 拿掉即可；或先執行 `huggingface-cli login` 登入。

**`pip install -e .` 安裝失敗**

先確認 Python 版本：本套件支援 **3.10–3.12**，Python 3.13（含以上）目前不支援。另外相依要求 `numpy>=1.26,<2.4`，若既有環境的 numpy 版本衝突，建議建立乾淨的虛擬環境安裝。

**聲音複製（`reference_wav_path`）的輸出內容錯誤、像在亂講話**

請確認使用的是 `step_0006000`（2026-07）之後的 checkpoint，更早的 checkpoint 不支援這條路徑。重新執行 `snapshot_download("OpenFormosa/BlueMagpie-TTS")` 會自動更新到最新版。另外，參考音檔請提供 3 秒以上的乾淨人聲。

**較新版 transformers（5.x）的 tokenizer 載入**

上面的範例都直接從 `tokenizer.json` 載入 tokenizer 再傳給 `from_local`，因此在 transformers 5.x 也能正常運作，不需額外處理（模型只用到 tokenizer 的 `encode`）。

若你改用 `from_local` 的自動載入（不傳入 `tokenizer`），在 transformers 5.x 可能會失敗。解析 `tokenizer_config.json` 時會出現

```
TypeError: ..._patch_mistral_regex() got multiple values for keyword argument 'fix_mistral_regex'
```

或載入看似成功、但呼叫 `generate()` 時才報 `ValueError: No tokenizer attached to BlueMagpieModel`。遇到時改回上面範例的明確載入方式即可。

[回到 README](../README.md)
