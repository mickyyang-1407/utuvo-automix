# UTUVO Automix

把分軌丟進去、選一種風格，在你自己的電腦上拿到一份混音。

[English](#english)

UTUVO Automix是一個好玩的自動混音工具：讀你的stems或多軌，自動判斷每一軌是什麼（鼓、貝斯、人聲、吉他、pad……），依你選的風格做增益、EQ、壓縮、空間、聲像與母帶響度，輸出一個WAV。它不是用來取代混音師的；正式作品還是請人好好混。它的目標是讓你很快聽到「這首歌換成某種風格大概會是什麼樣子」，而且至少比直接把分軌加起來好聽一點。

- 全部在本機執行，音訊不會上傳到任何地方。
- 14種風格：Acoustic、Ambient／Space、Cinematic、Funk／Disco、Heavy Rock／Metal、House／Dance、Indie Pop、Jazz／Natural、Lo-fi、Pop 2020s、Reggae／Groove、R&B／Soul、Rock、Trap／Hip-Hop。每種風格的數值都是UTUVO自己調出來的起點，不是任何人的配方，也不是音質保證。
- 等響度A/B：混音後與原始總和用同一個響度比較，不會因為比較大聲就覺得比較好。
- 每次混音都會存一份可編輯的`mixplan.json`；同一份計畫永遠渲染出同一個結果。
- 可以另外匯出處理後的分軌給DAW（不含bus/master處理）。

## 怎麼用

### 分軌要怎麼準備

- 所有分軌**從歌曲的同一個起點匯出**（DAW裡全部從第0秒開始）。程式只會在短的分軌尾巴補靜音，不會幫你找對齊。
- WAV可以直接讀；AIFF、FLAC、MP3、M4A、OGG需要電腦上有[ffmpeg](https://ffmpeg.org/)（macOS：`brew install ffmpeg`）。
- 單聲道、立體聲、不同取樣率都可以混在一起；取樣率會統一成多數分軌用的那個。
- 介面一次最多64軌，每個檔案最多256MB。
- 檔名有寫清楚會幫助判斷角色（例如`kick.wav`、`bass DI.wav`、`lead vox.wav`）；判斷錯了可以在介面上改。

### 從原始碼執行

需要Python 3.11以上。

```bash
python3 -m pip install -r requirements.txt
cd app
python3 app.py
```

瀏覽器會自動打開本機頁面（`http://127.0.0.1:8765/`左右）。拖進分軌、選風格、按「開始混音」。

指令列：

```bash
cd app
python3 -m automix.cli --list-styles
python3 -m automix.cli --stems /path/to/stems --style lofi --out /path/to/output
```

### 打包成macOS App

```bash
cd app
AUTOMIX_BUILD_PYTHON=/path/to/python3 ./build_app.sh
```

需要PyInstaller。打包出來的App只有本機簽章；要發給別人用，需要自己用Developer ID簽章並公證。

### 跑測試

```bash
python3 -m pip install -r requirements-dev.txt
PYTHONPATH=app python3 -m pytest app/tests
```

## 限制（請先看）

- 這是起點，不是成品。好不好聽要靠耳朵判斷。
- 自動判斷角色靠檔名與聲音特徵，可能判錯。
- 相位只保護你明確指定成一組的多支麥克風或立體聲對，不會自動幫所有來源校正相位。
- 匯出給DAW的分軌不含bus/master處理；直接加總不會等於`mix.wav`。
- 介面目前只有繁體中文。

## 授權

見[LICENSE](LICENSE)。

---

## English

Drop in your stems, pick a style, get a mix — on your own computer.

UTUVO Automix is a playful automatic mixer. It reads stems or multitracks, guesses what each track is (drums, bass, vocals, guitars, pads…), and applies gain staging, EQ, compression, space, panning and master loudness in the style you choose, then writes a WAV. It is not a replacement for a mix engineer — real releases still deserve a real mix. The goal is to hear quickly what a song might sound like in a given style, and to sound at least a bit better than just summing the stems.

- Runs locally; no audio leaves your machine.
- 14 styles. Every style's numbers are UTUVO's own starting points — nobody's recipe and no quality guarantee.
- Loudness-matched A/B between the mix and the raw sum, so louder never wins by accident.
- Every mix saves an editable `mixplan.json`; the same plan always renders the same audio.
- Optional processed-stem export for your DAW (without bus/master processing).

### Preparing stems

- Export every stem **from the same start point** (bar 1 / time 0). Shorter stems are padded at the end; nothing is time-aligned for you.
- WAV is read directly; AIFF, FLAC, MP3, M4A and OGG need [ffmpeg](https://ffmpeg.org/) (`brew install ffmpeg` on macOS).
- Mono/stereo and different sample rates may be mixed; everything is resampled to the rate most stems use.
- The GUI takes up to 64 stems, 256 MB per file.

### Run from source

Python 3.11+.

```bash
python3 -m pip install -r requirements.txt
cd app
python3 app.py            # opens the local web UI
python3 -m automix.cli --stems /path/to/stems --style lofi --out /path/to/output
```

Build a macOS app with `app/build_app.sh` (PyInstaller; ad-hoc signed — sign and notarize with your own Developer ID to distribute). Tests: `pip install -r requirements-dev.txt` then `PYTHONPATH=app python3 -m pytest app/tests`.

### Limits

A starting point, not a finished master; judge by ear. Role detection can be wrong. Phase is only protected for tracks you explicitly group. Exported DAW stems exclude bus/master processing. The UI is Traditional Chinese only for now.

### License

See [LICENSE](LICENSE).
