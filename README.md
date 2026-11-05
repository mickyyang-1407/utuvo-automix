# UTUVO Automix

[繁體中文](#繁體中文)

Drop in your stems, pick a style, get a mix — on your own computer.

UTUVO Automix is a playful automatic mixer. It reads stems or multitracks, guesses what each track is (drums, bass, vocals, guitars, pads…), and applies gain staging, EQ, compression, space, panning and master loudness in the style you choose, then writes a WAV. It is not a replacement for a mix engineer — real releases still deserve a real mix. The goal is to hear quickly what a song might sound like in a given style, and to sound at least a bit better than just summing the stems.

- Runs locally; no audio leaves your machine.
- 14 styles: Acoustic, Ambient/Space, Cinematic, Funk/Disco, Heavy Rock/Metal, House/Dance, Indie Pop, Jazz/Natural, Lo-fi, Pop 2020s, Reggae/Groove, R&B/Soul, Rock, Trap/Hip-Hop. Every style's numbers are UTUVO's own starting points — nobody's recipe and no quality guarantee.
- Mixing moves chosen by the style: the bass's low end ducks under the kick (a sidechain on the band below the kick's fundamental only, so the bass line keeps its level), parallel compression on the drums, a vocal delay locked to the song's tempo (a slapback when the tempo is not clear), the reverb's pre-delay and decay locked to the tempo too, and the vocal, drums and bass each checked against the rest of the mix as you hear it (through the master EQ) — when that check moves one of them, its reverb and delay move with it, like a post-fader send. Saturation is oversampled and follows each passage's own level, so it adds harmonics (each cycle is still shaped, as by any saturator) without pulling the quiet passages up against the loud ones. Compressors read only what has already happened, as a hardware compressor does, so a slow attack lets a drum hit through before the gain comes down. The vocal's reverb steps back while the voice sings and blooms in the gaps between phrases (as wet as before under the words). Multi-mic drums are lined up — kick in/out, snare top/bottom, the overheads onto the snare — and a reversed mic is flipped, but only when the measurement is clear; otherwise the mic stays as recorded and the report says why (Jazz and Acoustic only fix polarity; toms are only checked for polarity, rooms are never moved). The lead vocal gets a light artificial double (two copies a few ms late whose pitch drifts a few cents, left and right; skipped when the session already stacks the voice or the lead is already stereo), and its narrow resonances are pulled down only where they stick out (mostly 2–8 kHz, capped over the whole song). The report says what each one measurably did.
- When a mix is almost right: five faders under the player (vocals, drums, bass, instruments, FX & space). Each moves its group after every automatic decision, its reverb and delay with it; press “Apply and mix again”.
- Loudness-matched A/B between the mix and the raw sum, so louder never wins by accident.
- Every mix saves an editable `mixplan.json`; the same plan always renders the same audio.
- Optional processed-stem export for your DAW (without bus/master processing).
- English UI by default; switch to Traditional Chinese with the toggle in the header (Chinese browsers start in Chinese).

## Preparing stems

- Export every stem **from the same start point** (bar 1 / time 0). Shorter stems are padded at the end. Only the mics of a drum kit are lined up for you, and only when the measurement is clear.
- WAV is read directly; AIFF, FLAC, MP3 and other formats need [ffmpeg](https://ffmpeg.org/) (`brew install ffmpeg` on macOS).
- Mono/stereo and different sample rates may be mixed; everything is resampled to the rate most stems use.
- The GUI takes up to 64 stems, 256 MB per file.

## Run from source

Python 3.11+ (tested on 3.11 and 3.14).

```bash
python3 -m pip install -r requirements.txt
cd app
python3 app.py            # opens the local web UI
python3 -m automix.cli --stems /path/to/stems --style lofi --out /path/to/output
```

Build a macOS app with `app/build_app.sh` (PyInstaller; ad-hoc signed — sign and notarize with your own Developer ID to distribute). Tests: `pip install -r requirements-dev.txt` then `PYTHONPATH=app python3 -m pytest app/tests`.

## Limits

A starting point, not a finished master; judge by ear. Role detection can be wrong. Mics you group explicitly are never moved; drum mics are lined up only when the measurement is clear, and no other source is phase-corrected. Exported DAW stems exclude bus/master processing.

## License

See [LICENSE](LICENSE).

---

## 繁體中文

UTUVO Automix是一個好玩的自動混音工具：讀你的stems或多軌，自動判斷每一軌是什麼（鼓、貝斯、人聲、吉他、pad……），依你選的風格做增益、EQ、壓縮、空間、聲像與母帶響度，輸出一個WAV。它不是用來取代混音師的；正式作品還是請人好好混。它的目標是讓你很快聽到「這首歌換成某種風格大概會是什麼樣子」，而且至少比直接把分軌加起來好聽一點。

- 全部在本機執行，音訊不會上傳到任何地方。
- 14種風格：Acoustic、Ambient／Space、Cinematic、Funk／Disco、Heavy Rock／Metal、House／Dance、Indie Pop、Jazz／Natural、Lo-fi、Pop 2020s、Reggae／Groove、R&B／Soul、Rock、Trap／Hip-Hop。每種風格的數值都是UTUVO自己調出來的起點，不是任何人的配方，也不是音質保證。
- 依風格加上的混音手法：bass的低頻在大鼓打下去時讓開（sidechain，只動大鼓基頻附近以下，bass的線條不受影響）、鼓組平行壓縮、跟著歌曲速度的人聲delay（抓不到穩定速度時改用短回音slapback）、殘響的pre-delay與長度也對齊歌曲速度，以及人聲、鼓、bass各自對其他聲部的音量自動檢查（以經過母帶EQ、實際聽到的聲音來量）——檢查把哪一軌拉大或拉小，它的殘響和delay就跟著一起動（像推子後的送出）。飽和以超取樣處理，跟著每一段自己的大小走：加泛音（每個波形週期仍會被塑形，飽和本來就是這樣），但不會把小聲段落拉上來、把大小聲差距壓掉。多支麥克風的鼓會對齊：kick內外、snare上下、OH對到snare，反相的那支會翻回來——但只在量得很清楚時才動，不清楚就照原本錄的、報告寫出原因（Jazz與Acoustic只修極性；tom只檢查極性、room不動）。主唱會加一層很淡的人工雙軌（兩份延遲幾毫秒、音高飄幾個cent、左右各一；分軌裡已經疊了人聲或主唱本身是立體聲就不加），刺耳的窄共振只在突出的瞬間壓下（主要2–8kHz，整首有上限）。報告會寫出每一項實際量到的效果。
- 差一點點的時候：播放器下面有五條推子（人聲、鼓、貝斯、樂器、效果與空間），各自在所有自動決定之後再推大或推小，殘響和delay跟著走；按「套用並重新混音」。
- 等響度A/B：混音後與原始總和用同一個響度比較，不會因為比較大聲就覺得比較好。
- 每次混音都會存一份可編輯的`mixplan.json`；同一份計畫永遠渲染出同一個結果。
- 可以另外匯出處理後的分軌給DAW（不含bus/master處理）。

### 怎麼用

#### 分軌要怎麼準備

- 所有分軌**從歌曲的同一個起點匯出**（DAW裡全部從第0秒開始）。程式會在短的分軌尾巴補靜音；只有鼓組的多支麥克風會幫你對齊，而且只在量得很清楚時。
- WAV可以直接讀；AIFF、FLAC、MP3等其他格式需要電腦上有[ffmpeg](https://ffmpeg.org/)（macOS：`brew install ffmpeg`）。
- 單聲道、立體聲、不同取樣率都可以混在一起；取樣率會統一成多數分軌用的那個。
- 介面一次最多64軌，每個檔案最多256MB。
- 檔名有寫清楚會幫助判斷角色（例如`kick.wav`、`bass DI.wav`、`lead vox.wav`）；判斷錯了可以在介面上改。

#### 從原始碼執行

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

#### 打包成macOS App

```bash
cd app
AUTOMIX_BUILD_PYTHON=/path/to/python3 ./build_app.sh
```

需要PyInstaller。打包出來的App只有本機簽章；要發給別人用，需要自己用Developer ID簽章並公證。

#### 跑測試

```bash
python3 -m pip install -r requirements-dev.txt
PYTHONPATH=app python3 -m pytest app/tests
```

### 限制（請先看）

- 這是起點，不是成品。好不好聽要靠耳朵判斷。
- 自動判斷角色靠檔名與聲音特徵，可能判錯。
- 你明確指定成一組的多支麥克風或立體聲對永遠不動；鼓組麥克風只在量得很清楚時對齊，其他來源不做相位校正。
- 匯出給DAW的分軌不含bus/master處理；直接加總不會等於`mix.wav`。
- 介面預設英文，右上角可切換繁體中文（瀏覽器語言是中文時會自動用中文）。

### 授權

見[LICENSE](LICENSE)。
