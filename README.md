# ScoreForge — Gelişmiş Audio-to-MIDI & Nota Stüdyosu

Spektrogram üzerinde nota analizi ve düzenleme modern derin öğrenme modelleri (**Demucs 6-Stem** + **Spotify Basic Pitch**), müzik teorisi motoru ve canlı **MusicXML / PDF** nota kağıdı çıktısıyla birleştiren native PySide6 masaüstü uygulaması.

## Özellikler
- **6-Stem Ayrıştırma (CUDA / Demucs `htdemucs_6s`):** Şarkıyı otomatik olarak Piyano, Vokal, Gitar, Bas, Davul ve Diğer kanallarına ayırır.
- **Polifonik AI Transkripsiyon + Pitch Bend:** Basic Pitch (ONNX) ile yüksek doğruluklu çok sesli nota algılama ve davul stem'i için GM Kanal 10 vuruş analizi.
- **Nota-Hizalı CQT Spektrogram & Piano Roll Editörü:** 88 tuşlu (A0–C8) piyano klavyesi, spektrogram opaklık/renk paleti kontrolü, nota çizme/taşıma/uzatma/silme, Velocity düzenleyici, `Ctrl+Z` / `Ctrl+Y` geri al/ileri al.
- **Müzik Teorisi Motoru:** Otomatik BPM, ölçü (4/4, 3/4), ton/gam tespiti, zaman cetvelinde canlı akor etiketleri (`Am7`, `F`, `C`, `G`), kuantizasyon ve Tona Oturt (Scale Snap).
- **Senkron Ses + MIDI Oynatıcı:** Harici SoundFont gerektirmeden orijinal ses (veya tek tek stem'ler) ile MIDI'yi senkron dinleme.
- **Dışa Aktarım:** Çok kanallı `.mid`, `.musicxml`, `.pdf` nota kağıdı ve `.sforge` proje kaydı.

## Çalıştırma
`run.bat` dosyasına çift tıklayabilir veya terminalden çalıştırabilirsiniz:
```powershell
.\.venv\Scripts\python.exe main.py
```
