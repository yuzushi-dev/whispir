# Whispir - Local Transcription & Translation

Whispir è una Web Application locale, privata e sicura per la trascrizione e la traduzione di file audio e video fino a 2 ore, alimentata da **Streamlit** e **faster-whisper**.

L'applicazione gira interamente offline all'interno di un container Docker, senza inviare dati all'esterno ed esponendo il servizio sulla porta **9999**.

---

## Caratteristiche Principali

- 🐳 **Dockerizzato**: Facile da avviare, nessuna necessità di installare python, ffmpeg o librerie C++ complesse sul sistema host.
- 🏎️ **Super Veloce su CPU**: Utilizza `faster-whisper` con quantizzazione `int8` (CTranslate2) per massimizzare l'uso della CPU Intel i9-13950HX senza saturare la RAM.
- 🗄️ **Cache Persistente**: I modelli di Whisper scaricati vengono salvati sul sistema host nella cartella `./models_cache` in modo da non doverli riscaricare ad ogni riavvio del container.
- 📊 **Feedback in Tempo Reale**: Mostra l'avanzamento percentuale, il testo trascritto in tempo reale e stima il tempo residuo.
- 📂 **Export Hub**: Permette di scaricare i risultati nei formati `.txt` (testo semplice), `.srt` (sottotitoli standard) o `.vtt` (formato WebVTT).
- 🎙️ **VAD integrato**: Filtro Silero VAD abilitato di default per eliminare i silenzi ed evitare allucinazioni in video/audio lunghi.

---

## Requisiti

- Docker
- Docker Compose

---

## Come Avviare l'Applicazione

1. Posizionati nella directory principale del progetto:
   ```bash
   cd /home/daniele/Scrivania/Whispir
   ```

2. Costruisci ed avvia il container in background:
   ```bash
   docker compose up --build -d
   ```

3. Apri il browser all'indirizzo:
   [http://localhost:9999](http://localhost:9999)

---

## Spegnere il Servizio

Per fermare e rimuovere i container:
```bash
docker compose down
```

---

## Guida ai Modelli Whisper (Ottimizzati su CPU in `int8`)

| Modello | Dimensione File | Uso RAM (Appross.) | Velocità CPU | Precisione Consigliata |
| :--- | :--- | :--- | :--- | :--- |
| **tiny** | ~75 MB | ~150 MB | Ultra veloce | Bassa (Ideale per test o bozze veloci) |
| **base** | ~140 MB | ~250 MB | Molto veloce | Media |
| **small** *(Default)* | ~460 MB | ~600 MB | Veloce | Buona (Consigliato per trascrizioni generali) |
| **medium** | ~1.5 GB | ~2.0 GB | Moderato | Alta |
| **turbo** | ~1.6 GB | ~2.2 GB | Veloce | Ottima (Eccellente bilanciamento qualità/velocità) |
| **large-v3** | ~3.0 GB | ~3.8 GB | Lento | Massima (Ideale per linguaggi complessi o audio disturbati) |

*Nota: Durante il primo avvio di un modello selezionato, l'applicazione impiegherà alcuni minuti per scaricare i pesi da HuggingFace. Gli avvii successivi dello stesso modello saranno istantanei.*
