# Progetto di Riconoscimento Caratteri CJK tramite Architettura CNN Custom

Il presente progetto si focalizza sullo sviluppo di un sistema di riconoscimento automatico di caratteri CJK (Cinese, Giapponese, Coreano) basato su un’architettura di rete neurale convoluzionale (CNN) progettata ad hoc, con un forte orientamento all’efficienza computazionale e alle prestazioni. 

L’obiettivo è realizzare un modello con un numero ridotto di parametri, rapido nell’inferenza, leggero in termini di risorse computazionali, ma al contempo affidabile e accurato.



# Punti toccati
## 1. Librerie e strumenti per analisi e visualizzazione dati
Per l’analisi esplorativa e la visualizzazione dei dati sono state utilizzate librerie Python consolidate e performanti:

- **Seaborn e Matplotlib**: per la generazione di grafici statistici e visualizzazioni, in particolare per la matrice di confusione, strumento fondamentale per valutare le prestazioni del modello su classi multiple.



## 2. Analisi del dataset

Il dataset principale comprende 44 classi di caratteri, ciascuna rappresentata da circa 400 immagini, un volume dati comparabile con dataset pubblici di riferimento per il riconoscimento CJK.

- **Bilanciamento delle classi**: le classi risultano quais bilanciate, la differenza di campioni non supera il rapport 2:1 tra le 2 classi con differenza maggiore in numero di campioni.
- **Quantità dati**: la dimensione del dataset è stata valutata sufficiente per addestrare modelli CNN di complessità moderata su 44 classi.
- **Pulizia e selezione dati**: si è cercato di bilanciare le classi con lo stesso numero di campioni, in alcuni casi si è preferito un piccolo aumento di campioni quando il modello faticava per una certa classeì.



## 3. Architettura modulare del codice

Il progetto è stato strutturato in moduli distinti per favorire la manutenibilità, la scalabilità e la riusabilità del codice:

1. **Config schema**: definizione di schemi JSON (JSON Schema) per la validazione dei file di configurazione (`config.json` e `model_config.json`), assicurando coerenza e correttezza dei parametri.
2. **DataLoader**: utilizzo del dataset e del DataLoader di PyTorch per il caricamento efficiente delle immagini, con supporto a batching, processing delle immagini e shuffling.
3. **Models**: sviluppo iniziale di un modello CNN implementato manualmente, successivamente evoluto in un sistema dinamico che consente la definizione dell’architettura tramite file JSON, facilitando sperimentazioni rapide e configurazioni flessibili.
4. **Pre-processing**: pipeline di preprocessing che include il taglio delle immagini da griglie composite in singoli caratteri, suddivisione in set di training, validazione e test, ridimensionamento alle dimensioni specificate nel file di configurazione, normalizzazione e conversione opzionale in scala di grigi come solitamente accade con il riconoscimento di caratteri.
5. **Trainer**: modulo centrale che gestisce le fasi di training, validazione e testing, con una classe `CNN_engine` responsabile dell’addestramento e dell’inferenza, implementando le logiche di forward, backward e aggiornamento dei pesi.
6. **Utils**: raccolta di funzioni e classi di utilità, tra cui salvataggio delle configurazioni, early stopping, scheduler per il learning rate con selezione dinamica, ottimizzatori configurabili, enumerazioni per la gestione delle configurazioni e validazione dei parametri.
7. **Config e model_config**: file JSON che definiscono in modo parametrico e dinamico l’architettura della rete e i parametri di training, permettendo una facile modifica senza interventi sul codice.



## 4. Gestione delle configurazioni

L’intero sistema è parametrizzato tramite file JSON:

- **config.json**: definisce i parametri di training, preprocessing, data augmentation, e altre impostazioni operative.
- **model_config.json**: specifica la struttura della rete, inclusi layer, dimensioni kernel, funzioni di attivazione, dropout, batch normalization, ecc. e il flusso di lavoro dei layer.
- **config schema**: schema JSON utilizzato per validare i file di configurazione, prevenendo errori di sintassi o valori non validi.



## 5. Tecniche di data augmentation e preprocessing

- La data augmentation è configurabile tramite il file JSON, consentendo di abilitare o disabilitare specifiche tecniche (rotazioni, traslazioni, rumore, scaling, ecc.) e di definire il numero di immagini generate per ogni immagine originale.
- Il preprocessing prevede:
  - Segmentazione delle immagini da griglie composite in singoli caratteri.
  - Suddivisione del dataset in training (70%), validazione (20%) e test (10%).
  - Ridimensionamento e normalizzazione delle immagini secondo parametri definiti.
  - Conversione opzionale in scala di grigi con riduzione a un singolo canale.
- L’augmentation è applicata esclusivamente durante la fase di training per migliorare la generalizzazione del modello.



## 6. DataLoader e Dataset

Il DataLoader standard di PyTorch è risultato adeguato per il task, senza necessità di implementare classi custom per dataset o caricamento dati.



## 7. Validazione durante l’addestramento

- La suddivisione in train/val/test consente di monitorare in tempo reale le prestazioni del modello su dati non visti durante il training.
- La validazione è utilizzata per:
  - Monitorare l’andamento del training.
  - Salvare il modello con le migliori prestazioni.
  - Prevenire overfitting e migliorare la generalizzazione tramite early stopping.



## 8. Testing finale

Al termine dell’addestramento, il modello con le migliori metriche di validazione viene testato su un set di dati completamente nuovo, fornendo una stima accurata delle prestazioni reali.



## 9. Transfer Learning e Fine-tuning

- Il modello viene inizialmente addestrato su un dataset A senza data augmentation.
- Successivamente, si esegue un fine-tuning con un learning rate ridotto di un fattore 10, utilizzando un dataset A con data augmentation per aumentare la robustezza a variazioni geometriche e rumore.
- Per l’estensione a nuove classi, il modello viene ulteriormente addestrato su un dataset A + dataset B ampliato contenente sia nuove immagini che immagini del dataset originale, mitigando il fenomeno del catastrophic forgetting e preservando le conoscenze acquisite.



## 10. Salvataggio e ripresa del training

- Il sistema consente il salvataggio automatico del modello migliore basato su metriche configurabili (ad esempio, loss di validazione).
- I checkpoint includono pesi, configurazioni di training, stato dell’ottimizzatore e scheduler, permettendo di interrompere e riprendere l’addestramento senza perdita di informazioni.



## 11. Early stopping

- Implementazione di early stopping configurabile tramite parametri quali:
  - Epoca di inizio monitoraggio.
  - Pazienza (numero di epoche senza miglioramento prima di fermare il training).
  - Soglia di miglioramento atteso.
  - Metrica di riferimento (loss, accuracy, ecc.).
- Questa tecnica consente di evitare overfitting e di ridurre i tempi di addestramento.



## 12. Logging e monitoraggio con TensorBoard

- Registrazione di immagini di batch per analisi visiva del dataset.
- Tracciamento di grafici di training e validazione (loss, accuracy).
- Salvataggio di iperparametri, pesi, bias e gradienti per un monitoraggio dettagliato del processo di addestramento.



## 13. Sperimentazione con architetture e dataset

- La progettazione dell’architettura è stata incrementale, partendo da una CNN semplice fino all’introduzione di:
  - Maggior numero di layer e kernel.
  - Tecniche di regolarizzazione come dropout e batch normalization.
  - Pooling (average e max) per ridurre dimensionalità e migliorare la generalizzazione.
- Sono stati testati più dataset con classi differenti per validare la flessibilità e la robustezza del modello e con iperparametri variabili.



## 14. Ottimizzazione degli iperparametri

- Sperimentazione su learning rate, numero di epoche, batch size e altri parametri per stabilizzare il training e massimizzare le prestazioni.
- In particolare, durante transfer learning e fine-tuning, il numero di epoche può essere ridotto in quanto si parte da un modello pre-addestrato.
- L’ottimizzazione dinamica degli iperparametri è supportata da scheduler configurabili che permette di diminuire il learning rate man mano che il modello migliora.
- L'architettura include la possibilità di utilizzare un warmup-scheduler che permette di addestrare il modello usando un learning-rate incrementale a parte da un lr molto basso fino ad arrivare al lr di base, questo è utile all'inizio quando i pesi sono randomizzati e si vuole evitare che il gradiente influisca troppo sull'aggiornamento dei pesi.



## 15. Integrazione con interfaccia grafica (facoltativa)

- Il modello è stato integrato con un’applicazione Android sviluppata separatamente.
- L’applicazione consente il riconoscimento dei caratteri in tempo reale, comunicando con un server Python tramite API REST per eseguire l’inferenza.
