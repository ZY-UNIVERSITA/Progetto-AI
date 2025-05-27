## Step per ricreare la virtual environment

1. Installare Python 2.12  
2. Eseguire il comando per creare la virtual environment:  
   ```bash
   python -m venv ai_venv
   ```
3. Attivare la virtual environment:  
   - **Linux/macOS**:  
     ```bash
     source ai_venv/bin/activate
     ```
   - **Windows** (PowerShell):  
     ```powershell
     .\ai_venv\Scripts\Activate.ps1
     ```
     **Windows** Se si usa il prompt dei comandi di Windows (cmd):  
     ```cmd
     .\ai_venv\Scripts\activate.bat
     ```
4. Installare le librerie richieste:  
   ```bash
   pip install -r requirements.txt
   ```
