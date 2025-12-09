# DO NOT MAKE CHANGES ON MAIN. CLONE REPO LOCALLY.

## venv setup:

###venv installation (Windows)
"python -m venv venv"
"Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force"    
".\venv\Scripts\Activate.ps1"

####Packages
"python -m pip install --upgrade pip setuptools wheel"

"python -m pip install --upgrade "torch" "torchvision" "torchaudio" --index-url https://download.pytorch.org/whl/cpu"

"python -m pip install --no-cache-dir --prefer-binary -r requirements.txt"
