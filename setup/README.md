# Fast LambdaLabs Setup

## What to Transfer to `/home/ubuntu/NIR-DOT/`
1. `code/` folder (your source code)
2. `nirfaster-FF/` folder (NIRFASTer)
3. `setup/` folder (this folder)

## Setup Commands (Run each new instance)

### 1. Fast Bootstrap (2-3 minutes vs 20+ minutes!)
```bash
chmod +x setup/bootstrap_lambdalabs.sh
./setup/bootstrap_lambdalabs.sh
```

### 2. Install Dependencies  
```bash
source /home/ubuntu/NIR-DOT/venv_diss/bin/activate
cd mah422
pip install -r setup/requirements.txt
```

### 3. Start Training
```bash
source /home/ubuntu/NIR-DOT/venv_diss/bin/activate
python -m code.training.stage1_trainer
```

## Daily Usage (Persistent Environment)

After first setup, just activate environment:
```bash
source /home/ubuntu/NIR-DOT/venv_diss/bin/activate
python -m code.training.stage1_trainer
```

**Note**: Virtual environment is on persistent storage and survives instance restarts!