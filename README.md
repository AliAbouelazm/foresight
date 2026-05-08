# Foresight

> **Work in progress.** Architecture and training pipeline are functional but the models are still being improved.

Real-time sports trajectory prediction using deep learning. Given 50 frames of player tracking data, the model predicts the next 50 frames.

## Models

| Model | Architecture | ADE | FDE |
|---|---|---|---|
| TemporalCNN | 4x Conv1d, global avg pool | 0.065 | 0.118 |
| Seq2Seq Transformer | Encoder-decoder, d=128, 4+3 layers | 0.074 | 0.131 |

ADE/FDE in normalized screen coordinates (0-1 range). 50 input frames, 50 predicted frames. Uncertainty via Monte Carlo dropout (10 passes).

## Stack

**Backend:** FastAPI, PyTorch, HuggingFace Hub  
**Frontend:** React 18, Vite, Recharts  
**Data:** SportsMOT / DanceTrack multi-object tracking datasets

## Setup

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download data and preprocess
python -m app.data.preprocessor

# Train
python -m app.model.train transformer

# Evaluate
python -m app.model.evaluate transformer

# Run server
uvicorn app.main:app --reload --reload-dir app
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Set `VITE_API_URL` in `frontend/.env.local` to point to the backend when not running locally.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Model status |
| POST | `/api/predict` | Run prediction with MC dropout |
| GET | `/api/model/training-history` | Loss curves |
| GET | `/api/model/metrics` | ADE/FDE on test set |
| GET | `/api/trajectories/random` | Random test sample |
| GET | `/api/trajectories/samples/{id}` | Specific test sample |

## Data

Trajectories are loaded from [SportsMOT](https://github.com/MCG-NJU/SportsMOT) or [DanceTrack](https://github.com/DanceTrack/DanceTrack), annotation files only. Coordinates are normalized to [0,1] by image dimensions. Tracks shorter than 30 frames or with gaps larger than 3 frames are discarded.
