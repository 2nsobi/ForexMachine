from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()

def make_data_directory():
    (PROJECT_ROOT / f'Data/DataWithIndicators/').mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / f'Data/RawData/').mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / f'Data/TrainingData/').mkdir(parents=True, exist_ok=True)

make_data_directory()
