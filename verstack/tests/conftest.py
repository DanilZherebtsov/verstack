import sys
import os

# Get the verstack directory (parent of tests)
verstack_dir = os.path.dirname(os.path.dirname(__file__))
tests_dir = os.path.dirname(__file__)

# Remove any existing verstack-related paths from sys.path
sys.path = [p for p in sys.path if not (
    'verstack' in p and ('site-packages' in p or 'dist-packages' in p)
)]

# Insert local paths at the very beginning
sys.path.insert(0, verstack_dir)
sys.path.insert(0, tests_dir)

print(f"Verstack dir: {verstack_dir}")
print(f"Tests dir: {tests_dir}")
print(f"Cleaned sys.path first 5: {sys.path[:5]}")

# Debug: Check what's in the directories
print(f"Contents of verstack_dir: {os.listdir(verstack_dir)}")
if os.path.exists(os.path.join(verstack_dir, 'lgbm_optuna_tuning')):
    print(f"lgbm_optuna_tuning exists: {os.path.exists(os.path.join(verstack_dir, 'lgbm_optuna_tuning', 'LGBMTuner.py'))}")
if os.path.exists(os.path.join(verstack_dir, 'categoric_encoders')):
    print(f"categoric_encoders exists: {os.path.exists(os.path.join(verstack_dir, 'categoric_encoders', 'WeightOfEvidenceEncoder.py'))}")