import subprocess
import sys

STEPS = ['src.load_db', 'src.features', 'src.train', 'src.score', 'src.fairness']
for step in STEPS:
    print('>>>', step)
    subprocess.run([sys.executable, '-m', step], check=True)
