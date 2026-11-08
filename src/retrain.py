import subprocess
import sys

r = subprocess.run([sys.executable, '-m', 'src.monitor'], check=False)
if r.returncode == 1:
    print('Drift detected -> rebuilding features, retraining, rescoring')
    for mod in ['src.features', 'src.train', 'src.score']:
        subprocess.run([sys.executable, '-m', mod], check=True)
elif r.returncode == 0:
    print('No drift: model kept')
else:
    print('Monitoring failed, check the error above')