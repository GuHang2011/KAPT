import sys
sys.path.insert(0, '/mnt/c/Users/29828/Downloads/KAPT/KAPT-main')

from chemprop.parsing import parse_train_args
from chemprop.train.make_predictions import load_checkpoint
import inspect

args = parse_train_args([
    '--data_path', './data/bbbp.csv',
    '--dataset_type', 'classification',
    '--checkpoint_path', './dumped/pretrained_graph_encoder/original_CMPN_0623_1350_14000th_epoch.pkl'
])

model = load_checkpoint(args.checkpoint_path, device='cpu')

print("=" * 60)
print("Model type:", type(model).__name__)

if hasattr(model, 'encoder'):
    print("\nEncoder type:", type(model.encoder).__name__)
    print("Encoder forward signature:", inspect.signature(model.encoder.forward))

print("\nModel forward signature:", inspect.signature(model.forward))
print("=" * 60)
