# ===== 保存为 test_checkpoint_loading.py =====

import sys
import torch

# 模拟命令行参数
sys.argv = [
    'test_checkpoint_loading.py',
    '--data_path', './data/bbbp.csv',
    '--metric', 'auc',
    '--dataset_type', 'classification',
    '--epochs', '100',
    '--num_runs', '1',
    '--gpu', '0',
    '--batch_size', '50',
    '--seed', '4',
    '--split_type', 'scaffold_balanced',
    '--step', 'kapt',
    '--use_kapt',
    '--freeze_kano',
    '--prompt_lr', '1e-6',
    '--exp_name', 'test_checkpoint',
    '--exp_id', 'verify_loading',
    '--checkpoint_path', './dumped/pretrained_graph_encoder/original_CMPN_0623_1350_14000th_epoch.pkl'
]

sys.path.append('/mnt/c/Users/29828/Downloads/KAPT/KAPT-main')

from chemprop.parsing import parse_train_args
from chemprop.models import build_kapt_model

print("=" * 80)
print("🔍 Step 1: Parsing args...")
print("=" * 80)

args = parse_train_args()
print(f"✅ Args parsed successfully\n")

print("=" * 80)
print("🔍 Step 2: Building KAPT model...")
print("=" * 80)

model = build_kapt_model(args)
print(f"✅ Model built successfully\n")

print("=" * 80)
print("🔍 Step 3: Loading checkpoint...")
print("=" * 80)

checkpoint_path = args.checkpoint_path
state_dict = torch.load(checkpoint_path, map_location='cpu')

print(f"✅ Checkpoint loaded")
print(f"   Keys: {list(state_dict.keys())}\n")

print("=" * 80)
print("🔍 Step 4: Attempting to load encoder weights...")
print("=" * 80)

encoder_loaded = False

# 尝试方式 1
if 'encoder' in state_dict:
    print("🔍 Attempt 1: checkpoint['encoder']")
    try:
        encoder_state = state_dict['encoder']
        missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)
        print(f"   ✅ Success! Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        encoder_loaded = True
    except Exception as e:
        print(f"   ❌ Failed: {e}")

# 尝试方式 2
if not encoder_loaded:
    print("🔍 Attempt 2: Direct loading")
    try:
        missing, unexpected = model.encoder.load_state_dict(state_dict, strict=False)
        print(f"   ✅ Success! Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        encoder_loaded = True
    except Exception as e:
        print(f"   ❌ Failed: {e}")

# 尝试方式 3
if not encoder_loaded and 'model' in state_dict:
    print("🔍 Attempt 3: checkpoint['model']")
    try:
        model_state = state_dict['model']
        if 'encoder' in model_state:
            encoder_state = model_state['encoder']
        else:
            encoder_state = model_state
        missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)
        print(f"   ✅ Success! Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        encoder_loaded = True
    except Exception as e:
        print(f"   ❌ Failed: {e}")

if not encoder_loaded:
    print("❌ ALL LOADING ATTEMPTS FAILED!")
    sys.exit(1)

print("\n" + "=" * 80)
print("🔍 Step 5: Verifying weights...")
print("=" * 80)

first_param = next(model.encoder.parameters())
print(f"\n📊 First encoder parameter:")
print(f"   Shape: {first_param.shape}")
print(f"   Sum: {first_param.sum().item():.6f}")
print(f"   Mean: {first_param.mean().item():.6f}")
print(f"   Std: {first_param.std().item():.6f}")

print("\n" + "=" * 80)
print("🎯 FINAL VERDICT:")
print("=" * 80)

if abs(first_param.sum().item()) < 1e-6 and abs(first_param.mean().item()) < 1e-6:
    print("❌ Encoder weights are ZEROS! Loading FAILED!")
else:
    print("✅ Encoder weights are NON-ZERO! Loading SUCCESS!")

print("=" * 80)
