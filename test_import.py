"""测试实际调用时的参数传递"""

import sys
sys.path.insert(0, '.')

from chemprop.models.kapt_modules import StructureAwarePromptGenerator
import inspect

# 1. 检查类签名
sig = inspect.signature(StructureAwarePromptGenerator.__init__)
print("📋 StructureAwarePromptGenerator.__init__ 签名:")
print(f"   {sig}")

# 2. 测试正确调用
print("\n🧪 测试 1: 使用 num_patterns 参数")
try:
    sapg = StructureAwarePromptGenerator(
        node_dim=300,
        prompt_dim=128,
        num_patterns=5,
        dropout=0.0
    )
    print("   ✅ 成功")
except Exception as e:
    print(f"   ❌ 失败: {e}")

# 3. 测试错误调用（看是否会触发同样的错误）
print("\n🧪 测试 2: 使用不存在的参数")
try:
    sapg = StructureAwarePromptGenerator(
        node_dim=300,
        prompt_dim=128,
        wrong_param=5,  # 故意使用错误参数
        dropout=0.0
    )
    print("   ✅ 成功（不应该）")
except TypeError as e:
    print(f"   ❌ 预期的错误: {e}")

# 4. 模拟 model.py 中的调用方式
print("\n🧪 测试 3: 模拟 add_kapt_prompt() 中的调用")
hidden_dim = 300
prompt_dim = 128

class Args:
    num_struct_patterns = 5
    dropout = 0.0

args = Args()

try:
    sapg = StructureAwarePromptGenerator(
        node_dim=hidden_dim,
        prompt_dim=prompt_dim,
        num_patterns=getattr(args, 'num_struct_patterns', 5),
        dropout=getattr(args, 'dropout', 0.0)
    )
    print("   ✅ 成功")
except Exception as e:
    print(f"   ❌ 失败: {e}")
