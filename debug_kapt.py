"""诊断 KAPT 模块导入问题"""

import sys
import inspect

# 1. 检查 kapt_modules 的导入路径
try:
    from chemprop.models.kapt_modules import StructureAwarePromptGenerator
    print(f"✅ 成功导入 StructureAwarePromptGenerator")
    print(f"📍 模块路径: {inspect.getfile(StructureAwarePromptGenerator)}")
    print(f"📋 __init__ 签名:\n{inspect.signature(StructureAwarePromptGenerator.__init__)}")
except Exception as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# 2. 检查所有 KAPT 相关模块
print("\n" + "="*80)
print("🔍 检查已加载的 KAPT 相关模块:")
for name, module in sys.modules.items():
    if 'kapt' in name.lower():
        try:
            print(f"  - {name}: {module.__file__ if hasattr(module, '__file__') else 'built-in'}")
        except:
            print(f"  - {name}: (无法获取路径)")

# 3. 检查类的参数
print("\n" + "="*80)
print("🔍 StructureAwarePromptGenerator.__init__ 的参数:")
sig = inspect.signature(StructureAwarePromptGenerator.__init__)
for param_name, param in sig.parameters.items():
    if param_name != 'self':
        print(f"  - {param_name}: {param.default if param.default != inspect.Parameter.empty else '(必需)'}")

# 4. 测试创建实例
print("\n" + "="*80)
print("🧪 测试创建实例:")
try:
    sapg = StructureAwarePromptGenerator(
        node_dim=300,
        prompt_dim=128,
        num_patterns=5,
        dropout=0.0
    )
    print("✅ 成功创建实例（使用 num_patterns）")
except TypeError as e:
    print(f"❌ 失败: {e}")
    print("\n🔄 尝试不使用 num_patterns:")
    try:
        sapg = StructureAwarePromptGenerator(
            node_dim=300,
            prompt_dim=128,
            dropout=0.0
        )
        print("✅ 成功创建实例（不使用 num_patterns）")
    except Exception as e2:
        print(f"❌ 仍然失败: {e2}")
