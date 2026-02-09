# ===== ultimate_diagnose.py =====
"""
不依赖任何 chemprop 模块的诊断脚本
直接分析源代码找出问题
"""

import ast
import os


def find_len_assignments_in_all_files():
    """遍历所有 Python 文件，找出对 len 的赋值"""

    print("\n" + "=" * 70)
    print("🔍 扫描所有 chemprop Python 文件，找出对 'len' 的赋值")
    print("=" * 70 + "\n")

    findings = []

    for root, dirs, files in os.walk('chemprop'):
        # 跳过 __pycache__
        dirs[:] = [d for d in dirs if d != '__pycache__']

        for file in files:
            if not file.endswith('.py'):
                continue

            filepath = os.path.join(root, file)

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    source = f.read()

                tree = ast.parse(source)

                # 查找所有赋值
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id == 'len':
                                line_no = node.lineno
                                findings.append({
                                    'file': filepath,
                                    'line': line_no,
                                    'value_type': type(node.value).__name__
                                })
            except:
                pass

    if findings:
        print(f"⚠️ 找到 {len(findings)} 处对 'len' 的赋值:\n")
        for f in findings:
            print(f"  📍 {f['file']}:{f['line']}")
            print(f"     值的类型: {f['value_type']}\n")
        return findings
    else:
        print("✅ 没有找到对 'len' 的显式赋值\n")
        return []


def check_model_py_diagnostic_code():
    """检查 model.py 中的诊断代码"""

    print("\n" + "=" * 70)
    print("🔍 检查 model.py 中的诊断代码")
    print("=" * 70 + "\n")

    try:
        with open('chemprop/models/model.py', 'r') as f:
            lines = f.readlines()

        # 查找可疑的代码行
        suspicious_patterns = [
            'len', 'RuntimeError', 'CRITICAL', 'ENVIRONMENT'
        ]

        for i, line in enumerate(lines[:50], 1):
            for pattern in suspicious_patterns:
                if pattern in line:
                    print(f"  {i:3d}: {line.rstrip()}")
                    break
    except Exception as e:
        print(f"❌ 错误: {e}")


def main():
    print("\n\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "🔎 终极诊断：找出 len 被污染的真正原因" + " " * 20 + "║")
    print("╚" + "=" * 68 + "╝")

    findings = find_len_assignments_in_all_files()
    check_model_py_diagnostic_code()

    print("\n" + "=" * 70)
    print("📋 诊断建议：")
    print("=" * 70)
    if findings:
        print("\n❌ 发现了对 'len' 的赋值！必须修复：")
        for f in findings:
            print(f"\n  1. 打开文件: {f['file']}")
            print(f"  2. 跳转到第 {f['line']} 行")
            print(f"  3. 删除或重命名 'len =' 的赋值")
    else:
        print("\n✅ 源代码本身没有问题")
        print("⚠️ 问题可能在模块初始化或你添加的诊断代码中")
        print("\n请检查：")
        print("  1. chemprop/models/model.py 第 20-40 行")
        print("  2. 是否有自己添加的诊断/检查代码")
        print("  3. 注释掉这些代码后重试")


if __name__ == '__main__':
    main()
