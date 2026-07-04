"""環境検証スクリプト。ローカルでも Kaggle でも実行可能。"""
import sys


def check_import(module_name):
    try:
        __import__(module_name)
        print(f"[OK] {module_name}")
        return True
    except Exception as e:
        print(f"[NG] {module_name}: {e}")
        return False


def main():
    print("SNNAI environment verification")
    print("=" * 40)

    modules = ["torch", "snntorch", "brian2", "norse", "bindsnet", "numpy", "matplotlib", "pytest"]
    results = {m: check_import(m) for m in modules}

    try:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"CUDA check failed: {e}")

    if all(results.values()):
        print("\nAll imports OK.")
        sys.exit(0)
    else:
        print("\nSome imports failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
