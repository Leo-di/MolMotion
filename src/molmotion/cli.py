import argparse
import json
from pathlib import Path
from .api import evaluate_npz


def main():
    parser=argparse.ArgumentParser(description="MolMotion 四肽轨迹评估")
    commands=parser.add_subparsers(dest="command",required=True)
    evaluate=commands.add_parser("evaluate",help="评估含原子坐标的 NPZ")
    evaluate.add_argument("input")
    evaluate.add_argument("--output",type=Path)
    args=parser.parse_args()
    try:
        result=evaluate_npz(args.input)
    except (ValueError,OSError) as exc:
        parser.error(str(exc))
    text=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text+"\n",encoding="utf-8")
    else:
        print(text)

if __name__ == "__main__":
    main()
