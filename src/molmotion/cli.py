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
    generate=commands.add_parser("generate",help="从四肽 PDB 生成轨迹")
    generate.add_argument("pdb")
    generate.add_argument("--model",default="tetrapeptide-v0.1.0")
    generate.add_argument("--device",default="cpu")
    generate.add_argument("--samples",type=int,default=1)
    generate.add_argument("--seed",type=int,default=42)
    generate.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    if args.command == "generate":
        from .pipeline import MolMotionPipeline
        pipe=MolMotionPipeline.from_pretrained(args.model,device=args.device)
        result=pipe.generate(args.pdb,num_samples=args.samples,seed=args.seed)
        args.output_dir.mkdir(parents=True,exist_ok=True)
        result.save_npz(args.output_dir/"trajectories.npz")
        for i in range(args.samples):
            result.save_pdb(args.output_dir/f"sample-{i}.pdb",sample=i)
            result.save_xtc(args.output_dir/f"sample-{i}.xtc",sample=i)
        print(json.dumps({"bond_valid":result.bond_validity(),"samples":args.samples,"frames_including_initial":result.coordinates.shape[1]}))
        return
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
