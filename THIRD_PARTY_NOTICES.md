# 第三方来源与许可

MolMotion 原创代码按 Apache-2.0 提供。第三方文件保留原许可，不替换其版权声明。

## MDGen

来源 https://github.com/bjing2016/mdgen ，本地基准 commit `81482a403b91c5a8437046da1d8d321ba97089cc`。

`src/molmotion/_vendor/mdgen/geometry.py` 来自 MDGen，MIT 许可，版权 2024 Bowen Jing、Hannes Stärk、Tommi Jaakkola、Bonnie Berger。完整许可随包保存为 `LICENSE-MDGen`。

同目录 `rigid_utils.py`、`tensor_utils.py`、`residue_constants.py` 自带 Apache-2.0 文件头，版权 2021 AlQuraishi Laboratory 与 DeepMind Technologies Limited；完整文件头及许可保留。四个几何依赖文件在此版本未修改。`residue_constants` 内未使用的立体化学资源读取函数依赖 OpenFold 资源；当前推理链不调用该函数。

`research/` 中的完整训练脚本仍导入外部 MDGen 模型与 transport；轻量公开推理包仅使用上述内置几何子集。

## 数据与样例

数据源 https://huggingface.co/datasets/bjing-mit/tetrapeptide-sims 。2026-11-08 核对的 revision 为 `d42fdc6ff76a04cd8a9d4eaae08101da3149f99e`，数据卡声明 MIT。`examples/IMCP.pdb` 为单个初始结构；来源与 MIT 条款放在 examples 目录。没有打包整个数据集。

## 自训练权重

发布的是本项目训练所得主模型 85k 验证选中检查点，按 Apache-2.0 提供。权重中不包含 MDGen 预训练模型参数或优化器。许可不构成模型质量或物理有效性的担保；详见 MODEL_CARD.md。
