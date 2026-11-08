# 如何把 MolMotion 发布成别人能调用的工具

## 当前已经能做什么

别人可以克隆仓库后 `python -m pip install .`，调用 `molmotion.evaluate_npz()` 或 `molmotion evaluate`。模型类与低层采样器需要 `.[model]`；没有训练权重时不能生成有科学意义的结果。当前版本尚未发布 PyPI。

## 第一阶段：发布可复现的模型示例

先固定一个四肽检查点，与其配套发布模型结构配置、均值/尺度/残差尺度、原子与残基编码、几何表示版本、数据来源、采样步数和 SHA256。权重单独存放，不塞进 wheel。

接通以下路径：输入 PDB 和序列 → 检查四肽及拓扑 → 几何编码和标准化 → 已训练模型采样 → 反标准化与几何解码 → PDB/XTC 输出 → 与固定参考数据评价。确认 Å 与 nm、ps 与帧间隔的转换。

`from_pretrained`、`generate` 高层接口、`export` 和 `propose_seeds` 是待实现的接口，不应把设计示例写成已可调用的教程。

## 第二阶段：先构建包，再测试安装

从仓库根目录运行：

```bash
python -m pip install '.[model]' build twine
python -m unittest discover -s tests -v
python examples/check_install.py
python -m build
python -m twine check dist/*
```

在新的虚拟环境中安装生成的 wheel，再运行安装检查与真实模型示例。发布之前固定许可证和包元数据；建议为测试、文档与 GPU 推理分别记录依赖。不要将训练数据、账号凭据和检查点放入分发包。

## 第三阶段：TestPyPI 与 PyPI

以下是**维护者将来执行的发布步骤**，本仓库尚未执行上传。

1. 创建 TestPyPI 与 PyPI 账号，启用所需的账号验证和双因素认证。
2. 确认 `molmotion-research` 名称可用；目前只使用了本地分发名称，没有占用 PyPI 名称。
3. 将 `0.1.0.dev0` 更新为经过测试的发布版本，同时更新 `pyproject.toml` 与包内版本。
4. 清理本地旧构建产物，重新构建并检查，先上传 TestPyPI：

```bash
python -m twine upload --repository testpypi dist/*
```

5. 从 TestPyPI 获取指定版本 wheel，在干净环境中验证；依赖包可以预先从常规 PyPI 安装。测试通过后再向正式 PyPI 发布同一审核过的版本。
6. 后续可使用 GitHub Actions + PyPI Trusted Publishing。需要在 PyPI 绑定仓库所有者 `Leo-di`、仓库 `MolMotion`、发布工作流文件名及环境；这些设置不会仅因 README 写出而生效。

本次未添加会自动发布包的工作流，也未配置发布账号。初始发布先保证可用性，之后再自动化。

## 第四阶段：模型权重与 GitHub Release

许可明确后，权重可独立托管于模型平台并附模型卡：四肽适用范围、训练与验证数据、输入输出单位、硬件需求、结构和动态指标、当前失败情形。设置权重格式、版本、校验和与代码兼容版本。使用安全序列化格式时，要同时验证转换后的数值一致性。

GitHub Release 应关联固定 tag，并提供变更记录、安装说明和通过测试的 wheel。普通源码推送、GitHub Release、PyPI 发布与权重托管是四个独立动作。

## 发布前的科学表述

当前单尺度 baseline 在键长有效率上领先；模型起点在等仿真步数下有覆盖收益。不要承诺“全面超过 baseline”“任意分子适用”或“已实现端到端加速”。将真实效果边界写在模型卡与首页，使用者才知道如何选用。

## 官方参考

- [Python Packaging：项目构建与发布](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
