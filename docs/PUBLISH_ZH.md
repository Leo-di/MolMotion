# MolMotion v0.1.0 发布维护指南

## 固定发布内容

代码包 `molmotion-research`，版本 `0.1.0`；Python 导入名 `molmotion`。研究模型使用 seed43、85k 验证选中权重，safetensors 包含参数与三项归一化缓冲区。权重、配置、模型卡单独作为 GitHub Release 资产；wheel 不含权重。

## 本地验证

```bash
python -m pip install '.[model,release]'
python -m unittest discover -s tests -v
python examples/check_install.py
python -m build
python -m twine check dist/*
```

还必须用真实权重跑完 256 帧，验证输出有限、初始帧正确、原子拓扑和坐标/时间单位一致，并记录实际硬件、耗时与 SHA256。这是功能验证，不是方法优越性证明。`docs/release-validation.json` 保存发布测试结果。

## PyPI Trusted Publishing

PyPI pending publisher 使用：
- 项目名：`molmotion-research`
- GitHub owner：`Leo-di`
- 仓库：`MolMotion`
- 工作流：`publish.yml`
- GitHub environment：`pypi`

工作流仅手动触发，先测试并构建，再通过 OIDC 上传；不使用 PyPI 密码或长期 API Token。修改这五个字段必须同步 PyPI 侧配置。首次发布成功后 pending publisher 转为项目 publisher。

## 版本发布步骤

1. 固定源码、权重、数据与第三方许可，更新 README、模型卡和验证记录。
2. 构建 wheel/sdist，在干净环境安装并验证。
3. 推送固定 commit，创建版本 tag 和 GitHub Release，上传同版本权重、配置和代码分发文件。
4. 手动触发 Publish to PyPI；检查工作流成功和 PyPI 公开版本。
5. 从公开 PyPI 安装，重新执行真实样例；核对发布资产校验和。

PyPI 不允许用同一版本号覆盖已上传文件；后续修复必须递增版本，不能删除重传来伪装原版本。具体平台账号和认证凭据不进入仓库。

官方参考：[项目构建与发布](https://packaging.python.org/en/latest/tutorials/packaging-projects/)、[Trusted Publishing](https://docs.pypi.org/trusted-publishers/)。
