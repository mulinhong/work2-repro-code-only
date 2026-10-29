# 第三方来源与分发范围

- STreeD / pystreed 1.4.0：官方包采用 MIT，保留通知于 `licenses/STreeD-MIT.txt`。源码获取：https://pypi.org/project/pystreed/1.4.0/ 。官方 sdist SHA256 为 `63ff09838c5838cff30bbbceca56994417b8b0c05f9b082a666ba7e8a632e113`。本仓库只保留 Work2 自定义任务、实际 O3 入口和原始集成差异，不内置完整上游项目。
- PyDL8.5 0.1.8：独立比较依赖，官方项目为 https://github.com/aia-uclouvain/pydl8.5 ，历史固定提交为 `a7f0fee72b4c17bbcb727050ba102db7dda10119`。包信息采用 MIT：https://pypi.org/project/pydl8.5/0.1.8/ 。不将该算法归为 VEST-2。
- Qwen3-Embedding-4B：官方模型卡标注 Apache-2.0：https://huggingface.co/Qwen/Qwen3-Embedding-4B 。仅声明获取来源，不分发权重。
- CodeSearchNet：工具仓库 MIT 通知保存在 `licenses/CodeSearchNet-tooling-MIT.txt`，来源 https://github.com/github/CodeSearchNet 。工具仓库许可证不自动替代每个原始代码项目的授权；服务器语料、对应版本和行号映射沿用既有冻结材料，不在本仓库重新分发。
- Work2 作者源码：本次私有归档不替作者新增开源授权；公开发布时由作者决定授权与分发范围。
