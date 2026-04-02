# 统计讲义智答

面向应用统计课堂展示的中文 RAG 项目，基于清华大学公开统计讲义构建，支持检索、重排、生成回答，以及简单的网页演示界面。

## 当前模型组合

- Generation: `Qwen/Qwen3-14B`
- Embedding: `Qwen/Qwen3-Embedding-4B`
- Rerank: `Qwen/Qwen3-Reranker-4B`

## 功能概览

- 读取中文统计讲义 PDF
- 自动清理重复页眉、页脚和页码
- 按页聚合并切分检索 chunk
- 使用 Chroma 构建向量库
- 通过 Embedding + Rerank 完成检索
- 使用大模型生成带来源的中文回答
- 提供一个适合课堂展示的 Web 界面
- Web 回答区支持 Markdown 和 LaTeX 公式渲染

## 目录说明

- `stats_rag.py`: 命令行版 RAG 主程序
- `stats_rag_web.py`: Web 演示界面
- `knowledge_base/tsinghua_stats`: 清华统计讲义 PDF
- `RAG_USAGE.md`: 更完整的使用说明
- `.env.example`: 环境变量模板

## 快速开始

1. 根据 `.env.example` 创建自己的 `.env`
2. 安装依赖并激活虚拟环境
3. 运行配置检查

```powershell
E:\RAG_projects\venv\Scripts\python.exe E:\RAG_projects\stats_rag.py check
```

4. 构建知识库

```powershell
E:\RAG_projects\venv\Scripts\python.exe E:\RAG_projects\stats_rag.py build --force
```

5. 启动网页界面

```powershell
E:\RAG_projects\venv\Scripts\python.exe E:\RAG_projects\stats_rag_web.py --host 127.0.0.1 --port 7860
```

浏览器打开 `http://127.0.0.1:7860`

## 说明

- `.env`、`venv`、本地模型、向量库和日志默认不会提交到 GitHub。
- 如果需要离线重建，重新执行 `build --force` 即可生成向量库。
- 更详细的参数和运行方式见 `RAG_USAGE.md`。
