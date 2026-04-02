# 应用统计中文 RAG 使用说明

本项目当前使用以下模型组合：

- Embedding: `Qwen/Qwen3-Embedding-4B`
- Rerank: `Qwen/Qwen3-Reranker-4B`
- Generation: `Qwen/Qwen3-14B`

知识库来源：

- `E:\RAG_projects\knowledge_base\tsinghua_stats`

主脚本：

- `E:\RAG_projects\stats_rag.py`
- `E:\RAG_projects\stats_rag_web.py`

## 已实现能力

- 读取清华统计讲义 PDF
- 自动清理重复页眉页脚和页码
- 按页聚合切分为检索 chunk
- 调用硅基流动 Embedding 建立 Chroma 向量库
- 调用硅基流动 Rerank 重排检索结果
- 调用硅基流动回答模型生成中文答案

## 当前默认参数

- `CHUNK_TARGET_CHARS=550`
- `CHUNK_OVERLAP_PAGES=1`
- `MIN_CHUNK_CHARS=120`
- `RETRIEVAL_TOP_K=12`
- `RERANK_TOP_N=5`
- `EMBEDDING_DIMENSIONS=1024`

这些参数即使不写入 `.env`，脚本也会使用默认值。

## 运行方式

使用项目虚拟环境：

```powershell
E:\RAG_projects\venv\Scripts\python.exe E:\RAG_projects\stats_rag.py check
```

构建知识库：

```powershell
E:\RAG_projects\venv\Scripts\python.exe E:\RAG_projects\stats_rag.py build --force
```

只看检索结果：

```powershell
E:\RAG_projects\venv\Scripts\python.exe E:\RAG_projects\stats_rag.py search "什么情况下使用方差分析？"
```

完整问答：

```powershell
E:\RAG_projects\venv\Scripts\python.exe E:\RAG_projects\stats_rag.py ask "什么情况下使用方差分析？"
```

交互式问答：

```powershell
E:\RAG_projects\venv\Scripts\python.exe E:\RAG_projects\stats_rag.py shell
```

启动网页界面：

```powershell
E:\RAG_projects\venv\Scripts\python.exe E:\RAG_projects\stats_rag_web.py --host 127.0.0.1 --port 7860
```

启动后在浏览器打开：

```text
http://127.0.0.1:7860
```

## 结果文件

- 向量库目录：`E:\RAG_projects\rag_store\stats_chroma`
- 构建清单：`E:\RAG_projects\rag_store\stats_manifest.json`

## 推荐演示问题

- `什么情况下使用方差分析？`
- `回归分析和方差分析有什么联系？`
- `什么是广义线性模型，它和线性回归有什么区别？`
- `什么情况下适合使用泊松回归？`
- `自助法为什么能用来估计统计量的不确定性？`

## 可选环境变量

如果你想微调第一版效果，可以在 `.env` 中追加：

```env
EMBEDDING_DIMENSIONS=1024
RETRIEVAL_TOP_K=12
RERANK_TOP_N=5
CHUNK_TARGET_CHARS=550
CHUNK_OVERLAP_PAGES=1
MIN_CHUNK_CHARS=120
GENERATION_ENABLE_THINKING=false
EMBEDDING_QUERY_INSTRUCTION=Retrieve passages from a statistics course knowledge base that best answer the user's question.
RERANK_INSTRUCTION=Please rerank the documents based on how well they answer the query about statistics concepts, methods, assumptions, and interpretation.
```

## 备注

- 代码里已经显式禁用了系统代理继承，避免 TUN/代理模式影响硅基流动 API 调用。
- 当前知识库为课堂展示型第一版，后续可以继续加入北大讲义或国家统计局材料。
- 网页界面中的回答区域会把 Markdown 渲染成 HTML，并把 LaTeX 公式在服务端转换为 MathML，默认不再依赖 MathJax CDN。
- 来源片段仍然是 PDF 文本抽取结果，复杂公式不保证排版完全正确；如需看最准确公式，请点击来源里的“打开原 PDF”。
- 如果某个非常复杂的 LaTeX 公式无法转换，页面会回退显示原始公式文本，你仍然可以通过原 PDF 核对。
