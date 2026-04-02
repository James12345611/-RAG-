from __future__ import annotations

import argparse
import asyncio
import html
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

from aiohttp import web
from latex2mathml.converter import convert as latex_to_mathml
from markdown_it import MarkdownIt

from stats_rag import AppConfig, Generator, get_collection, search


ROOT = Path(__file__).resolve().parent
MD = MarkdownIt("commonmark", {"breaks": True, "html": False})
DISPLAY_MATH_PATTERNS = [
    re.compile(r"(?<!\\)\$\$(.+?)(?<!\\)\$\$", re.S),
    re.compile(r"\\\[(.+?)\\\]", re.S),
]
INLINE_MATH_PATTERNS = [
    re.compile(r"\\\((.+?)\\\)"),
    re.compile(r"(?<!\\)(?<!\$)\$(?!\$)([^\n$]+?)(?<!\\)\$(?!\$)"),
]

EXAMPLE_QUESTIONS = [
    "什么情况下使用方差分析？",
    "回归分析和方差分析有什么联系？",
    "什么是广义线性模型，它和线性回归有什么区别？",
    "什么时候适合使用泊松回归？",
    "自助法为什么能用来估计统计量的不确定性？",
]


def load_manifest(config: AppConfig) -> dict[str, Any]:
    if not config.manifest_path.exists():
        return {}
    try:
        return json.loads(config.manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def build_page(config: AppConfig) -> str:
    manifest = load_manifest(config)
    chunk_count = manifest.get("chunk_count", "未知")
    pdf_count = manifest.get("pdf_count", "未知")
    example_buttons = "\n".join(
        f'<button class="example-chip" data-question="{html.escape(question, quote=True)}">{html.escape(question)}</button>'
        for question in EXAMPLE_QUESTIONS
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>统计学 RAG 演示</title>
  <style>
    :root {{
      --panel: rgba(255, 251, 245, 0.92);
      --panel-strong: #fffaf2;
      --line: rgba(104, 79, 45, 0.14);
      --text: #30261b;
      --muted: #6a5a46;
      --accent: #0e6b63;
      --accent-soft: #d7efe8;
      --accent-strong: #124f58;
      --shadow: 0 24px 80px rgba(60, 45, 28, 0.12);
      --radius-xl: 28px;
      --radius-lg: 20px;
      --radius-md: 14px;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(255, 218, 167, 0.45), transparent 32%),
        radial-gradient(circle at bottom right, rgba(123, 201, 189, 0.32), transparent 28%),
        linear-gradient(135deg, #f7efe2 0%, #efe6d8 42%, #f8f5ef 100%);
      min-height: 100vh;
    }}

    .shell {{
      width: min(1200px, calc(100vw - 32px));
      margin: 24px auto;
      display: grid;
      grid-template-columns: 380px 1fr;
      gap: 20px;
      align-items: start;
    }}

    .card {{
      background: var(--panel);
      backdrop-filter: blur(18px);
      border: 1px solid var(--line);
      border-radius: var(--radius-xl);
      box-shadow: var(--shadow);
      overflow: hidden;
    }}

    .sidebar {{
      padding: 28px;
      position: sticky;
      top: 20px;
    }}

    .eyebrow {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      font-size: 13px;
      background: var(--accent-soft);
      color: var(--accent-strong);
      margin-bottom: 16px;
    }}

    h1 {{
      font-size: clamp(28px, 4vw, 42px);
      line-height: 1.08;
      margin: 0 0 12px;
      letter-spacing: -0.03em;
    }}

    .subtle {{
      color: var(--muted);
      line-height: 1.7;
      font-size: 15px;
      margin: 0;
    }}

    .stats {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 12px;
      margin: 24px 0 28px;
    }}

    .stat {{
      padding: 14px 16px;
      border-radius: var(--radius-md);
      background: rgba(255, 255, 255, 0.68);
      border: 1px solid rgba(104, 79, 45, 0.1);
    }}

    .stat-label {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }}

    .stat-value {{
      font-size: 20px;
      font-weight: 700;
    }}

    .section-title {{
      font-size: 13px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
      margin: 0 0 12px;
    }}

    .examples {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}

    .example-chip {{
      border: none;
      padding: 10px 14px;
      border-radius: 999px;
      background: #fff;
      color: var(--text);
      cursor: pointer;
      border: 1px solid rgba(104, 79, 45, 0.12);
      transition: transform 160ms ease, background 160ms ease, border-color 160ms ease;
    }}

    .example-chip:hover {{
      transform: translateY(-1px);
      background: #fff8ed;
      border-color: rgba(14, 107, 99, 0.24);
    }}

    .main {{
      padding: 24px;
      display: grid;
      gap: 18px;
    }}

    .composer {{
      padding: 22px;
      background: var(--panel-strong);
      border-radius: var(--radius-lg);
      border: 1px solid rgba(104, 79, 45, 0.08);
      display: grid;
      gap: 14px;
    }}

    textarea {{
      width: 100%;
      min-height: 136px;
      resize: vertical;
      padding: 18px 18px 16px;
      border-radius: 18px;
      border: 1px solid rgba(104, 79, 45, 0.12);
      font: inherit;
      line-height: 1.7;
      color: var(--text);
      background: #fff;
      outline: none;
      transition: border-color 160ms ease, box-shadow 160ms ease;
    }}

    textarea:focus {{
      border-color: rgba(14, 107, 99, 0.52);
      box-shadow: 0 0 0 4px rgba(14, 107, 99, 0.12);
    }}

    .actions {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }}

    .hint {{
      color: var(--muted);
      font-size: 13px;
    }}

    .primary {{
      border: none;
      background: linear-gradient(135deg, #0e6b63 0%, #17415f 100%);
      color: white;
      padding: 12px 20px;
      border-radius: 999px;
      font-size: 15px;
      font-weight: 600;
      cursor: pointer;
      transition: transform 160ms ease, opacity 160ms ease;
    }}

    .primary:hover {{
      transform: translateY(-1px);
    }}

    .primary:disabled {{
      opacity: 0.6;
      cursor: wait;
      transform: none;
    }}

    .status {{
      min-height: 24px;
      color: var(--muted);
      font-size: 14px;
    }}

    .status.loading {{
      color: var(--accent-strong);
    }}

    .status.error {{
      color: #9d2a2a;
    }}

    .result-panel {{
      background: var(--panel-strong);
      border-radius: var(--radius-lg);
      border: 1px solid rgba(104, 79, 45, 0.08);
      padding: 22px;
      display: grid;
      gap: 18px;
      min-height: 420px;
    }}

    .result-section {{
      padding: 18px;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.78);
      border: 1px solid rgba(104, 79, 45, 0.08);
    }}

    .result-section h2 {{
      font-size: 16px;
      margin: 0 0 12px;
    }}

    .answer {{
      line-height: 1.8;
      font-size: 15px;
    }}

    .answer > :first-child {{
      margin-top: 0;
    }}

    .answer > :last-child {{
      margin-bottom: 0;
    }}

    .answer p,
    .answer ul,
    .answer ol,
    .answer blockquote {{
      margin: 0 0 14px;
    }}

    .answer h1,
    .answer h2,
    .answer h3,
    .answer h4 {{
      margin: 22px 0 12px;
      line-height: 1.35;
    }}

    .answer h1 {{
      font-size: 26px;
    }}

    .answer h2 {{
      font-size: 22px;
    }}

    .answer h3 {{
      font-size: 18px;
    }}

    .answer h4 {{
      font-size: 16px;
    }}

    .answer ul,
    .answer ol {{
      padding-left: 22px;
    }}

    .answer li + li {{
      margin-top: 6px;
    }}

    .answer code {{
      font-family: "Consolas", "SFMono-Regular", monospace;
      font-size: 0.94em;
      background: rgba(14, 107, 99, 0.08);
      padding: 2px 6px;
      border-radius: 6px;
    }}

    .answer pre {{
      margin: 0 0 14px;
      padding: 14px 16px;
      border-radius: 14px;
      background: #f7f2e8;
      overflow-x: auto;
    }}

    .answer pre code {{
      background: transparent;
      padding: 0;
    }}

    .math-block {{
      display: block;
      overflow-x: auto;
      margin: 16px 0;
      padding: 10px 14px;
      border-radius: 14px;
      background: #fcf8ef;
      border: 1px solid rgba(104, 79, 45, 0.08);
    }}

    .math-inline {{
      display: inline-block;
      vertical-align: middle;
      margin: 0 2px;
    }}

    .math-block math,
    .math-inline math {{
      font-size: 1.02em;
    }}

    .math-fallback code {{
      white-space: pre-wrap;
    }}

    .sources {{
      display: grid;
      gap: 12px;
    }}

    .source-item {{
      border-radius: 16px;
      padding: 14px 16px;
      background: #fff;
      border: 1px solid rgba(104, 79, 45, 0.08);
    }}

    .source-meta {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
    }}

    .source-snippet {{
      color: var(--text);
      line-height: 1.75;
      font-size: 14px;
      white-space: pre-wrap;
    }}

    .source-link {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      margin-top: 10px;
      color: var(--accent-strong);
      font-size: 13px;
      text-decoration: none;
      font-weight: 600;
    }}

    .source-link:hover {{
      text-decoration: underline;
    }}

    .empty {{
      display: grid;
      place-items: center;
      min-height: 280px;
      text-align: center;
      color: var(--muted);
      border: 1px dashed rgba(104, 79, 45, 0.16);
      border-radius: 18px;
      padding: 24px;
      background: rgba(255, 255, 255, 0.45);
    }}

    @media (max-width: 980px) {{
      .shell {{
        grid-template-columns: 1fr;
      }}

      .sidebar {{
        position: static;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <aside class="card sidebar">
      <div class="eyebrow">应用统计 · 中文 RAG</div>
      <h1>统计学课堂展示助手</h1>
      <p class="subtle">
        基于清华大学公开统计讲义构建，支持统计概念解释、方法选型、检验思路和结果理解。
      </p>
      <div class="stats">
        <div class="stat">
          <span class="stat-label">PDF 数量</span>
          <span class="stat-value">{pdf_count}</span>
        </div>
        <div class="stat">
          <span class="stat-label">Chunk 数量</span>
          <span class="stat-value">{chunk_count}</span>
        </div>
      </div>
      <p class="section-title">示例问题</p>
      <div class="examples">
        {example_buttons}
      </div>
    </aside>

    <main class="card main">
      <section class="composer">
        <textarea id="question" placeholder="例如：什么情况下使用方差分析？&#10;或者：广义线性模型和线性回归有什么区别？"></textarea>
        <div class="actions">
          <div class="hint">回答会先检索讲义，再给出带来源的中文解释。按 Ctrl+Enter 也可以提交。</div>
          <button id="askBtn" class="primary">开始提问</button>
        </div>
        <div id="status" class="status"></div>
      </section>

      <section class="result-panel">
        <div class="result-section">
          <h2>回答</h2>
          <p class="subtle" style="margin: 0 0 12px; font-size: 13px;">
            回答中的 Markdown 会转成网页排版，LaTeX 公式会在服务端直接渲染为数学表达式。
          </p>
          <div id="answer" class="answer empty">输入一个统计学问题，我们会先检索讲义，再生成适合课堂展示的中文解释。</div>
        </div>

        <div class="result-section">
          <h2>引用来源</h2>
          <p class="subtle" style="margin: 0 0 12px; font-size: 13px;">
            这里展示的是 PDF 文本抽取结果。复杂公式若想看准确排版，请点击“打开原 PDF”。
          </p>
          <div id="sources" class="sources">
            <div class="empty">这里会展示回答所依据的讲义页码和相关片段。</div>
          </div>
        </div>
      </section>
    </main>
  </div>

  <script>
    const questionInput = document.getElementById("question");
    const askBtn = document.getElementById("askBtn");
    const answerEl = document.getElementById("answer");
    const sourcesEl = document.getElementById("sources");
    const statusEl = document.getElementById("status");

    function setStatus(message, kind = "") {{
      statusEl.textContent = message;
      statusEl.className = kind ? "status " + kind : "status";
    }}

    function escapeHtml(value) {{
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }}

    function renderAnswer(answerHtml) {{
      answerEl.className = "answer";
      answerEl.innerHTML = answerHtml;
    }}

    function renderSources(sources) {{
      if (!sources || !sources.length) {{
        sourcesEl.innerHTML = '<div class="empty">没有可展示的来源。</div>';
        return;
      }}

      sourcesEl.innerHTML = sources.map((source, index) => {{
        return '<article class="source-item">' +
          '<div class="source-meta">[' + (index + 1) + '] ' + escapeHtml(source.source) +
          ' · 第 ' + escapeHtml(source.page_start) + '-' + escapeHtml(source.page_end) +
          ' 页 · rerank=' + escapeHtml(source.relevance_score) + '</div>' +
          '<div class="source-snippet">' + escapeHtml(source.snippet) + '</div>' +
          '<a class="source-link" href="' + escapeHtml(source.pdf_url) + '" target="_blank" rel="noopener noreferrer">打开原 PDF</a>' +
          '</article>';
      }}).join("");
    }}

    async function askQuestion() {{
      const question = questionInput.value.trim();
      if (!question) {{
        setStatus("请先输入一个问题。", "error");
        questionInput.focus();
        return;
      }}

      askBtn.disabled = true;
      setStatus("正在检索讲义并生成回答，请稍候...", "loading");
      answerEl.className = "answer";
      answerEl.textContent = "";

      try {{
        const response = await fetch("/api/ask", {{
          method: "POST",
          headers: {{
            "Content-Type": "application/json"
          }},
          body: JSON.stringify({{ question }})
        }});

        const data = await response.json();
        if (!response.ok) {{
          throw new Error(data.error || "请求失败");
        }}

        renderAnswer(data.answer_html || escapeHtml(data.answer).replaceAll("\\n", "<br />"));
        renderSources(data.sources);
        setStatus("回答已生成。");
      }} catch (error) {{
        answerEl.className = "answer empty";
        answerEl.textContent = "当前请求失败，请检查服务日志或稍后重试。";
        sourcesEl.innerHTML = '<div class="empty">来源暂不可用。</div>';
        setStatus(error.message || "请求失败", "error");
      }} finally {{
        askBtn.disabled = false;
      }}
    }}

    askBtn.addEventListener("click", askQuestion);
    questionInput.addEventListener("keydown", (event) => {{
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {{
        askQuestion();
      }}
    }});

    document.querySelectorAll(".example-chip").forEach((button) => {{
      button.addEventListener("click", () => {{
        questionInput.value = button.dataset.question;
        questionInput.focus();
      }});
    }});
  </script>
</body>
</html>"""


def build_sources(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in contexts:
        metadata = item["metadata"]
        snippet = item["text"]
        if len(snippet) > 340:
            snippet = snippet[:340].rstrip() + "..."
        items.append(
            {
                "source": metadata["source"],
                "page_start": metadata["page_start"],
                "page_end": metadata["page_end"],
                "relevance_score": f"{item.get('relevance_score', 0):.4f}",
                "snippet": snippet,
                "pdf_url": f"/kb/{quote(metadata['source'])}#page={metadata['page_start']}",
            }
        )
    return items


def render_formula_html(latex: str, *, display: bool) -> str:
    expression = latex.strip()
    if not expression:
        return ""

    wrapper = "div" if display else "span"
    css_class = "math-block" if display else "math-inline"

    try:
        mathml = latex_to_mathml(expression)
    except Exception:
        delimiter = "$$" if display else "$"
        fallback = html.escape(f"{delimiter}{expression}{delimiter}")
        return f'<{wrapper} class="{css_class} math-fallback"><code>{fallback}</code></{wrapper}>'

    if display:
        mathml = mathml.replace(' display="inline"', ' display="block"', 1)

    return (
        f'<{wrapper} class="{css_class}" data-tex="{html.escape(expression, quote=True)}">'
        f"{mathml}"
        f"</{wrapper}>"
    )


def render_answer_html(answer: str) -> str:
    placeholder_map: dict[str, str] = {}
    processed = answer.replace("\r\n", "\n").strip()

    def reserve_formula(latex: str, *, display: bool) -> str:
        token = f"RAGMATH{'BLOCK' if display else 'INLINE'}TOKEN{len(placeholder_map)}"
        placeholder_map[token] = render_formula_html(latex, display=display)
        if display:
            return f"\n\n{token}\n\n"
        return token

    for pattern in DISPLAY_MATH_PATTERNS:
        processed = pattern.sub(lambda match: reserve_formula(match.group(1), display=True), processed)

    for pattern in INLINE_MATH_PATTERNS:
        processed = pattern.sub(lambda match: reserve_formula(match.group(1), display=False), processed)

    rendered = MD.render(processed)
    for token, formula_html in placeholder_map.items():
        rendered = rendered.replace(f"<p>{token}</p>\n", formula_html + "\n")
        rendered = rendered.replace(f"<p>{token}</p>", formula_html)
        rendered = rendered.replace(token, formula_html)
    return rendered


def answer_question(config: AppConfig, question: str) -> dict[str, Any]:
    contexts = search(config, question)
    generator = Generator(config)
    answer = generator.answer(question, contexts).strip()
    return {
        "answer": answer,
        "answer_html": render_answer_html(answer),
        "sources": build_sources(contexts),
    }


async def index(request: web.Request) -> web.Response:
    config: AppConfig = request.app["config"]
    return web.Response(text=build_page(config), content_type="text/html")


async def ask_api(request: web.Request) -> web.Response:
    config: AppConfig = request.app["config"]
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "请求体不是合法 JSON。"}, status=400)

    question = str(payload.get("question", "")).strip()
    if not question:
        return web.json_response({"error": "question 不能为空。"}, status=400)

    try:
        result = await asyncio.to_thread(answer_question, config, question)
        return web.json_response(result)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


async def health_api(request: web.Request) -> web.Response:
    config: AppConfig = request.app["config"]
    collection = get_collection(config)
    return web.json_response(
        {
            "status": "ok",
            "collection_name": collection.name,
            "chunk_count": collection.count(),
            "generation_model": config.generation_model,
            "embedding_model": config.embedding_model,
            "rerank_model": config.rerank_model,
        }
    )


async def kb_pdf(request: web.Request) -> web.StreamResponse:
    config: AppConfig = request.app["config"]
    name = request.match_info["name"]
    if "/" in name or "\\" in name:
        raise web.HTTPBadRequest(text="非法文件名。")
    path = config.knowledge_base_dir / name
    if not path.exists() or not path.is_file():
        raise web.HTTPNotFound(text="未找到 PDF 文件。")
    return web.FileResponse(path)


def create_app() -> web.Application:
    config = AppConfig.load()
    app = web.Application(client_max_size=2 * 1024 * 1024)
    app["config"] = config
    app.add_routes(
        [
            web.get("/", index),
            web.post("/api/ask", ask_api),
            web.get("/api/health", health_api),
            web.get("/kb/{name}", kb_pdf),
        ]
    )
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计学 RAG Web 演示服务")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认 127.0.0.1")
    parser.add_argument("--port", type=int, default=7860, help="监听端口，默认 7860")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = create_app()
    web.run_app(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
