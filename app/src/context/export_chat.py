from datetime import datetime
from typing import List, Dict, Any


# Formats in-memory session messages and citations into a clean Markdown document.
def format_chat_as_markdown(
    messages: List[Dict[str, Any]], session_title: str = "Chat Session"
) -> str:
    lines = [
        f"# {session_title}",
        f"*Exported on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "---",
        "",
    ]

    for idx, msg in enumerate(messages, 1):
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "").strip()

        lines.append(f"### {role}")
        lines.append(content)
        lines.append("")

        sources = msg.get("sources", [])
        if sources:
            lines.append("**Sources & Citations:**")
            for s_idx, src in enumerate(sources, 1):
                page_info = f" (Page {src['page']})" if src.get("page") else ""
                lines.append(
                    f"- **[{s_idx}] {src.get('file', 'Document')}{page_info}**"
                )
                raw_snippet = (
                    src.get("text", "").replace("<mark>", "**").replace("</mark>", "**")
                )
                lines.append(f"  > {raw_snippet}")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)
