import streamlit.components.v1 as components


def render_copy_button(text_to_copy: str, key_suffix: str):
    escaped_text = (
        text_to_copy.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    )

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            background: transparent;
            font-family: "Source Sans Pro", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            display: flex;
            align-items: center;
            height: 100%;
        }}
        .copy-btn {{
            background: transparent;
            border: none;
            color: rgba(250, 250, 250, 0.8);
            font-size: 14px;
            font-weight: 400;
            cursor: pointer;
            padding: 4px 6px;
            border-radius: 6px;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            transition: color 0.15s ease, background-color 0.15s ease;
            user-select: none;
        }}
        .copy-btn:hover {{
            color: #ff4b4b; /* Exact Streamlit tertiary red accent */
            background-color: rgba(255, 75, 75, 0.08);
        }}
        .copy-btn:active {{
            transform: scale(0.97);
        }}
    </style>
    </head>
    <body>
        <button id="copy-btn-{key_suffix}" class="copy-btn" onclick="copyText_{key_suffix}()">
            <span style="font-size: 13px;">📋</span>
            <span id="copy-text-{key_suffix}">Copy</span>
        </button>

        <script>
        function copyText_{key_suffix}() {{
            const text = `{escaped_text}`;
            navigator.clipboard.writeText(text).then(() => {{
                const btn = document.getElementById("copy-btn-{key_suffix}");
                const label = document.getElementById("copy-text-{key_suffix}");
                label.innerText = "Copied!";
                btn.style.color = "#2ed573";
                setTimeout(() => {{
                    label.innerText = "Copy";
                    btn.style.color = "";
                }}, 1800);
            }}).catch(err => {{
                console.error("Clipboard error:", err);
            }});
        }}
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=32)
