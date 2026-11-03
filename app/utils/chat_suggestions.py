import json
import streamlit as st
from pathlib import Path

from app.core.config import DB_PATH


def get_prompts_for_scope(selected_file: str) -> list[str]:
    registry_file = Path(DB_PATH) / "document_prompts.json"
    if registry_file.exists():
        try:
            with open(registry_file, "r", encoding="utf-8") as f:
                registry = json.load(f)

            if selected_file in registry:
                return registry[selected_file]

            if selected_file == "All Documents" and registry:
                all_collected = []
                for file_prompts in registry.values():
                    if file_prompts:
                        all_collected.append(file_prompts[0])
                if len(all_collected) >= 4:
                    return all_collected[:4]
                elif all_collected:
                    # Pad with remaining prompts
                    for file_prompts in registry.values():
                        for p in file_prompts[1:]:
                            if p not in all_collected:
                                all_collected.append(p)
                            if len(all_collected) == 4:
                                break
                    return all_collected[:4]

        except Exception as e:
            print(f"Error reading document_prompts.json: {e}")

    # Fallback if no cached prompts exist yet
    target = (
        f"in {selected_file}"
        if selected_file != "All Documents"
        else "in these documents"
    )
    return [
        f"What is the main topic discussed {target}?",
        f"Summarize the key events or decisions {target}.",
        f"Who are the important entities or people involved?",
        f"What are the final conclusions or results reached?",
    ]


# Renders tailored prompt cards based on the active document.
def render_starter_prompts():
    selected_doc = st.session_state.get("selected_file", "All Documents")
    prompts = get_prompts_for_scope(selected_doc)

    scope_name = (
        f"{selected_doc}"
        if selected_doc != "All Documents"
        else "all indexed documents"
    )

    st.markdown(
        f"""
        <div style="margin: 28px 0 16px 0; text-align: center;">
            <p style="color: #8892b0; font-size: 0.95em; margin: 0;">
                💡 Suggested questions for {scope_name}:
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    for idx, prompt_text in enumerate(prompts):
        target_col = col1 if idx % 2 == 0 else col2
        with target_col:
            if st.button(
                f"💬 {prompt_text}",
                key=f"dyn_prompt_{idx}",
                use_container_width=True,
                type="secondary",
            ):
                st.session_state.retry_query = prompt_text
                st.rerun()
