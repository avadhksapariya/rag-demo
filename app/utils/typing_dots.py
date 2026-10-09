TYPING_DOTS_LOADER_HTML = """
<div class="typing-indicator">
  <span></span>
  <span></span>
  <span></span>
</div>
<style>
.typing-indicator {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 8px 12px;
}
.typing-indicator span {
  width: 7px;
  height: 7px;
  background-color: #888888;
  border-radius: 50%;
  animation: pulse 1.4s infinite ease-in-out both;
}
.typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
.typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
@keyframes pulse {
  0%, 80%, 100% { transform: scale(0); opacity: 0.3; }
  40% { transform: scale(1); opacity: 1; }
}
</style>
"""
