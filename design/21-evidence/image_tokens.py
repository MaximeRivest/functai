"""Inkling's picture token count, measured from tml-renderers 0.4.1 (Inkling2 renderer)."""
import io, base64, math
from PIL import Image
from tml_renderers import renderers, chat
r = renderers.Inkling2()
def tokens(w, h):
    b = io.BytesIO(); Image.new("RGB", (w, h)).save(b, "PNG")
    u = "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
    m = chat.OpenAIMessage.from_openai_chat_dicts([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": u}}]}])
    return [s.span for s in r.render_for_completion_with_effort(m, 0.0)[0] if type(s.span).__name__ == "ImageAssetPointerTokenSpan"][0].num_tokens
for w, h in [(1, 1), (39, 39), (40, 40), (41, 41), (79, 79), (80, 80), (81, 81), (119, 40), (120, 40), (121, 40),
             (40, 119), (40, 120), (40, 121), (390, 844), (1279, 800), (1280, 800), (1281, 800), (4000, 3000), (8000, 8000)]:
    client = (w // 40 + 1) * math.ceil(h / 40); server = math.ceil(w / 40) * math.ceil(h / 40); got = tokens(w, h)
    print(f"{w}x{h}: renderer {got}  (floor(w/40)+1)*ceil(h/40) {client}  ceil(w/40)*ceil(h/40) {server}  {'differs: Tinker refuses these (width a multiple of 40)' if got != server else ''}")
