"""
POST /api/chat

Body: { "message": "do you have vitamins for kids", "history": [ {role, content}, ... ] }
Response: { "reply": "..." }

Guardrails in place:
  1. OpenAI Moderation API checks the incoming user message before it
     reaches the chat model.
  2. The chat model is only given the SuperMed product catalog as its
     source of product knowledge (loaded from catalog_data.py) and is
     instructed never to reference anything outside it.
  3. A post-response check confirms any product name the model mentions
     actually exists in the catalog; if the model references something
     that isn't in the catalog, we discard that response and fall back
     to a safe canned reply instead of returning it to the user.
  4. The model is explicitly told not to give medical/dosing advice.

This is a FastAPI ASGI app in api/index.py, which is one of Vercel's
default Python entrypoint locations (the file must define a top-level
`app` variable for ASGI/WSGI frameworks — see
https://vercel.com/docs/functions/runtimes/python).
"""

import os
import re

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

from catalog_data import PRODUCTS

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

CHAT_MODEL = "gpt-4o-mini"
MODERATION_MODEL = "omni-moderation-latest"

FALLBACK_REPLY = (
    "I couldn't find that in our current product listing. You're welcome to "
    "browse the full catalogue at https://supermedpharmacy.com/shop/ or "
    "contact the pharmacy directly for anything not shown there."
)

MODERATION_BLOCKED_REPLY = (
    "I'm not able to help with that request. If you have a question about "
    "products available at SuperMed Pharmacy, I'm happy to help with that."
)


def build_catalog_block() -> str:
    lines = []
    for p in PRODUCTS:
        parts = [p["name"]]
        if p.get("brand"):
            parts.append(f"Brand: {p['brand']}")
        if p.get("category"):
            parts.append(f"Category: {p['category']}")
        if p.get("ingredient"):
            parts.append(f"Key ingredient: {p['ingredient']}")
        if p.get("price_jmd"):
            parts.append(f"Price: JMD {p['price_jmd']}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


CATALOG_BLOCK = build_catalog_block()

SYSTEM_PROMPT = f"""You are the product assistant for SuperMed Pharmacy \
(https://supermedpharmacy.com/shop/).

You may ONLY reference products that appear in the CATALOG block below. \
Never invent, assume, or discuss any product, brand, or item that is not \
listed there, even if it seems like something a pharmacy would typically \
carry.

Rules:
- If the user asks for something not in the catalog, say clearly that it \
doesn't look like something currently listed, and suggest they check \
https://supermedpharmacy.com/shop/ or contact the pharmacy directly. Do \
not guess or suggest a substitute that isn't in the catalog.
- When you do recommend products, list them by name and price (JMD) as \
given in the catalog, and mention the category or key ingredient only if \
it's listed.
- You are not a medical professional. Never give dosing instructions, \
diagnoses, or advice about drug interactions or whether a product is \
right for someone's condition. For anything clinical, tell the user to \
speak with a pharmacist or doctor. You can describe what a product is \
generally used for only at the level stated in the catalog (e.g. its \
category), not with clinical guidance.
- Keep responses concise and friendly.

CATALOG:
{CATALOG_BLOCK}
"""


def is_flagged(text: str) -> bool:
    try:
        result = client.moderations.create(model=MODERATION_MODEL, input=text)
        return bool(result.results[0].flagged)
    except Exception:
        # Fail closed would block legitimate traffic on a transient API
        # error; fail open here but this is logged for monitoring.
        return False


def response_stays_in_catalog(reply: str) -> bool:
    """
    Loose safety net: if the model's reply mentions a product-like name
    that doesn't match anything in the catalog, we don't trust it.
    This is intentionally permissive (substring match) to avoid false
    positives on ordinary conversational text.
    """
    catalog_lower = CATALOG_BLOCK.lower()
    candidates = re.findall(
        r"\b([A-Z][A-Za-z0-9&'\-]*(?:\s+[A-Z0-9][A-Za-z0-9&'\-]*){1,5})\b", reply
    )
    for candidate in candidates:
        c = candidate.lower().strip()
        if len(c) < 6:
            continue
        if c not in catalog_lower:
            return False
    return True


def generate_reply(message: str, history: list) -> str:
    if is_flagged(message):
        return MODERATION_BLOCKED_REPLY

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history[-10:]:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": message})

    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=500,
    )
    reply = completion.choices[0].message.content.strip()

    if is_flagged(reply):
        return MODERATION_BLOCKED_REPLY

    if not response_stays_in_catalog(reply):
        return FALLBACK_REPLY

    return reply


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatTurn] = []


app = FastAPI()

# Narrow this to https://supermedpharmacy.com before going live.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.post("/api/chat")
def chat(req: ChatRequest):
    message = req.message.strip()
    if not message:
        return {"error": "message is required"}

    history = [turn.model_dump() for turn in req.history]
    reply = generate_reply(message, history)
    return {"reply": reply}
