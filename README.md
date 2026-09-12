# SuperMed Pharmacy Product Assistant

A catalog-grounded chatbot for supermedpharmacy.com/shop/. Built as a
Python serverless function on Vercel, using OpenAI for chat + moderation.

## How it works

- `api/catalog_data.py` holds the full product catalog (345 products),
  generated from the product CSV. It's loaded directly into the model's
  system prompt on every request — no vector DB needed at this catalog size.
- `api/chat.py` is the serverless function: it moderates the incoming
  message, sends it + the catalog to OpenAI, checks the reply only
  mentions catalog products, and returns it.
- `public/index.html` is a bare-bones test page so you can try the bot
  before embedding it in the real site.

## Setup

1. Install the Vercel CLI if you don't have it: `npm i -g vercel`
2. From this folder, log in and link the project:
   ```
   vercel login
   vercel link
   ```
3. Add your OpenAI API key as an environment variable (do this in the
   Vercel dashboard under Project Settings → Environment Variables, or
   via CLI):
   ```
   vercel env add OPENAI_API_KEY
   ```
4. Deploy:
   ```
   vercel --prod
   ```
5. Open the deployed URL — `public/index.html` will be served at the
   root, and you can test the chatbot there. The API itself lives at
   `/api/chat`.

## Updating the product catalog

Whenever the SuperMed product sheet changes, export it to CSV with the
same columns as the current one, then run:

```
python3 build_catalog.py path/to/updated_products.csv
```

This regenerates `api/catalog_data.py`. Redeploy (`vercel --prod`) to
push the update live.

## Embedding on the real WordPress site

Once you're happy with it, replace the `API_URL` in a chat widget
(reuse the JS in `public/index.html`, or build a nicer floating widget)
and either:
- Add it via a "Custom HTML" block/plugin in WordPress, or
- Load it as an enqueued script from your theme.

Make sure `Access-Control-Allow-Origin` in `api/chat.py` is narrowed
from `*` to `https://supermedpharmacy.com` before going live, so only
your site can call the API.

## Guardrails in place

1. **Moderation**: every user message (and the model's reply) is checked
   against OpenAI's Moderation API before use.
2. **Catalog grounding**: the system prompt instructs the model to only
   reference products in the CATALOG block, never to invent or assume
   products, and never to give dosing/diagnostic/medical advice.
3. **Post-response check**: a lightweight filter scans the model's reply
   for product-like phrases not found in the catalog text; if one shows
   up, the reply is discarded and a safe fallback message is returned
   instead.

## Notes on scale

At ~345 products, sending the whole catalog as context on each request
is simpler than a vector-search/RAG setup and keeps this stack easy to
maintain. If the catalog grows into the low thousands, this approach
will start costing more per request and it'll be worth revisiting
embeddings + pgvector (as discussed) instead of full-context loading.
