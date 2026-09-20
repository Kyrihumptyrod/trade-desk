# Cloud setup — about 10 minutes, once

Works from a phone browser (use "Request desktop site" in Safari if a button is missing) or the Mac.

## 1. Create the repo
1. github.com → **+** → **New repository**. Name it `trade-desk`. Set it to **Private**. Tick **Add a README**. Create.
2. **Add file → Upload files**. Upload `trade_desk.py`, `README.md`, `SETUP_GITHUB.md`. Commit.
3. **Add file → Create new file**. In the filename box type exactly:
   `.github/workflows/daily.yml`
   (typing the slashes creates the folders). Paste in the contents of `daily.yml` from this pack. Commit.
4. Same again: filename `docs/.gitkeep`, leave the file empty, commit.

## 2. Add your API key
**Settings → Secrets and variables → Actions → New repository secret.**
Name: `ANTHROPIC_API_KEY`. Value: your key. Add.
(Without it the job still runs; picks just come without the written reasoning.)

## 3. Turn on the web page
**Settings → Pages → Source: Deploy from a branch → Branch: main, folder: /docs → Save.**
Your page will be `https://YOUR-USERNAME.github.io/trade-desk/`

## 4. Run it once now
**Actions** tab → **Daily picks** → **Run workflow** → Run. Wait a minute, refresh: green tick means it worked.
Open the page URL from step 3 (first publish can take up to 5 minutes). On iPhone: Share → **Add to Home Screen**.

## What happens every day
- 08:05 AWST: the job pulls prices, grades yesterday's open picks, adds today's picks, rewrites the page.
- Every run is committed, so `docs/history.csv` is a tamper-proof record with the full commit history.
- Price data: Hyperliquid first, Kraken as backup (Kraken has no HYPE, so HYPE is skipped if Hyperliquid is unreachable).

## Changing things
- Coins or thresholds: edit the `python trade_desk.py daily ...` line in `.github/workflows/daily.yml`.
  `--threshold 6` for fewer, stronger picks; `--max-picks 2` to cap at two.
- Time: the cron line is in UTC. `5 0` = 08:05 AWST.
- Stop it: **Actions → Daily picks → ⋯ → Disable workflow.**

## If a run fails
Open the failed run in Actions and read the last red step. The usual causes: the API key secret name is misspelt,
or Pages isn't enabled yet. Paste the error into the chat and I'll fix it.
