# SAM 3 Visual Match

React Native Web interface for iterative same-image grounding. Draw a positive
box to find similar objects, add another positive box for a missed variation, or
draw a negative box to suppress a false-positive family.

From the repository root, start the FastAPI server:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

In another terminal, start the Expo web app:

```powershell
cd app
npm run web
```

Open `http://localhost:8081`. The first image upload loads the local checkpoint
and can take longer than later prompt updates. Set `EXPO_PUBLIC_API_URL` when the
API is not running at `http://127.0.0.1:8000`.

The API reads `weight/sam3.1_multiplex.pt` and `weight/visual_token.pt` by
default. Override them with `SAM3_WEIGHT`, `SAM3_VISUAL`, and `SAM3_DEVICE`.
