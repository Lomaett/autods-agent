# AutoDS Frontend

Next.js + Tailwind UI for the AutoDS FastAPI backend.

## Setup

```bash
cd frontend
cp .env.example .env.local
npm install
npm run dev
```

Frontend: `http://localhost:3000`

Expected backend API base: `http://127.0.0.1:8080`

## Pages

- `/` dashboard
- `/run` launch `/eda` and `/analyse`
- `/models` list trained models
- `/reports` list and open generated reports
- `/predict` send inference requests
