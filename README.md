# Smart Attendance / Check-in System (Face Recognition)

A 5-member project. Each component lives on its own feature branch and integrates via a shared API contract.

## Architecture

```
Flutter (camera) ──▶ Spring Boot (backend/DB) ──▶ AI/ML face service (local, pretrained)
                                  │
                                  └──▶ React (admin dashboard)
```

| Component | Branch |
|-----------|--------|
| AI/ML face recognition service | `feature/ai-face-service` |
| Flutter mobile app (camera/check-in) | `feature/flutter-camera` |
| Spring Boot backend API | `feature/springboot-api` |
| React admin dashboard | `feature/react-dashboard` |

## Branch workflow

- `main` — always working. No direct commits.
- `feature/*` — one branch per component. Work here, then open a Pull Request.
- Pull from `main` often to stay in sync and avoid large merge conflicts.

## AI/ML service notes

- Fully **local** — no external/cloud API calls.
- **Pretrained** model (no training from scratch).
- Optimized for **fast** recognition (embedding compared in milliseconds).
