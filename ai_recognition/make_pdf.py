# Generates a PDF documenting the full design conversation for the
# AI/ML face-recognition service (Smart Attendance project).
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, HRFlowable
)

OUT = r"C:\Users\INTEL\Downloads\DEMO_PROJECT\Project_Discussion_AI_Recognition.pdf"

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=18,
                    textColor=colors.HexColor("#1a3c6e"), spaceAfter=8)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13,
                    textColor=colors.HexColor("#23527c"), spaceBefore=12, spaceAfter=6)
BODY = ParagraphStyle("BODY", parent=styles["Normal"], fontSize=10.5, leading=15,
                      spaceAfter=6)
BULLET = ParagraphStyle("BULLET", parent=BODY, leftIndent=12, bulletIndent=2)
Q = ParagraphStyle("Q", parent=BODY, textColor=colors.HexColor("#7a3b00"),
                   fontName="Helvetica-Bold", spaceBefore=8)
SUB = ParagraphStyle("SUB", parent=styles["Normal"], fontSize=9,
                     textColor=colors.grey)

story = []

def h1(t): story.append(Paragraph(t, H1))
def h2(t): story.append(Paragraph(t, H2))
def p(t): story.append(Paragraph(t, BODY))
def b(items):
    for it in items:
        story.append(Paragraph("&bull;&nbsp;&nbsp;" + it, BULLET))
def q(t): story.append(Paragraph(t, Q))
def sp(h=6): story.append(Spacer(1, h))
def rule(): story.append(HRFlowable(width="100%", thickness=0.6,
                                     color=colors.HexColor("#cccccc"),
                                     spaceBefore=6, spaceAfter=6))

def tbl(data, widths):
    t = Table(data, colWidths=widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a3c6e")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#bbbbbb")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f2f6fb")]),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(t)

# ---------------- TITLE ----------------
h1("Smart Attendance &mdash; AI/ML Face Recognition Service")
p("<b>Project:</b> Smart Attendance / Check-in System (Face Recognition)")
p("<b>Component owner role:</b> AI/ML face-recognition service")
p("<b>Document:</b> Design discussion &amp; decision record")
p("<b>Date:</b> 21 June 2026 &nbsp;|&nbsp; <b>Target demo:</b> Saturday, 27 June 2026 (group interview)")
sp(); rule()

# ---------------- 1. OVERVIEW ----------------
h2("1. Project Overview")
p("A smart attendance system that marks attendance using <b>live face recognition</b>. "
  "Built by a 5-member team, each owning one component, integrated via shared API contracts "
  "and GitHub feature branches.")
tbl([
    ["Component", "Responsibility"],
    ["Flutter (mobile)", "Camera captures face, sends image to backend"],
    ["Spring Boot (backend)", "Receives image, calls AI service, stores attendance in DB"],
    ["AI/ML service (THIS)", "Face recognition: image -> identity (local, pretrained, fast)"],
    ["React (admin)", "Dashboard: attendance logs, who checked in and when"],
    ["Lead / DB / DevOps", "Schema, integration, deployment glue"],
], [110, 330])
sp()

# ---------------- 2. ROLE & CONSTRAINTS ----------------
h2("2. My Role and Hard Constraints")
p("I own the <b>AI/ML face-recognition service</b> only. It talks to <b>Spring Boot only</b> "
  "(Flutter and React never call it directly).")
p("<b>Hard constraints agreed:</b>")
b([
    "<b>No external/cloud API</b> &mdash; no AWS Rekognition, Azure Face, Google Vision. Everything runs locally.",
    "<b>Pretrained model</b> &mdash; download weights once, zero training from scratch.",
    "<b>Fast / no delay</b> &mdash; recognition must feel instant.",
    "<b>Minimal storage, small hardware</b> &mdash; must fit a small edge device.",
])

# ---------------- 3. KEY CONCEPT ----------------
h2("3. The Core Concept: Embeddings (not fingerprints)")
p("The pretrained model is <b>not</b> trained to recognise specific people. It converts <b>any</b> "
  "face into a numeric vector called an <b>embedding</b> (512 numbers) &mdash; the same person's "
  "face always lands close together; different people land far apart.")
p("Note: 'fingerprint' was only used as an <b>analogy</b> for this embedding. The project is "
  "<b>purely face recognition</b> &mdash; no finger/thumb biometrics anywhere.")
p("<b>Adding a person = enrollment = one-shot:</b> run their face through the model once, store the "
  "embedding. No retraining. Recognising = embed the new face, compare to stored embeddings, pick "
  "the closest if it's within a threshold.")

# ---------------- 4. ARCHITECTURE DECISIONS ----------------
h2("4. Architecture Decisions")
tbl([
    ["Decision", "Choice", "Why"],
    ["Model / library", "InsightFace buffalo_s (ArcFace, MobileFaceNet-class), ONNX",
     "~tens of MB, CPU-only, ~10-30 ms/face; edge-friendly; pretrained"],
    ["Runtime", "onnxruntime", "Runs model locally on CPU, no GPU, no API"],
    ["Service framework", "FastAPI + uvicorn", "Local HTTP 'door' so Spring Boot can reach the model"],
    ["State", "Stateless service; backend DB owns embeddings",
     "Backend is source of truth; service is pure compute"],
    ["Speed trick", "Embeddings cached in RAM for matching",
     "Matching is microseconds; ~2 KB per face (512 floats)"],
    ["Matching owner", "The AI service does the matching",
     "Keeps ML logic + threshold on the AI side; backend stays simple"],
], [95, 150, 195])
sp()

# ---------------- 5. FASTAPI CLARIFICATION ----------------
h2("5. Does FastAPI break the 'no API' rule? No.")
p("'API' is an overloaded word:")
b([
    "<b>Avoided:</b> calling someone else's <b>cloud</b> service over the internet (costs money, needs WiFi, sends data out).",
    "<b>Used:</b> FastAPI is a <b>local doorway</b> on your own machine so two programs can talk. Nothing leaves the device.",
])
p("FastAPI does not do recognition &mdash; the model does. FastAPI is just the messenger that lets "
  "Spring Boot hand over an image and get an answer back. It works fully offline (WiFi off).")

# ---------------- 6. HOW IT TALKS TO SPRING BOOT ----------------
h2("6. How the Model Talks to Spring Boot")
b([
    "Flutter -> photo -> Spring Boot",
    "Spring Boot -> <b>POST http://localhost:8000/recognize</b> (with the image) -> AI service",
    "AI service -> model embeds + matches -> returns {person_id, confidence}",
    "Spring Boot -> saves attendance record (who + timestamp) to its DB",
    "React -> reads records -> shows the dashboard",
])

# ---------------- 7. API CONTRACT ----------------
h2("7. API Contract (the one coordination point with backend)")
tbl([
    ["Endpoint", "Input", "Output"],
    ["POST /enroll", "{person_id, image}", "{status, embedding}"],
    ["POST /recognize", "{image}", "{person_id, confidence} | 'unknown' | 'no_face'"],
    ["GET /health", "-", "{status: 'ok'}"],
], [120, 150, 170])
sp()

# ---------------- 8. FINE-TUNING ----------------
h2("8. Should We Fine-Tune the Model? No.")
p("Fine-tuning is the wrong instinct here because face recognition uses <b>metric learning</b> &mdash; "
  "recognising a new person is a one-shot <b>enrollment</b>, not a training problem.")
b([
    "It breaks the constraints (needs GPU, hours, a dataset; we want it fast and today).",
    "We don't have enough per-person data; small-data fine-tuning overfits and can make the model worse.",
    "Re-enrollment is free: add/remove a person = add/remove an embedding.",
])
p("<b>When fine-tuning IS legitimate:</b> domain shift (thermal/IR cameras, heavy masks, very low-res "
  "CCTV), or when you are building the embedding model itself. Not our case (normal RGB webcam).")

# ---------------- 9. WHAT IS ACTUALLY MY ENGINEERING WORK ----------------
h2("9. My Real Engineering Work (not 'copy-paste')")
p("The pretrained model is ~10% of the system. The engineering is everything around it:")
b([
    "<b>Threshold tuning</b> &mdash; choosing the similarity cut-off by testing false-accept vs false-reject.",
    "<b>Architecture</b> &mdash; stateless service + in-RAM matching + backend as source of truth.",
    "<b>API contract design</b> &mdash; the Python<->Java integration surface.",
    "<b>Edge-case hardening</b> &mdash; no face, multiple faces, unknown person, blurry image, persistence on restart.",
])

# ---------------- 10. HOW IT WORKS (FLOW) ----------------
h2("10. How My Service Works &mdash; Step by Step")
q("A. Startup (once)")
b([
    "Load pretrained model into memory.",
    "Load enrolled embeddings from backend / local file into RAM.",
    "Listen on localhost:8000. No internet used.",
])
q("B. Enroll a person")
b([
    "Receive POST /enroll {person_id, image}.",
    "Detect face -> align -> embed (512 numbers).",
    "Store embedding in RAM and return it so backend saves it in DB.",
])
q("C. Recognise (live check-in &mdash; hot path)")
b([
    "Receive POST /recognize {image}.",
    "Detect face. If none -> 'no_face'. If multiple -> pick largest / reject.",
    "Align + embed the face.",
    "Cosine-compare vs in-RAM embeddings -> closest match + score.",
    "Above threshold -> {person_id, confidence}; below -> 'unknown'.",
    "Compare step takes microseconds = the 'no delay'.",
])

# ---------------- 11. CONSTRAINTS CHECK ----------------
h2("11. How the Design Honours Every Constraint")
tbl([
    ["Constraint", "How it is met"],
    ["Local / no cloud API", "Model runs on device; zero internet; works WiFi-off"],
    ["Fast / no delay", "Embeddings in RAM; matching in microseconds; embed ~10-30 ms"],
    ["Minimal storage", "~2 KB per face; store embeddings, not photos"],
    ["Small hardware", "Tiny model (~tens of MB), CPU-only, runs on a Pi"],
    ["Pretrained, no fine-tuning", "Enrollment is one-shot embedding &mdash; correct by design"],
], [150, 290])
sp()

# ---------------- 12. DEMO STRATEGY ----------------
h2("12. Demo Strategy (Saturday, group interview)")
b([
    "Build the AI service to <b>stand alone</b> &mdash; demo works even if backend/Flutter aren't ready.",
    "Show: enrolled set -> live check-in -> instant match with name + confidence.",
    "<b>Killer move:</b> turn WiFi OFF and recognise a face &mdash; proves 'no API / fully local'.",
    "<b>Edge-case demo:</b> point at no one ('no_face'), then an un-enrolled person ('unknown') &mdash; proves it's a system, not a toy.",
    "Have a recorded video backup in case hardware misbehaves.",
])

# ---------------- 13. GIT / TEAM ----------------
h2("13. Repository &amp; Team Workflow")
p("GitHub repo: <b>NexTechANKARA/smart-attendance</b> (public). Branch per component:")
b([
    "main &mdash; always working; no direct commits.",
    "feature/ai-face-service &mdash; my branch.",
    "feature/flutter-camera, feature/springboot-api, feature/react-dashboard.",
    "Integrate via Pull Requests into main; pull often to stay in sync.",
])

# ---------------- 14. INTERVIEW SOUNDBITE ----------------
h2("14. Interview Soundbite")
p("<i>\"I built a stateless recognition service around a pretrained embedding model. My work was the "
  "engineering around it: I designed the enroll/recognise API contract with the backend, implemented "
  "in-RAM matching for sub-millisecond lookup, tuned the similarity threshold against "
  "false-accept/reject rates, and handled the real-world edge cases &mdash; no face, multiple faces, "
  "unknown person, and persistence across restarts. The pretrained model is one dependency; the "
  "system is mine.\"</i>")

story.append(Spacer(1, 14))
story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#cccccc")))
story.append(Paragraph("Generated design-discussion record &mdash; AI/ML face-recognition service.", SUB))

doc = SimpleDocTemplate(OUT, pagesize=A4,
                        leftMargin=18*mm, rightMargin=18*mm,
                        topMargin=16*mm, bottomMargin=16*mm,
                        title="Smart Attendance - AI Face Recognition Discussion")
doc.build(story)
print("PDF written to:", OUT)
