---
name: beamer-talk
description: Build a new LaTeX beamer talk (seminar, conference, popular lecture) on quantum technologies, quantum computing, machine learning/AI or many-body physics from the material in doc/Talks and its subfolders. Use when asked to draft, extend or restyle a talk, to turn a paper (e.g. paper.tex) into slides, or to reuse slides and figures from existing talks such as qai.tex and pinns.tex.
---

# Beamer talks from the doc/Talks material

Talks in this repository live in `doc/Talks/` (and older ones in
`doc/Talks/TalksMaterialQML/`). They are plain `beamer` documents, compiled
with `pdflatex`, and they reuse a common pool of figures and reference slides.
This skill explains how to build a new talk in the same style.

## 1. Gather the material first

Before writing a single frame:

1. Read the two reference talks for style: `doc/Talks/pinns.tex` (short,
   equation-driven research talk, Madrid theme) and `doc/Talks/qai.tex`
   (longer overview talk, Madrid + seagull, `[plain,fragile]` frames, block
   environments, hyperlinks to papers). Copy their conventions, not their text.
2. Read the source material named by the user (a `paper.tex`, an earlier talk,
   notebooks under `doc/Programs`). Extract: the physical system, the
   Hamiltonian/model, the method, the 3-5 key equations, the figures
   (file names *and* captions), the headline numbers, the open challenges.
3. Look in `reference/figure-catalog.md` for reusable figures and in
   `reference/reusable-frames.md` for frames that can be lifted verbatim
   (acknowledgements, sponsors, ML taxonomy, Di Vincenzo criteria, quantum
   metrology, electrons-on-helium platform, ...).
4. Ask (or decide, if unattended) about: talk length, audience (popular /
   mixed / specialists), title, date and venue, and which application the
   talk should zoom in on.

## 2. Time budget and structure

Rule of thumb: about **1 slide per minute** including the title, and never
more than 1.2. A 20-minute talk is 18-22 frames; a 45-minute seminar 40-50.
Typical skeleton for an "overarching level" talk that ends in one concrete
application:

| Part | Share | Content |
|---|---|---|
| Opening | 3 frames | title, "what is this talk about" block, thanks + sponsors |
| Big picture | 30 % | what we mean by AI/ML, what we mean by quantum technologies, why the two are linked (AI for quantum, quantum for AI, ML as a tool for physics), figures from the catalog |
| Application | 45 % | platform, physical idea, model in 1-2 equations, where the ML/AI method enters, results with paper figures, limitations |
| Closing | 3-4 frames | perspectives/open questions, references with links, thank you |

Each frame should carry **one idea**: a short title, at most 5 bullets or one
displayed equation plus 2-3 lines of text, or one figure with a one-line
message. Put derivations in backup frames after `\appendix`.

## 3. Style conventions (match the existing talks)

Preamble to start from (see `reference/template.tex` for the full file):

```latex
\documentclass{beamer}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,bm}
\usepackage{physics}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{booktabs}
\usetheme{Madrid}
\usecolortheme{seagull}
\graphicspath{{./}{figures/}{TalksMaterialQML/figures/}{TalksMaterialQML/qcfigures/}}
```

* Title block in the form `\title[Short]{\textbf{Long title}}`,
  `\author{Morten Hjorth-Jensen}`, institute
  `Department of Physics and Center for Computing in Science Education, University of Oslo, Norway`,
  date = event and place.
* Frames: `\begin{frame}[plain,fragile]` + `\frametitle{...}` for content
  slides; `\begin{frame}{Title}` is fine for short ones. Use
  `\begin{block}{}` for a highlighted paragraph, `\begin{alertblock}{...}`
  for criteria/warnings, `\begin{columns}` for figure + text.
* Figures: `\centerline{\includegraphics[width=0.9\linewidth]{file}}` and a
  one-line source/credit under the figure (`Taken from ...`, or the paper
  reference with `\href`). Never rescale beyond `1.05\linewidth`.
* Equations: display math `\[ ... \]` with `\bm{}` for vectors, `\ket{}`
  from `physics`. Keep operator/notation identical to the source paper.
* Separate sections with `%-----` comment lines and `\section{}` so the
  outline (`\tableofcontents`) works.
* Always end with a references frame (author list, bold title, journal and
  `\href{url}{\nolinkurl{url}}`) and a `\Huge Thank you` frame.
* Language: English, first person plural ("we"), plain and pedestrian in
  the big-picture part, precise in the application part.

## 4. Write, compile, check

1. Write the talk as `doc/Talks/<shortname>.tex` (lower-case, no spaces).
   Paper figures used by the talk (e.g. `fig1.pdf`-`fig7.pdf`) must be in the
   same folder or on the `\graphicspath`.
2. Compile with `cd doc/Talks && pdflatex -interaction=nonstopmode <shortname>.tex`
   (twice for the outline). Fix every `!` error; grep the log for
   `Overfull \hbox` above 20pt and for missing figures
   (`File ... not found`).
3. Count frames (`grep -c 'begin{frame}'`) against the time budget, and
   check that every figure has a credit line and that every claim taken
   from the paper keeps its caveats (e.g. "theoretical proposal", "static
   field only").
4. Deliver the `.tex` and the compiled `.pdf`; list the frames in the reply
   so the user can see the storyline at a glance.

## 5. Don'ts

* Don't paste paragraphs from the paper: turn them into bullets or a single
  equation plus a spoken message.
* Don't invent results, numbers or references that are not in the source
  material or the catalog.
* Don't change the theme/colour scheme unless asked.
* Don't put tables of hyperparameters on main slides; use a backup frame.
