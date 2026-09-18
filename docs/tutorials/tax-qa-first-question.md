# Getting started: ask the Tax Q&A chatbot a question

This tutorial walks you through opening the Income Tax Q&A chatbot and asking your first
question. By the end, you'll have asked a question in Hebrew, read the agent's answer, checked
where that answer came from in the Income Tax Ordinance, and started a fresh conversation.

You don't need any technical background to follow this. If someone on your team has already set
up and started the chatbot for you, skip straight to [Step 2](#step-2-ask-a-question).

## Prerequisites

- The chatbot has been started and is running somewhere you can reach in a browser (see
  [Step 1](#step-1-open-the-chatbot)). If nobody has set this up yet, see the
  [how-to guide](../how-to/tax-qa-run-and-troubleshoot.md) for running it locally or with Docker.
- A modern web browser.

## Step 1: Open the chatbot

Go to the address your team gave you for the chatbot (if you started it yourself locally, this
is usually `http://localhost:3000/`).

You'll see a chat screen with a text box at the bottom and two buttons: one to send your
question, and one labeled **שאלה חדשה** ("New question").

## Step 2: Ask a question

Type a question in Hebrew about the Income Tax Ordinance into the text box — for example, a
question about a deduction, an exemption, or a reporting obligation. Press **Enter**, or click
the send button.

The agent reads your question, searches the Income Tax Ordinance for relevant sections, and
writes back an answer. This can take a few seconds — you'll see a thinking indicator while it
works.

## Step 3: Read the answer

Every answer arrives with a colored badge at the top telling you how confident the agent is:

| Badge | Meaning |
| --- | --- |
| ✔ נענתה (answered) | The agent found a section of the Ordinance that directly addresses your question. |
| ⚠ ודאות לא נמוכה (uncertain) | The agent found a related section, but isn't fully confident it answers your exact question. Read it critically. |
| ✗ לא נמצא (not found) | The agent couldn't find anything in the Ordinance that addresses your question. No answer is given — only the closest sections it considered, for reference. |

Below the answer text, a **מקורות** ("Sources") panel lists the section(s) of the Ordinance the
answer is based on, each with a short quoted excerpt. Always check the sources — they're what
let you verify the answer against the actual text of the law rather than taking the agent's
word for it.

If the badge shows **✗ לא נמצא**, there's no synthesized answer — only a list of the nearest
sections the agent found, so you can judge for yourself whether any of them are relevant.

## Step 4: Ask a follow-up

You can keep asking questions in the same conversation — type your next question the same way.
The agent remembers what you've already asked in this session.

## Step 5: Start over

When you want to ask about something unrelated, click **שאלה חדשה** ("New question"). This
clears the conversation and starts a fresh session — the agent won't carry over context from
your previous questions.

## Where to go next

- **[How to run and troubleshoot the chatbot](../how-to/tax-qa-run-and-troubleshoot.md)** — for
  running it yourself locally or with Docker, switching providers, and what to do about
  uncertain or not-found answers.
