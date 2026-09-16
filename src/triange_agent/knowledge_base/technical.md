# Technical Support Policy

This document covers troubleshooting scope, escalation paths, and known
limitations for technical issues reported by customers.

## Login and Authentication Issues

The most common login failure is an expired or stale session token, which
resolves by having the customer log out completely and log back in. If a
customer is stuck in a password-reset loop (the reset email arrives but the
new password is immediately rejected), the cause is almost always a cached
old session on a different device; ask the customer to reset the password
and then sign out of all devices from the account security page before
signing back in. Multi-factor authentication (MFA) lockouts, where a customer
has lost access to their authenticator app, require identity verification
through the account-recovery flow before MFA can be disabled — support agents
cannot disable MFA on request without that verification, regardless of how
urgent the customer's request sounds.

## Sync and Data Loss

Reported data loss is first checked against the sync log, which retains the
last 14 days of sync events per account. Most "lost" data is actually stuck
in a sync conflict on another device, not truly deleted. If the sync log
shows the data was uploaded and then explicitly deleted by a client action
(not a sync failure), that is user-initiated deletion, not a bug, and is
out of scope for a technical fix — the customer should be directed to the
account's trash/recovery feature if the deletion was within the last 30 days.
True sync failures, where data was created locally but never appears in the
sync log at all, should be escalated to the engineering on-call queue with
the account ID and approximate timestamp.

## Performance and Timeouts

Slow load times reported by a single customer are usually network-side and
should be triaged with a traceroute or a request to try a different network
before assuming a server-side issue. Widespread performance complaints across
many customers in a short window indicate a platform incident and should be
escalated immediately rather than troubleshot individually — check the status
page first, since an active incident may already explain the report.
Timeouts on large file uploads are a known limitation for files over 2GB;
there is no current workaround besides splitting the file, and this should be
communicated as a known limitation, not treated as a bug to fix.

## Mobile App Crashes

Crash reports are automatically collected from the mobile app if the customer
has crash reporting enabled in settings; ask the customer to confirm this
setting before requesting manual logs, since manual log collection is a much
slower path. Crashes on app launch (before any UI renders) are almost always
resolved by a full app reinstall, since this clears any corrupted local
cache that a normal update does not touch. Crashes that occur only on a
specific action (e.g., always crashes when opening the export screen) should
be escalated with the exact reproduction steps, since these are more likely
to indicate an actual regression the engineering team needs to fix.

## Browser Extension Conflicts

A meaningful fraction of "the website is broken" reports are caused by
ad-blocking or privacy browser extensions interfering with the page's
scripts. The standard troubleshooting step is to ask the customer to try the
site in a private/incognito window with extensions disabled before escalating
further. If the issue persists in incognito mode, it is not an extension
conflict and should proceed to normal escalation.
