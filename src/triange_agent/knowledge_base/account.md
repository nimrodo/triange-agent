# Account Policy

This document covers account ownership, access changes, deletion, and data
export requests.

## Email and Username Changes

Customers can change the email address on their account themselves from
account settings, provided they can log in and confirm the change via a link
sent to the new address. If a customer has lost access to their current
email and cannot receive the confirmation link, they must go through identity
verification with support before the email can be changed manually. Username
changes (where usernames are distinct from email) are self-service and take
effect immediately, but a username cannot be reused by a different account
for 90 days after being freed up, to avoid impersonation confusion.

## Account Recovery

Account recovery for a customer who has lost access to both their password
and their registered email requires two independent pieces of identity
verification (for example, a government ID plus a recent billing statement
matching the account's payment history). A single piece of evidence is not
sufficient, even if the customer is highly confident and frustrated — this
is a hard policy line to prevent account takeover via social engineering.
Recovery requests are handled by a dedicated recovery queue, not by
first-line support directly changing account access.

## Account Deletion

Account deletion requests are honored within 30 days, during which the
account enters a "pending deletion" state and can still be reactivated by the
customer logging back in. After the 30-day window, the deletion is permanent
and cannot be reversed, including by support staff. Deletion removes personal
data and content but does not automatically cancel an active subscription —
subscriptions must be cancelled separately before or during the deletion
request, otherwise billing continues even though account access is being
removed. Support should always confirm the customer has cancelled any active
subscription when processing a deletion request.

## Data Export

Customers can request a full export of their account data at any time from
account settings; the export is generated asynchronously and emailed as a
download link within 24 hours. Export links expire after 7 days for security
reasons, after which the customer must request a new export rather than
support re-sending the old link. Data export requests are separate from, and
do not trigger, account deletion — a customer can export their data and keep
their account active indefinitely.

## Team and Shared Account Access

For accounts with multiple team members, only the account owner can add or
remove other members' access, or transfer ownership to another member.
Ownership transfer requires the current owner to confirm the transfer via a
secondary email or SMS confirmation, and cannot be completed by a support
agent on the owner's behalf even with a support ticket describing the
request. If the owner has left the organization and is unreachable, the
recovery queue's business-verification path (not the standard account
recovery path above) applies, since it requires proving organizational
authority rather than personal identity.
