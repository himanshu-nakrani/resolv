"""resolv as a GitHub service (Phase 2).

Three cooperating processes share a database and artifact storage:

- web: receives GitHub webhooks and serves the maintainer dashboard. It holds
  the webhook secret and the OAuth client secret, but not the GitHub App key.
- gateway: the only component holding the GitHub App private key. It verifies
  triggers, pins the base commit, fetches the issue and a repository snapshot,
  posts status comments, and publishes approved patches as draft pull requests
  using short-lived, single-repository installation tokens.
- worker: runs the orchestrator against the snapshot inside the Docker sandbox.
  It never holds GitHub credentials.
"""
