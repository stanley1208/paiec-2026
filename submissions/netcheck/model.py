"""Network check, not a model: are api.anthropic.com and api.openai.com reachable during scoring?

It predicts a constant 0.5. The run fails on purpose if api.anthropic.com cannot be reached, so
the submission's status alone answers the question; both results also go to stderr for the
logs. No credentials are used: any HTTP reply to an unauthenticated request (such as 401)
shows that the host is reachable.
"""

import sys
import urllib.error
import urllib.request

HOSTS = ("api.anthropic.com", "api.openai.com")


def _reachable(host):
    try:
        urllib.request.urlopen(urllib.request.Request(f"https://{host}/v1/models"), timeout=10)
        return "reachable"
    except urllib.error.HTTPError as error:
        return f"reachable (HTTP {error.code})"
    except Exception as error:
        return f"unreachable ({type(error).__name__})"


STATUS = {host: _reachable(host) for host in HOSTS}
print(f"[network-check] {STATUS}", file=sys.stderr, flush=True)
if not STATUS["api.anthropic.com"].startswith("reachable"):
    raise RuntimeError(f"[network-check] api.anthropic.com {STATUS['api.anthropic.com']}")


def predict(input, labeled=None):
    return 0.5
