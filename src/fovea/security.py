"""Request guards for a server that has no user accounts.

Fovea trusts whoever can reach it, so the boundary is *who can reach it*:

* Host check — a browser page on another site can make its own domain resolve to 127.0.0.1 (DNS rebinding)
  and then read every endpoint as same-origin. Such a request still carries the attacker's domain in its
  Host header, so only loopback names, IP literals, this machine's name and FOVEA_ALLOWED_HOSTS pass.
* Origin check — state-changing requests (POST/PUT/PATCH/DELETE) sent by a page of another origin are
  refused, whatever their content type, so a malicious page can never clear labels or start jobs.
* Optional token — FOVEA_TOKEN, required from non-local clients when set (Docker / LAN use).
* Remote AI guard — cloud providers spend the owner's money: requests from other machines may not set keys
  or call them unless FOVEA_ALLOW_REMOTE_AI=1.
* scrub() — removes API keys and bearer tokens from any text that can reach a client or a log.
"""
from __future__ import annotations

import hmac
import html
import ipaddress
import json
import logging
import os
import re
import socket
import threading
from typing import Any, Iterable, List, Optional, Tuple
from urllib.parse import parse_qsl, quote, quote_from_bytes, urlencode, urlsplit

from starlette.datastructures import MutableHeaders

log = logging.getLogger("fovea.security")

UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
# Never a single-label test name such as 'testserver': on a LAN an attacker can answer one (LLMNR, mDNS, a DNS search
# suffix) and rebind it to 127.0.0.1. Tests allow TestClient's name through FOVEA_ALLOWED_HOSTS (tests/conftest.py).
LOOPBACK_NAMES = frozenset({"localhost", "127.0.0.1", "::1"})
TOKEN_COOKIE = "fovea_token"


def _env_list(name: str) -> List[str]:
    return [x.strip() for x in os.environ.get(name, "").split(",") if x.strip()]


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


# ------------------------------------------------------------------------------------------- clients

def _local_nets() -> List[ipaddress._BaseNetwork]:
    """Extra client networks treated as this machine. Docker publishes the port through its bridge, so
    a browser on the host reaches the container from the bridge gateway (e.g. 172.17.0.1), never from
    127.0.0.1 — docker-compose binds the port to 127.0.0.1 and lists the bridge range here."""
    nets = []
    for raw in _env_list("FOVEA_LOCAL_NETS"):
        try:
            nets.append(ipaddress.ip_network(raw, strict=False))
        except ValueError:
            log.warning("FOVEA_LOCAL_NETS: ignoring %r (not a network)", raw)
    return nets


_LOCAL_NETS = _local_nets()


def _client_host(scope_or_request: Any) -> str:
    client = None
    if isinstance(scope_or_request, dict):
        client = scope_or_request.get("client")
    else:
        client = getattr(scope_or_request, "client", None)
    if not client:
        return ""
    host = client[0] if isinstance(client, (tuple, list)) else getattr(client, "host", "")
    return (host or "").strip()


def is_local_address(host: str) -> bool:
    if not host:
        return False
    if host in ("testclient", "localhost"):
        return True                         # Starlette's TestClient runs in-process
    try:
        ip = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_loopback or any(ip in n for n in _LOCAL_NETS)


def is_local_request(request: Any) -> bool:
    """True when the request comes from this machine (loopback, or a network listed in FOVEA_LOCAL_NETS).
    Accepts a Starlette Request or a raw ASGI scope."""
    return is_local_address(_client_host(request))


# ------------------------------------------------------------------------------------------- remote AI

def remote_ai_allowed() -> bool:
    return _truthy("FOVEA_ALLOW_REMOTE_AI")


def ai_access_allowed(request: Any) -> bool:
    return remote_ai_allowed() or is_local_request(request)


REMOTE_AI_MESSAGE = ("Cloud AI providers and API keys can only be used from the computer Fovea runs on. "
                     "Start Fovea with FOVEA_ALLOW_REMOTE_AI=1 to allow other machines.")


def require_ai_access(request: Any, action: str = "") -> None:
    """Raise 403 unless this request may set/clear keys or call a cloud provider."""
    if ai_access_allowed(request):
        return
    from fastapi import HTTPException
    raise HTTPException(403, (f"{action}: " if action else "") + REMOTE_AI_MESSAGE)


def ai_guard(request: "Any") -> None:
    """FastAPI dependency form of require_ai_access: `dependencies=[Depends(security.ai_guard)]`."""
    require_ai_access(request)


# ------------------------------------------------------------------------------------------- hosts

def bind_host() -> str:
    return os.environ.get("FOVEA_BIND_HOST", "").strip()


def loopback_only() -> bool:
    """Whether the server listens on loopback only (None/unknown counts as not loopback-only)."""
    h = bind_host()
    if not h:
        return False
    if h == "localhost":
        return True
    try:
        return ipaddress.ip_address(h.strip("[]")).is_loopback
    except ValueError:
        return False


def _machine_names() -> set:
    names = set()
    try:
        hn = socket.gethostname().lower().rstrip(".")
        if hn:
            names.add(hn)
            short = hn.split(".", 1)[0]
            names.update({short, f"{short}.local"})
    except OSError:
        pass
    return names


_ALLOWED_HOSTS = [h.lower().rstrip(".") for h in _env_list("FOVEA_ALLOWED_HOSTS")]
_ANY_HOST = "*" in _ALLOWED_HOSTS
_MACHINE_NAMES = _machine_names()


def split_host(host_header: str) -> Tuple[str, Optional[int]]:
    """'Example.com:8000' -> ('example.com', 8000); '[::1]:8000' -> ('::1', 8000)."""
    h = (host_header or "").strip().lower()
    port: Optional[int] = None
    if h.startswith("["):
        end = h.find("]")
        if end == -1:
            return h, None
        rest = h[end + 1:]
        h = h[1:end]
        if rest.startswith(":") and rest[1:].isdigit():
            port = int(rest[1:])
        return h, port
    if h.count(":") == 1:
        name, _, p = h.partition(":")
        if p.isdigit():
            port = int(p)
        h = name
    return h.rstrip("."), port


def _is_ip(name: str) -> bool:
    try:
        ipaddress.ip_address(name)
        return True
    except ValueError:
        return False


def host_allowed(host_header: str) -> bool:
    """A Host header DNS rebinding cannot produce: rebinding needs a domain name the attacker controls,
    so IP literals are always fine (a browser only sends one when the user typed that address)."""
    if _ANY_HOST:
        return True
    name, _port = split_host(host_header)
    if not name:
        return False
    if name in LOOPBACK_NAMES or name.endswith(".localhost") or _is_ip(name):
        return True
    if name in _MACHINE_NAMES:
        return True
    for pat in _ALLOWED_HOSTS:
        if pat == name or (pat.startswith("*.") and name.endswith(pat[1:])):
            return True
    return False


def _default_port(scheme: str) -> Optional[int]:
    return {"http": 80, "https": 443}.get(scheme)


def origin_matches(origin: str, host_header: str, scheme: str = "http") -> bool:
    """Same-origin test for an Origin header against the Host the request was sent to. The scheme is not
    compared: behind a TLS-terminating proxy the app sees http while the page is https."""
    if not origin or origin == "null":
        return False
    if origin.lower() in (o.lower().rstrip("/") for o in _env_list("FOVEA_ALLOWED_ORIGINS")):
        return True
    try:
        u = urlsplit(origin)
    except ValueError:
        return False
    if not u.hostname:
        return False
    o_port = u.port or _default_port(u.scheme)
    h_name, h_port = split_host(host_header)
    if h_port is None:
        h_port = _default_port(scheme)
        if h_port != o_port and o_port == _default_port(u.scheme):
            h_port = o_port              # Host without a port behind a proxy that changed the scheme
    return u.hostname.lower().rstrip(".") == h_name and o_port == h_port


def cross_site_reason(method: str, headers: dict, scheme: str = "http") -> Optional[str]:
    """Why an unsafe request must be refused as cross-site, or None. Requests without Origin and
    Sec-Fetch-Site (curl, scripts, the Python SDK) are not browser CSRF and pass."""
    if method.upper() not in UNSAFE_METHODS:
        return None
    sfs = (headers.get("sec-fetch-site") or "").lower()
    if sfs and sfs not in ("same-origin", "none"):
        return f"cross-site request blocked (Sec-Fetch-Site: {sfs})"
    origin = headers.get("origin")
    if origin is not None and not origin_matches(origin, headers.get("host", ""), scheme):
        return f"cross-origin request blocked (Origin: {origin[:80]})"
    return None


# ------------------------------------------------------------------------------------------- token

def access_token() -> str:
    return os.environ.get("FOVEA_TOKEN", "").strip()


def _token_ok(given: str, token: str) -> bool:
    return bool(given) and hmac.compare_digest(given.encode(), token.encode())


def _cookie(headers: dict, name: str) -> str:
    for part in (headers.get("cookie") or "").split(";"):
        k, _, v = part.strip().partition("=")
        if k == name:
            return v
    return ""


_LOGIN_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fovea · access token</title><style>
:root{color-scheme:light dark;--bg:#f6f7f8;--fg:#1c2024;--muted:#5b636b;--line:#d5d9dd;--accent:#0f766e;--on-accent:#fff;--err:#c62f2f}
@media (prefers-color-scheme:dark){:root{--bg:#16181b;--fg:#e6e8ea;--muted:#a0a7ae;--line:#33383d;--accent:#2dd4bf;--on-accent:#062521;--err:#ff8a80}}
body{margin:0;min-height:100vh;display:grid;place-items:center;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
form{width:min(360px,calc(100vw - 32px));display:grid;gap:12px}h1{font-size:18px;margin:0}p{margin:0;color:var(--muted)}
input{font:inherit;padding:8px 10px;border:1px solid var(--line);border-radius:6px;background:transparent;color:inherit}
button{font:inherit;padding:8px 12px;border:0;border-radius:6px;background:var(--accent);color:var(--on-accent);cursor:pointer}
.err{color:var(--err)}</style></head><body><form method="post" action="__ACTION__">
<h1>Fovea</h1><p>This server requires an access token (FOVEA_TOKEN) for connections from other machines.</p>
__ERR__<input name="token" type="password" autocomplete="current-password" autofocus aria-label="Access token" placeholder="Access token">
<button type="submit">Continue</button></form></body></html>"""
_LOGIN_ERROR = '<p class="err">That token is not right.</p>'
_FORM_LIMIT = 8192


def _login_page(action: str, wrong: bool) -> str:
    # POSTed to the page's own URL: a GET form would replace the query (a shared deep link such as
    # /d/x/annotate?img=5 lost its image) and put the typed token, right or wrong, into the address bar and history.
    return _LOGIN_PAGE.replace("__ACTION__", html.escape(action, quote=True)).replace("__ERR__", _LOGIN_ERROR if wrong else "")


def _self_location(scope: dict, params: List[Tuple[str, str]]) -> str:
    """This request's own path and query without any token parameter, as a path on this server (never
    '//other.host/…', which a browser would follow off-site)."""
    raw = scope.get("raw_path")
    if isinstance(raw, (bytes, bytearray)) and raw:
        path = quote_from_bytes(bytes(raw), safe="/%:@!$&'()*+,;=~")
    else:
        path = quote(scope.get("path") or "/", safe="/:@!$&'()*+,;=~")
    path = "/" + path.lstrip("/\\")
    rest = urlencode([(k, v) for k, v in params if k != "token"])
    return path + (f"?{rest}" if rest else "")


async def _form_token(receive) -> str:
    """The 'token' field of a urlencoded form body (empty when absent or the body is unreasonably long)."""
    body = b""
    while True:
        msg = await receive()
        if msg.get("type") != "http.request":
            return ""
        body += msg.get("body", b"")
        if len(body) > _FORM_LIMIT:
            return ""
        if not msg.get("more_body"):
            break
    return next((v for k, v in parse_qsl(body.decode("utf-8", "replace"), keep_blank_values=True) if k == "token"), "")


# ------------------------------------------------------------------------------------------- scrubbing

_SECRET_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{6,}"), "sk-ant-…"),
    (re.compile(r"\bsk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_\-]{16,}"), "sk-…"),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}"), "AIza…"),
    (re.compile(r"\b(?:ghp|gho|ghs|github_pat|hf|xox[abpr])_[A-Za-z0-9_\-]{12,}"), "[redacted]"),
    (re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._\-~+/]{8,}=*"), r"\1 …"),
    (re.compile(r"(?i)([?&](?:key|api[_-]?key|apikey|token|access_token|auth|secret)=)[^&\s\"'<>]+"), r"\1…"),
    (re.compile(r"(?i)((?:x-api-key|x-goog-api-key|api[_-]?key|authorization|secret|password|token)[\"']?\s*[:=]\s*[\"']?)"
                r"(?!…)[^\s\"',;}{]{6,}"), r"\1…"),
    # ai-cloud hook: credentials inside a URL (a gateway base URL set as http://user:password@host/v1)
    (re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://)[^\s/?#@'\"<>]+@"), r"\1…@"),
]
_KNOWN_SECRETS: set = set()
_KNOWN_LOCK = threading.Lock()


def register_secret(value: Optional[str]) -> None:
    """Make scrub() remove this exact value too (keys whose format no pattern knows, e.g. a local
    OpenAI-compatible server's key). Values shorter than 8 characters are ignored: too likely to collide."""
    if value and len(value) >= 8:
        with _KNOWN_LOCK:
            _KNOWN_SECRETS.add(value)


def scrub(text: Any) -> str:
    """Remove API keys, bearer tokens and key=... query values from text bound for a client or a log."""
    if text is None:
        return ""
    s = str(text)
    if not s:
        return s
    with _KNOWN_LOCK:
        known = sorted(_KNOWN_SECRETS, key=len, reverse=True)
    tok = access_token()
    if tok and len(tok) >= 4:
        known.append(tok)
    for v in known:
        if v in s:
            s = s.replace(v, "[redacted]")
    for pat, rep in _SECRET_PATTERNS:
        s = pat.sub(rep, s)
    return s


# 'token' only as a whole name part: hf_token / token are secrets, max_tokens / token_budget settings are not.
_SECRET_NAME = re.compile(r"(?i)(api[_\-]?key|apikey|secret|passw(?:or)?d|credential|private[_\-]?key|access[_\-]?key|"
                          r"(?:^|[_\-.])(?:auth[_\-]?)?token(?:$|[_\-.](?!budget|limit|count))|^key$|bearer)")
_SECRET_VALUE = re.compile(r"^(?:sk-ant-|sk-|AIza|Bearer\s|ghp_|hf_|xox[abpr]-)")


def is_secret_name(name: Any) -> bool:
    return isinstance(name, str) and bool(_SECRET_NAME.search(name))


def looks_like_secret(value: Any) -> bool:
    return isinstance(value, str) and len(value) >= 12 and bool(_SECRET_VALUE.match(value.strip()))


def find_secrets(obj: Any, path: str = "") -> List[str]:
    """Paths of entries that look like credentials: by key name (api_key, token, ...) or by value shape."""
    out: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            if is_secret_name(k) and v not in (None, "", False):
                out.append(p)
            else:
                out.extend(find_secrets(v, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(find_secrets(v, f"{path}[{i}]"))
    elif looks_like_secret(obj):
        out.append(path or "value")
    return out


def strip_secrets(obj: Any) -> Any:
    """A copy of obj without credential-looking entries (defence for settings written by older versions)."""
    if isinstance(obj, dict):
        return {k: strip_secrets(v) for k, v in obj.items() if not is_secret_name(k) and not looks_like_secret(v)}
    if isinstance(obj, list):
        return [strip_secrets(v) for v in obj if not looks_like_secret(v)]
    return obj


class RedactingFilter(logging.Filter):
    """Scrubs log records (message, args and formatted tracebacks) before any handler writes them."""

    @staticmethod
    def _arg(a: Any) -> Any:
        # Keep numbers as numbers (formatters use %d); anything else is rendered and scrubbed now, so an
        # exception or URL object passed as an argument cannot smuggle a key past the filter.
        if a is None or isinstance(a, (bool, int, float)):
            return a
        return scrub(a if isinstance(a, str) else str(a))

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if isinstance(record.msg, str):
                record.msg = scrub(record.msg)
            # The structure of args must survive: uvicorn's access formatter unpacks a 5-tuple.
            if isinstance(record.args, tuple):
                record.args = tuple(self._arg(a) for a in record.args)
            elif isinstance(record.args, dict):
                record.args = {k: self._arg(v) for k, v in record.args.items()}
            if record.exc_info and not record.exc_text:
                record.exc_text = scrub(logging.Formatter().formatException(record.exc_info))
            elif record.exc_text:
                record.exc_text = scrub(record.exc_text)
        except Exception:  # noqa: BLE001 — logging must never fail because of the filter
            pass
        return True


_filter = RedactingFilter()


def install_log_redaction(names: Iterable[str] = ("", "uvicorn", "uvicorn.error", "uvicorn.access", "fovea")) -> None:
    """Attach the redacting filter to every handler of these loggers (filters on a logger do not see
    records propagated from its children; handler filters do)."""
    fovea_log = logging.getLogger("fovea")
    if not fovea_log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(levelname)s:     %(name)s: %(message)s"))
        fovea_log.addHandler(h)
        fovea_log.setLevel(logging.INFO)
        fovea_log.propagate = False
    for name in names:
        lg = logging.getLogger(name)
        for h in lg.handlers:
            if _filter not in h.filters:
                h.addFilter(_filter)
    last = logging.lastResort
    if last is not None and _filter not in last.filters:
        last.addFilter(_filter)


# ------------------------------------------------------------------------------------------- middleware

def _json_response(status: int, detail: str, extra_headers: Optional[List[Tuple[bytes, bytes]]] = None):
    body = json.dumps({"detail": detail}).encode()
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()),
               (b"cache-control", b"no-store")] + (extra_headers or [])
    return {"type": "http.response.start", "status": status, "headers": headers}, {"type": "http.response.body", "body": body}


class GuardMiddleware:
    """Pure ASGI (streaming- and SSE-safe): Host + Origin checks, optional token, security headers."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers") or []}
        host = headers.get("host", "")
        if not host_allowed(host):
            log.warning("Refused request for Host %r (set FOVEA_ALLOWED_HOSTS to allow it)", host[:100])
            return await self._reply(send, 400, "Invalid host header. Fovea only answers to localhost, IP addresses, "
                                                "this computer's name and the names in FOVEA_ALLOWED_HOSTS.")
        method = scope.get("method", "GET")
        reason = cross_site_reason(method, headers, scope.get("scheme", "http"))
        if reason:
            log.warning("Refused %s %s: %s", method, scope.get("path", ""), reason)
            return await self._reply(send, 403, f"Forbidden: {reason}.")
        token = access_token()
        set_cookie: Optional[str] = None
        if token and not is_local_request(scope):
            qs = scope.get("query_string", b"").decode("latin-1")
            params = parse_qsl(qs, keep_blank_values=True)
            given_q = next((v for k, v in params if k == "token"), "")
            auth = headers.get("authorization", "")
            given_h = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
            given_c = _cookie(headers, TOKEN_COOKIE)
            cookie = f"{TOKEN_COOKIE}={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age=2592000"
            if _token_ok(given_q, token) and method == "GET":
                # Exchange the query token for a cookie and drop it from the address bar (and history).
                return await self._redirect(send, _self_location(scope, params), cookie)
            if not (_token_ok(given_h, token) or _token_ok(given_c, token)):
                page = not scope.get("path", "").startswith("/api/") and "text/html" in headers.get("accept", "")
                form = headers.get("content-type", "").split(";")[0].strip().lower() == "application/x-www-form-urlencoded"
                if page and method == "POST" and form:
                    # The token page's own form (same origin: the Origin check above passed). The body is consumed
                    # here and never reaches the app; success lands on the very URL the visitor opened.
                    if _token_ok(await _form_token(receive), token):
                        return await self._redirect(send, _self_location(scope, params), cookie)
                    return await self._html(send, 401, _login_page(_self_location(scope, params), True))
                if page and method == "GET":
                    return await self._html(send, 401, _login_page(_self_location(scope, params), bool(given_q)))
                return await self._reply(send, 401, "This server requires an access token (FOVEA_TOKEN).")

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                h = MutableHeaders(scope=message)
                h.setdefault("X-Content-Type-Options", "nosniff")
                h.setdefault("Referrer-Policy", "same-origin")
                h.setdefault("X-Frame-Options", "DENY")
                h.setdefault("Content-Security-Policy", "frame-ancestors 'none'")
                if set_cookie:
                    h.append("Set-Cookie", set_cookie)
            await send(message)

        await self.app(scope, receive, send_with_headers)

    @staticmethod
    async def _reply(send, status: int, detail: str):
        start, body = _json_response(status, detail, [(b"x-content-type-options", b"nosniff")])
        await send(start)
        await send(body)

    @staticmethod
    async def _html(send, status: int, html: str):
        body = html.encode()
        await send({"type": "http.response.start", "status": status, "headers": [
            (b"content-type", b"text/html; charset=utf-8"), (b"content-length", str(len(body)).encode()),
            (b"cache-control", b"no-store"), (b"x-frame-options", b"DENY"),
            (b"content-security-policy", b"default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'")]})
        await send({"type": "http.response.body", "body": body})

    @staticmethod
    async def _redirect(send, location: str, cookie: str):
        await send({"type": "http.response.start", "status": 303, "headers": [
            (b"location", location.encode("latin-1", "replace")), (b"set-cookie", cookie.encode("latin-1")),
            (b"cache-control", b"no-store"), (b"content-length", b"0")]})
        await send({"type": "http.response.body", "body": b""})


def server_info(request: Any = None) -> dict:
    """What the UI shows in its remote-access note (never any secret)."""
    return {
        "bind_host": bind_host() or None,
        "loopback_only": loopback_only(),
        "request_is_local": is_local_request(request) if request is not None else None,
        "remote_ai_allowed": remote_ai_allowed(),
        "token_required": bool(access_token()),
        "allowed_hosts": [h for h in _ALLOWED_HOSTS if h != "*"],
    }
