"""Authenticate with USGS EarthExplorer (ERS) and save session cookies.

USGS LandsatLook asset downloads require an ERS account.  Running this script
once saves a Netscape-format cookie file that GDAL (and therefore rasterio) can
use for subsequent HTTP requests — no credentials are stored, only the session
cookie.

Usage
-----
    python -m itslive_cloudfree.usgs_login
    python -m itslive_cloudfree.usgs_login --username myuser --cookie-file ~/.usgs_cookies.txt

After the script succeeds it prints the environment variables you need to set
so that GDAL/rasterio picks up the cookies automatically:

    export GDAL_HTTP_COOKIEFILE=~/.usgs_landsat_cookies.txt
    export GDAL_HTTP_COOKIEJAR=~/.usgs_landsat_cookies.txt
"""

from __future__ import annotations

import argparse
import getpass
import http.cookiejar
import re
import sys
from pathlib import Path

import requests

_LOGIN_URL = "https://ers.cr.usgs.gov/login"
_DEFAULT_COOKIE_FILE = Path.home() / ".usgs_landsat_cookies.txt"

# A URL that requires auth — used to verify the cookie works.
_TEST_URL = (
    "https://landsatlook.usgs.gov/data/collection02/level-1/standard/oli-tirs/"
    "2020/224/115/LC08_L1GT_224115_20200111_20201016_02_T2/"
    "LC08_L1GT_224115_20200111_20201016_02_T2_B4.TIF"
)


def _get_csrf(session: requests.Session) -> str:
    """Fetch the ERS login page and extract the CSRF token."""
    r = session.get(_LOGIN_URL, timeout=15)
    r.raise_for_status()
    m = re.search(r'<input[^>]+name=["\']csrf["\'][^>]+value=["\'](.*?)["\']', r.text)
    if not m:
        raise RuntimeError("Could not find CSRF token on ERS login page.")
    return m.group(1)


def login(username: str, password: str, cookie_file: Path) -> None:
    """Log in to USGS ERS and write cookies to *cookie_file*.

    Parameters
    ----------
    username, password:
        ERS account credentials.
    cookie_file:
        Destination for the Netscape-format cookie file.

    Raises
    ------
    RuntimeError
        If login fails or the resulting cookie does not grant access.
    """
    session = requests.Session()
    session.headers["User-Agent"] = "itslive-cloudfree/0.1"

    csrf = _get_csrf(session)

    resp = session.post(
        _LOGIN_URL,
        data={"username": username, "password": password, "csrf": csrf},
        allow_redirects=True,
        timeout=15,
    )
    resp.raise_for_status()

    # ERS redirects to the home page on success; a failed login stays on /login.
    if resp.url.rstrip("/").endswith("/login"):
        raise RuntimeError(
            "Login failed — check your username and password.\n"
            "Register at https://ers.cr.usgs.gov/register if you don't have an account."
        )

    # Verify the cookie actually grants access to a protected asset.
    check = session.head(_TEST_URL, allow_redirects=True, timeout=15)
    if "ers.cr.usgs.gov" in check.url:
        raise RuntimeError(
            "Authentication succeeded but the cookie does not grant access to "
            "LandsatLook assets.  Your account may not have bulk-download privileges.\n"
            "Visit https://ers.cr.usgs.gov/profile and ensure bulk download is enabled."
        )

    # Write cookies in Netscape format so GDAL can read them.
    jar = http.cookiejar.MozillaCookieJar(str(cookie_file))
    for c in session.cookies:
        jar.set_cookie(http.cookiejar.Cookie(
            version=0,
            name=c.name,
            value=c.value,
            port=None,
            port_specified=False,
            domain=c.domain or ".usgs.gov",
            domain_specified=bool(c.domain),
            domain_initial_dot=c.domain.startswith(".") if c.domain else True,
            path=c.path or "/",
            path_specified=bool(c.path),
            secure=c.secure,
            expires=c.expires,
            discard=c.expires is None,
            comment=None,
            comment_url=None,
            rest={},
        ))
    cookie_file.parent.mkdir(parents=True, exist_ok=True)
    jar.save(ignore_discard=True, ignore_expires=True)
    cookie_file.chmod(0o600)

    print(f"Login successful.  Cookies saved to: {cookie_file}")
    print()
    print("Set these environment variables so GDAL/rasterio uses the cookies:")
    print()
    print(f'    export GDAL_HTTP_COOKIEFILE="{cookie_file}"')
    print(f'    export GDAL_HTTP_COOKIEJAR="{cookie_file}"')
    print()
    print("Or pass them at runtime:")
    print()
    print("    import rasterio")
    print(f'    with rasterio.Env(GDAL_HTTP_COOKIEFILE="{cookie_file}",')
    print(f'                      GDAL_HTTP_COOKIEJAR="{cookie_file}"):')
    print("        ...")


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="itslive-usgs-login",
        description=(
            "Log in to USGS EarthExplorer (ERS) and save session cookies so that "
            "GDAL/rasterio can download LandsatLook assets without prompting for "
            "credentials on every request."
        ),
    )
    p.add_argument("--username", "-u", default=None, help="ERS username (prompted if omitted).")
    p.add_argument(
        "--cookie-file",
        default=str(_DEFAULT_COOKIE_FILE),
        metavar="PATH",
        help=f"Where to write the cookie file (default: {_DEFAULT_COOKIE_FILE}).",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    username = args.username or input("USGS ERS username: ").strip()
    if not username:
        print("ERROR: username is required.", file=sys.stderr)
        sys.exit(1)

    password = getpass.getpass(f"Password for {username}: ")
    if not password:
        print("ERROR: password is required.", file=sys.stderr)
        sys.exit(1)

    try:
        login(username, password, Path(args.cookie_file))
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
