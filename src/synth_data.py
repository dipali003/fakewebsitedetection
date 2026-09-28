"""Deterministic synthetic phishing/legitimate corpora.

Used when no real dataset is supplied in ``data/raw``.  Everything is seeded,
so runs are reproducible.  The generators mimic the statistical quirks that
char-CNN / BiLSTM models actually learn from:

* phishing URLs  → brand impersonation, homoglyphs, punycode, hyphens,
  embedded "@", IP literals, suspicious TLDs, long random paths,
  credential keywords ("login", "verify", "secure" …).
* phishing mails → urgency, threats, generic greetings, "verify your
  account", odd payloads, spoofed display names.
"""
from __future__ import annotations

import random
import string

import pandas as pd

# -------------------------------------------------------------------------- vocab
LEGIT_DOMAINS = [
    "google.com", "youtube.com", "amazon.com", "facebook.com", "wikipedia.org",
    "apple.com", "microsoft.com", "github.com", "stackoverflow.com",
    "nytimes.com", "bbc.co.uk", "cnn.com", "linkedin.com", "netflix.com",
    "paypal.com", "chase.com", "bankofamerica.com", "wellsfargo.com",
    "dropbox.com", "adobe.com", "ebay.com", "target.com", "walmart.com",
    "instagram.com", "twitter.com", "reddit.com", "quora.com", "medium.com",
    "harvard.edu", "mit.edu", "stanford.edu", "nasa.gov", "who.int",
    "python.org", "mozilla.org", "ubuntu.com", "canon.com", "nike.com",
    "spotify.com", "zoom.us", "icloud.com", "office.com", "live.com",
    "booking.com", "expedia.com", "tripadvisor.com", "airbnb.com",
]

PHISH_BRANDS = [
    "paypal", "apple", "amazon", "netflix", "microsoft", "office365",
    "outlook", "gmail", "facebook", "instagram", "whatsapp", "linkedin",
    "chase", "wellsfargo", "hsbc", "barclays", "icici", "hdfcbank",
    "sbi", "axisbank", "coinbase", "binance", "dhl", "fedex", "usps",
    "irs", "hmrc", "aadhaar", "steam", "roblox", "dropbox", "zoom",
]

HOMOGLYPHS = {"a": "а", "e": "е", "o": "о", "i": "і", "u": "u", "c": "с",
              "m": "m", "n": "n", "p": "р", "x": "х", "y": "у", "s": "ѕ"}

LEGIT_PATHS = [
    "", "about", "contact", "help", "search", "products", "pricing",
    "blog", "docs", "guides", "careers", "press", "terms", "privacy",
    "shop/item", "questions/tagged", "watch", "article", "en-us",
]

PHISH_PATH_WORDS = [
    "login", "verify", "secure", "account", "update", "confirm", "signin",
    "webscr", "billing", "payment", "unlock", "suspended", "limited",
    "recover", "restore", "wallet", "seed", "support", "case", "id",
    "session", "expired", "password", "reset", "activity", "alert",
]

SUSPICIOUS_TLDS = ["tk", "ml", "ga", "cf", "gq", "xyz", "top", "club", "online",
                   "site", "icu", "buzz", "click", "link", "rest", "cyou"]
NORMAL_TLDS = ["com", "org", "net", "edu", "gov", "co", "io", "co.uk", "us", "in"]

COMPANIES = ["ACME Corp", "TechNova", "GlobalBank", "CloudNine", "DataWorks",
             "BrightPath", "SkyLine Logistics", "MediCare Plus", "EduLearn",
             "UrbanEats", "SwiftShip", "GreenEnergy Co"]

NAMES = ["James", "Maria", "David", "Sarah", "Michael", "Laura", "Raj",
         "Chen", "Aisha", "Tom", "Emma", "Alex", "Priya", "John"]


def _apply_homoglyph(word: str, rng: random.Random, p: float = 0.5) -> str:
    out = []
    for ch in word:
        low = ch.lower()
        if low in HOMOGLYPHS and rng.random() < p:
            out.append(HOMOGLYPHS[low])
        else:
            out.append(ch)
    return "".join(out)


def _rand_token(rng: random.Random, lo: int = 6, hi: int = 14) -> str:
    return "".join(rng.choice(string.ascii_lowercase + string.digits)
                   for _ in range(rng.randint(lo, hi)))


def _rand_hex(rng: random.Random, n: int = 8) -> str:
    return "".join(rng.choice("0123456789abcdef") for _ in range(n))


# -------------------------------------------------------------------------- URLs
def _legit_url(rng: random.Random) -> str:
    domain = rng.choice(LEGIT_DOMAINS)
    scheme = "https" if rng.random() < 0.8 else "http"
    sub = rng.choice(["", "www.", "www.", "mail.", "docs.", "blog.", "support."])
    path = rng.choice(LEGIT_PATHS)
    if path and rng.random() < 0.5:
        path += "/" + _rand_token(rng, 4, 10)
    query = ""
    if rng.random() < 0.3:
        query = "?" + rng.choice(["q", "id", "page", "ref", "utm_source"]) + "=" + _rand_token(rng, 3, 8)
    return f"{scheme}://{sub}{domain}/{path}{query}"


def _phish_url(rng: random.Random) -> str:
    brand = rng.choice(PHISH_BRANDS).strip()
    tld = rng.choice(SUSPICIOUS_TLDS) if rng.random() < 0.7 else rng.choice(NORMAL_TLDS)
    scheme = "http" if rng.random() < 0.75 else "https"
    style = rng.randrange(7)
    kw1 = rng.choice(PHISH_PATH_WORDS)
    kw2 = rng.choice(PHISH_PATH_WORDS)

    if style == 0:      # brand as subdomain of an unrelated domain
        host = f"{brand}.{_rand_token(rng)}.{tld}"
        url = f"{scheme}://{host}/{kw1}-{kw2}/{_rand_hex(rng)}"
    elif style == 1:    # brand embedded mid-domain with hyphens
        host = f"www-{brand}{rng.choice(['-secure', '-verify', '-support', '-account', ''])}.{tld}"
        url = f"{scheme}://{host}/{kw1}/{kw2}.html?user={_rand_token(rng, 4, 8)}"
    elif style == 2:    # homoglyph typosquat
        host = _apply_homoglyph(f"{brand}.com", rng)
        url = f"{scheme}://{host}/{kw1}/{kw2}"
    elif style == 3:    # brand + keyword compound TLD, long path
        host = f"{brand}-{kw1}.{tld}"
        url = f"{scheme}://{host}/{kw2}/{_rand_token(rng, 10, 18)}/index.php"
    elif style == 4:    # "@" trick: real brand in userinfo, junk host after
        real = rng.choice(LEGIT_DOMAINS)
        url = f"{scheme}://{brand}.com@{_rand_token(rng)}.{tld}/{kw1}"
    elif style == 5:    # IP literal host
        ip = ".".join(str(rng.randint(1, 254)) for _ in range(4))
        url = f"{scheme}://{ip}/{brand}/{kw1}/{kw2}.php?token={_rand_hex(rng)}"
    else:               # brand as path on random host + punycode occasionally
        host = f"xn--{_rand_token(rng, 6, 10)}.{tld}" if rng.random() < 0.3 else f"{_rand_token(rng)}.{tld}"
        url = f"{scheme}://{host}/{brand}-{kw1}-{kw2}/verify"

    if rng.random() < 0.35:
        url += rng.choice(["#", "&", "%20", "="])
    return url


def synthetic_url_frame(n: int, seed: int) -> pd.DataFrame:
    """Balanced synthetic URL dataset of ``n`` rows."""
    rng = random.Random(seed)
    half = n // 2
    rows = [( _phish_url(rng), 1) for _ in range(half)] + \
           [(_legit_url(rng), 0) for _ in range(n - half)]
    rng.shuffle(rows)
    return pd.DataFrame(rows, columns=["url", "label"])


# -------------------------------------------------------------------------- emails
def _legit_email(rng: random.Random) -> str:
    company = rng.choice(COMPANIES)
    name = rng.choice(NAMES)
    templates = [
        "Hi {name},\n\nThanks for attending yesterday's meeting. The notes and action items are shared in the drive. Let me know if anything is unclear.\n\nBest regards,\n{sender}\n{company}",
        "Hello,\n\nYour order #{order} has shipped and is expected on {date}. You can track the parcel from your account page.\n\nThank you for shopping with {company}.",
        "Hi {name},\n\nYour monthly statement is now available in the customer portal. No action is required.\n\nRegards,\n{company} Billing Team",
        "Hey {name},\n\nThe project report is attached for review. Could you send feedback by {date}?\n\nThanks,\n{sender}",
        "Dear {name},\n\nWelcome to {company}! Your registration is confirmed. We look forward to seeing you at the upcoming events.\n\nWarm regards,\n{sender}",
        "Hi,\n\nReminder: the maintenance window is scheduled this weekend between 2 and 4 AM. Services may be briefly unavailable.\n\nIT Department\n{company}",
        "Hello {name},\n\nYour reservation for {date} is confirmed. Reply to this email if you need to change anything.\n\n{company} Support",
        "Hi {name},\n\nPlease find the invoice for last month attached. Payment is due within 30 days.\n\nAccounts Team\n{company}",
    ]
    t = rng.choice(templates)
    return t.format(name=name, company=company, sender=rng.choice(NAMES),
                    date=rng.choice(["Monday", "March 14", "next Friday", "the 21st"]),
                    order=str(rng.randint(10000, 99999)))


def _phish_email(rng: random.Random) -> str:
    brand = rng.choice(["PayPal", "Apple", "Amazon", "Netflix", "Microsoft",
                        "Bank of America", "Chase", "DHL", "IRS", "WhatsApp",
                        "Facebook", "Coinbase", "Adobe"]).strip()
    templates = [
        "Dear Customer,\n\nWe detected unusual activity in your {brand} account. Your access has been SUSPENDED. Verify your identity within 24 hours or the account will be permanently closed.\n\nVerify here: {url}\n\n{brand} Security Team",
        "URGENT: Your password will expire today!\n\nUpdate your {brand} password immediately to avoid losing access to your mailbox. Confirm your credentials at:\n{url}\n\nIT Support",
        "Congratulations! You have won a ${amount} gift card from {brand}. Claim your reward now — limited time offer!\n\nClaim: {url}\n\nReply with your full name and phone number.",
        "Your {brand} account has been locked due to a violation of our terms.\n\nTo restore access, confirm your login details and card information at:\n{url}\n\nFailure within 12 hours will result in permanent deletion.",
        "Dear user,\n\nA signin attempt from an unrecognised device (IP {ip}) was blocked. If this was you, ignore this email. Otherwise, secure your {brand} account now:\n{url}",
        "Invoice #{inv} — payment declined.\n\nWe could not process your payment of ${amount}. Update your billing information to avoid service interruption:\n{url}\n\n{brand} Accounts",
        "Important notice from {brand}.\n\nWe have reason to believe your account was compromised. Please review recent activity and verify your information immediately:\n{url}\n\nDo not share this email with anyone.",
        "FINAL REMINDER: Your {brand} subscription could not be renewed.\n\nClick below to update your card and resume service:\n{url}\n\nThank you,\n{brand} Billing",
    ]
    t = rng.choice(templates)
    url = _phish_url(rng)
    return t.format(brand=brand, url=url,
                    amount=str(rng.choice(["50", "100", "250", "500", "1000"])),
                    ip=".".join(str(rng.randint(1, 254)) for _ in range(4)),
                    inv=str(rng.randint(10000, 99999)))


def synthetic_email_frame(n: int, seed: int) -> pd.DataFrame:
    """Balanced synthetic email dataset of ``n`` rows."""
    rng = random.Random(seed + 1)
    half = n // 2
    rows = [(_phish_email(rng), 1) for _ in range(half)] + \
           [(_legit_email(rng), 0) for _ in range(n - half)]
    rng.shuffle(rows)
    return pd.DataFrame(rows, columns=["text", "label"])
