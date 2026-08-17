"""
Name: cite_sources.py
Author: OpenAI Codex
Date: 08/16/2026
Description: An IEEE citation generator.
"""

import datetime


def format_authors_ieee(authors):
    if not authors:
        return ""

    formatted = []
    for author in authors:
        if isinstance(author, str):
            formatted.append(author)
            continue

        first, last = author
        initials = " ".join(f"{name[0]}." for name in first.split())
        formatted.append(f"{initials} {last}")

    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"
    if len(formatted) == 3:
        return f"{formatted[0]}, {formatted[1]}, and {formatted[2]}"
    return f"{formatted[0]} et al."


def generate_ieee_book(source):
    prefix = f"[{source['index']}]\t"
    authors = format_authors_ieee(source.get("authors"))
    author_str = f"{authors}, " if authors else ""
    loc_str = f"{source.get('location')}: " if source.get("location") else ""
    pub_str = f"{source.get('publisher')}, " if source.get("publisher") else ""
    year_str = f"{source.get('pub_year')}." if source.get("pub_year") else ""

    return f"{prefix}{author_str}*{source['title']}*. {loc_str}{pub_str}{year_str}"


def generate_ieee_website(source):
    prefix = f"[{source['index']}]\t"
    authors = format_authors_ieee(source.get("authors"))
    author_str = f"{authors}, " if authors else ""
    site_str = f" in *{source.get('container_title')}*," if source.get("container_title") else ""

    if source.get("pub_date"):
        dt = datetime.date(*source["pub_date"])
        date_str = f" {dt.strftime('%b. %d, %Y')},"
    elif source.get("pub_year"):
        date_str = f" {source['pub_year']},"
    else:
        date_str = ""

    url_str = f" [Online]. Available: {source.get('url')}" if source.get("url") else " [Online]."
    access_date = datetime.date.today().strftime("%b. %d, %Y")
    access_str = f" [Accessed: {access_date}]."

    return f"{prefix}{author_str}\"{source['title']},\"{site_str}{date_str}{url_str}{access_str}"


def generate_ieee_source(source):
    source_type = source.get("source_type", "").lower()
    if source_type == "book":
        return generate_ieee_book(source)
    if source_type in ("website", "web"):
        return generate_ieee_website(source)
    raise ValueError(f"Unsupported source type: {source.get('source_type')}")


def print_sources(source_list):
    print("--- IEEE REFERENCE LIST ---")
    for source in sorted(source_list, key=lambda item: item.get("index", 0)):
        print(generate_ieee_source(source))