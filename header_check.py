#!/usr/bin/env python3
"""
check_fits_header.py

Validate and dump the header(s) of a FITS file (Flexible Image Transport
System -- the astronomy image/table format, also seen with a .fit extension).

FITS structure recap:
  * The file is a sequence of Header-Data Units (HDUs): one primary HDU
    followed by optional extensions.
  * A header is made of 80-byte ASCII "card images", packed into 2880-byte
    blocks (36 cards per block), and terminated by an "END" card.
  * Value cards have "= " in columns 9-10. Strings are single-quoted,
    logicals are T/F, everything else is numeric. A "/" starts a comment.
  * Mandatory primary keywords, in order: SIMPLE, BITPIX, NAXIS, NAXIS1..n.
    Extensions use XTENSION instead of SIMPLE.
  * The data array (if NAXIS > 0) is |BITPIX|/8 * product(NAXISn) bytes,
    padded up to the next 2880-byte boundary.
"""

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Set the file to check here. Leave it "" to be prompted for a path.
# ---------------------------------------------------------------------------
FITS_PATH = "/Users/none/internship/fit_files/MERIDIAN 8-0001_bin4_3s.fit"

BLOCK = 2880
CARD = 80

BITPIX_MEANING = {
    8:   "8-bit unsigned integer",
    16:  "16-bit signed integer",
    32:  "32-bit signed integer",
    64:  "64-bit signed integer",
    -32: "32-bit IEEE float",
    -64: "64-bit IEEE float",
}


@dataclass
class Card:
    keyword: str
    value: object        # str | int | float | bool | None
    comment: str = ""
    raw: str = ""


@dataclass
class HDU:
    index: int
    cards: list[Card] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def get(self, keyword: str):
        for c in self.cards:
            if c.keyword == keyword:
                return c.value
        return None


def parse_value(rest: str):
    """Parse the part of a card after 'KEYWORD= ' into (value, comment)."""
    s = rest.strip()
    if not s:
        return None, ""

    # String value: single-quoted, with '' as an embedded quote.
    if s.startswith("'"):
        i = 1
        chars = []
        while i < len(s):
            ch = s[i]
            if ch == "'":
                if i + 1 < len(s) and s[i + 1] == "'":
                    chars.append("'")
                    i += 2
                    continue
                i += 1
                break
            chars.append(ch)
            i += 1
        value = "".join(chars).rstrip()
        rest_after = s[i:].lstrip()
        comment = rest_after[1:].strip() if rest_after.startswith("/") else ""
        return value, comment

    # Non-string: split off a trailing "/ comment".
    if "/" in s:
        val_part, comment = s.split("/", 1)
        val_part, comment = val_part.strip(), comment.strip()
    else:
        val_part, comment = s, ""

    if val_part in ("T", "F"):
        return (val_part == "T"), comment

    for conv in (int, float):
        try:
            return conv(val_part), comment
        except ValueError:
            pass

    return val_part, comment  # leave as-is if unrecognized


def parse_card(text: str) -> Card | None:
    keyword = text[:8].strip()
    if not keyword:
        return None
    # Commentary keywords carry free text, not a value.
    if keyword in ("COMMENT", "HISTORY") or text[8:10] != "= ":
        return Card(keyword=keyword, value=None, comment=text[8:].rstrip(), raw=text)
    value, comment = parse_value(text[10:])
    return Card(keyword=keyword, value=value, comment=comment, raw=text)


def data_bytes(hdu: HDU) -> int:
    naxis = hdu.get("NAXIS") or 0
    if not naxis:
        return 0
    bitpix = hdu.get("BITPIX") or 0
    count = 1
    for n in range(1, naxis + 1):
        count *= (hdu.get(f"NAXIS{n}") or 0)
    nbytes = abs(bitpix) // 8 * count
    # Round up to a whole number of 2880-byte blocks.
    return ((nbytes + BLOCK - 1) // BLOCK) * BLOCK


def read_hdus(path: str) -> list[HDU]:
    hdus: list[HDU] = []
    with open(path, "rb") as f:
        index = 0
        while True:
            hdu = HDU(index=index)
            ended = False
            block_count = 0
            while not ended:
                block = f.read(BLOCK)
                if len(block) < BLOCK:
                    if block_count == 0:
                        return hdus  # clean end of file
                    hdu.errors.append("header not terminated by END card")
                    break
                block_count += 1
                for i in range(0, BLOCK, CARD):
                    text = block[i:i + CARD].decode("ascii", "replace")
                    if text[:8].strip() == "END":
                        ended = True
                        break
                    card = parse_card(text)
                    if card:
                        hdu.cards.append(card)
            hdus.append(hdu)
            # Skip over this HDU's data array to reach the next header.
            f.seek(data_bytes(hdu), 1)
            index += 1


def validate(hdu: HDU) -> None:
    first = hdu.cards[0].keyword if hdu.cards else ""
    if hdu.index == 0:
        if first != "SIMPLE":
            hdu.errors.append(f"primary header must start with SIMPLE (got {first!r})")
        elif hdu.get("SIMPLE") is not True:
            hdu.errors.append("SIMPLE is not T (file may not conform to the standard)")
    else:
        if first != "XTENSION":
            hdu.errors.append(f"extension header must start with XTENSION (got {first!r})")
    if hdu.get("BITPIX") not in BITPIX_MEANING:
        hdu.errors.append(f"BITPIX missing or invalid: {hdu.get('BITPIX')!r}")
    if hdu.get("NAXIS") is None:
        hdu.errors.append("NAXIS keyword missing")


def report(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(10)
    except OSError as exc:
        print(f"error: cannot read {path}: {exc}")
        return False

    if head[:6] != b"SIMPLE" or b"=" not in head[6:]:
        print(f"{path}: does not start with 'SIMPLE =' -- not a FITS file")
        print("  first bytes:", " ".join(f"{b:02X}" for b in head))
        return False

    hdus = read_hdus(path)
    print(f"{path}: FITS file with {len(hdus)} HDU(s)\n")

    all_ok = True
    for hdu in hdus:
        validate(hdu)
        kind = "primary" if hdu.index == 0 else (hdu.get("XTENSION") or "extension")
        print(f"HDU {hdu.index} [{kind}] -- {len(hdu.cards)} keywords")

        bitpix = hdu.get("BITPIX")
        naxis = hdu.get("NAXIS") or 0
        print(f"  BITPIX = {bitpix} ({BITPIX_MEANING.get(bitpix, 'unknown')})")
        if naxis:
            dims = " x ".join(str(hdu.get(f"NAXIS{n}")) for n in range(1, naxis + 1))
            print(f"  NAXIS  = {naxis}  -> dimensions {dims}")
        else:
            print(f"  NAXIS  = 0  (header-only, no data array)")

        for key in ("OBJECT", "TELESCOP", "INSTRUME", "DATE-OBS", "EXPTIME", "BUNIT"):
            val = hdu.get(key)
            if val is not None:
                print(f"  {key:<8} = {val}")

        if hdu.errors:
            all_ok = False
            for e in hdu.errors:
                print(f"  ! {e}")
        print()

    print(f"=> {'VALID FITS file' if all_ok else 'FITS file with WARNINGS (see above)'}")
    return all_ok


if __name__ == "__main__":
    p = FITS_PATH.strip() or input("Path to FITS (.fit/.fits) file: ").strip()
    report(p)