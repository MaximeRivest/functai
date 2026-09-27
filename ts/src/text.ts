/**
 * Text rules the contract spells out (contract/scores.md): which characters
 * are white space, and Unicode full case folding. JavaScript's own `\s`,
 * `trim()` and `toLowerCase()` differ from both, so they are not used here.
 */

import { CASEFOLD } from "./generated/contract.ts";

/** White space: what Python's `str.split()` and `str.strip()` treat as white space. */
const WHITE = new Set<number>([
  0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x85, 0xa0, 0x1680,
  0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200a,
  0x2028, 0x2029, 0x202f, 0x205f, 0x3000,
]);

export function isWhite(codePoint: number): boolean {
  return WHITE.has(codePoint);
}

/** The pieces of `text` between runs of white space. */
export function splitWhite(text: string): string[] {
  const out: string[] = [];
  let word = "";
  for (const ch of text) {
    if (WHITE.has(ch.codePointAt(0)!)) {
      if (word) out.push(word);
      word = "";
    } else {
      word += ch;
    }
  }
  if (word) out.push(word);
  return out;
}

/** `text` without white space at either end. */
export function trimWhite(text: string): string {
  const chars = Array.from(text);
  let a = 0;
  let b = chars.length;
  while (a < b && WHITE.has(chars[a]!.codePointAt(0)!)) a++;
  while (b > a && WHITE.has(chars[b - 1]!.codePointAt(0)!)) b--;
  return chars.slice(a, b).join("");
}

/** Unicode full case folding, code point by code point (Python's `str.casefold`). */
export function casefold(text: string): string {
  let out = "";
  for (const ch of text) {
    const folded = CASEFOLD[ch.codePointAt(0)!.toString(16).toUpperCase().padStart(4, "0")];
    out += folded ?? ch;
  }
  return out;
}

/** Text as exact_match compares it: white space collapsed, case folded. */
export function normalize(text: string): string {
  return casefold(splitWhite(text).join(" "));
}
