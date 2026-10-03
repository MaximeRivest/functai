/**
 * Checking a judge's evidence. A judge written as an ordinary AI function (a
 * 0–10 score with the quotes it rests on) is only as good as its quotes: a
 * judge that invents evidence is caught by checking that every quote is
 * really in the text.
 *
 * ```ts
 * const v = await judge({ answer, source });
 * quotesFound(source, v.quotes);          // [true, true, false]: the third is invented
 * ```
 */

import { casefold } from "./text.ts";

const SAME: Record<string, string> = {
  "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'", "`": "'", "\u00b4": "'",
  "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"', "\u2033": '"', "\u00ab": '"', "\u00bb": '"',
  "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2015": "-", "\u2212": "-",
  "\u2026": "...", "\u00a0": " ",
};
const EDGES = new Set([..." \t\n\"'.,;:!?…“”‘’«»()[]"]);

/** Text as the check compares it: Unicode compatibility form, case folded, curly quotes straight, dashes one dash, white space one space. */
function plain(text: string): string {
  const mapped = [...String(text).normalize("NFKC")].map((ch) => SAME[ch] ?? ch).join("");
  return casefold(mapped.replace(/\s+/gu, " ").trim());
}

function strip(text: string): string {
  const chars = [...text];
  let a = 0;
  let b = chars.length;
  while (a < b && EDGES.has(chars[a]!)) a++;
  while (b > a && EDGES.has(chars[b - 1]!)) b--;
  return chars.slice(a, b).join("");
}

const found = (haystack: string, quote: string) => {
  const needle = strip(plain(quote));
  return needle.length > 0 && haystack.includes(needle);
};

/**
 * Whether each quote is in the text, word for word. White space, case, curly
 * and straight quotes, dashes and a quote's own surrounding quotation marks
 * and final punctuation do not count; any other difference does (a changed
 * word, a paraphrase, an invented sentence). Deterministic, and costs nothing.
 * For one quote, whether it is found; for a list, one answer per quote.
 */
export function quotesFound(text: string, quotes: string): boolean;
export function quotesFound(text: string, quotes: Iterable<string>): boolean[];
export function quotesFound(text: string, quotes: string | Iterable<string>): boolean | boolean[] {
  const haystack = plain(text);
  if (typeof quotes === "string") return found(haystack, quotes);
  return [...quotes].map((q) => found(haystack, q));
}
