"""Preprocesamiento comun para documentos y consultas.

Decisiones: HTML/URL fuera, minusculas, tokens alfanumericos, stopwords inglesas
y stemming de Porter. No requiere descargar recursos adicionales de NLTK.
"""
import html
import re

from nltk.stem import PorterStemmer


STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "could", "d", "did", "do",
    "does", "doing", "don", "down", "during", "each", "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself",
    "just", "ll", "m", "ma", "me", "might", "more", "most", "must", "my", "myself",
    "need", "no", "nor", "not", "now", "o", "of", "off", "on", "once", "only", "or",
    "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same",
    "shan", "she", "should", "so", "some", "such", "t", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "ve", "very", "was", "we", "were",
    "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with",
    "won", "would", "y", "you", "your", "yours", "yourself", "yourselves",
}

STEMMER = PorterStemmer()
URL_RE = re.compile(r"https?://\S+|www\.\S+")
TOKEN_RE = re.compile(r"[a-z0-9]+")


def preprocess(text: str) -> list[str]:
    """Convierte texto crudo en terminos normalizados."""
    normalized = URL_RE.sub(" ", html.unescape(text or "")).lower()
    tokens = (token for token in TOKEN_RE.findall(normalized) if len(token) > 1)
    return [STEMMER.stem(token) for token in tokens if token not in STOPWORDS]

