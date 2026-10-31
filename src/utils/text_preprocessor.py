"""
Text Preprocessor Module for Indonesian Government App Reviews (IGAR)
Integrates Member 1's exact text cleaning & slang normalization pipeline.
Preserves all negation words and sentiment markers without alteration.
"""

import re
import unicodedata

# Exact slang and abbreviation normalization dictionary from Member 1's preprocessing
SLANG_DICT = {
    # Negasi (krusial untuk polaritas sentimen)
    'tdk': 'tidak', 'gak': 'tidak', 'ga': 'tidak', 'gk': 'tidak',
    'nggak': 'tidak', 'ngga': 'tidak', 'engga': 'tidak', 'enggak': 'tidak',
    'gx': 'tidak', 'tida': 'tidak', 'tdak': 'tidak',
    'jgn': 'jangan', 'jng': 'jangan',
    'blm': 'belum', 'blum': 'belum',
    
    # Kata hubung & preposisi ('dr' dipertahankan untuk mencegah konflik 'dokter')
    'yg': 'yang', 'dgn': 'dengan', 'dg': 'dengan', 'dng': 'dengan',
    'utk': 'untuk', 'krn': 'karena', 'krna': 'karena', 'karna': 'karena',
    'tp': 'tapi', 'tpi': 'tapi',
    'dri': 'dari',
    'sm': 'sama', 'dlm': 'dalam',
    'spt': 'seperti', 'sprti': 'seperti',
    'stlh': 'setelah', 'sblm': 'sebelum',
    'klo': 'kalau', 'kalo': 'kalau', 'klu': 'kalau',
    
    # Kata kerja umum
    'bs': 'bisa', 'bsa': 'bisa',
    'sdh': 'sudah', 'udh': 'sudah', 'udah': 'sudah',
    'hrs': 'harus', 'hrus': 'harus',
    'dpt': 'dapat',
    'pke': 'pakai', 'pake': 'pakai',
    'smpe': 'sampai', 'sampe': 'sampai',
    'trs': 'terus', 'trus': 'terus',
    'cb': 'coba',
    
    # Kata sifat & intensitas sentimen
    'bgus': 'bagus', 'bgs': 'bagus',
    'jlek': 'jelek', 'jlk': 'jelek',
    'bner': 'benar', 'bnr': 'benar',
    'lmyn': 'lumayan', 'lumyan': 'lumayan',
    'mantep': 'mantap', 'mantab': 'mantap',
    'bgt': 'banget', 'bngt': 'banget',
    'krg': 'kurang',
    'bnyk': 'banyak', 'byk': 'banyak',
    
    # Pronomina (diselaraskan ke 'saya' dan 'kamu')
    'sy': 'saya', 'ak': 'saya', 'aku': 'saya', 'gue': 'saya', 'gw': 'saya',
    'lu': 'kamu', 'km': 'kamu',
    'org': 'orang', 'orng': 'orang',
    
    # Adverbia & kata umum
    'jg': 'juga', 'lg': 'lagi',
    'msh': 'masih',
    'skrg': 'sekarang', 'skr': 'sekarang',
    'srng': 'sering', 'sllu': 'selalu',
    'emg': 'memang', 'emang': 'memang',
    'aja': 'saja', 'doang': 'saja',
    'cmn': 'cuma',
    'gmn': 'bagaimana', 'gmna': 'bagaimana', 'gimana': 'bagaimana',
    'knp': 'kenapa',
    'mgkn': 'mungkin', 'mngkn': 'mungkin',
    'pdhl': 'padahal', 'mslh': 'masalah',
    'mksh': 'terima kasih', 'makasih': 'terima kasih',
    'thx': 'terima kasih', 'tq': 'terima kasih', 'trims': 'terima kasih',
    'aplk': 'aplikasi', 'apk': 'aplikasi',
    'blh': 'boleh',
    
    # Reduksi varian elongasi umum
    'baguus': 'bagus', 'jelekk': 'jelek', 'mantapp': 'mantap',
    'kerenn': 'keren', 'okee': 'oke', 'parahh': 'parah',
    'lemott': 'lemot', 'susahh': 'susah', 'lambatt': 'lambat',
    'payahh': 'payah'
}


def clean_text(text: str) -> str:
    """
    Applies Member 1's exact text cleaning pipeline:
    1. Unicode decomposition (NFKD) to convert stylized/math fonts to standard Latin.
    2. Case folding to lowercase.
    3. Remove URLs (http, https, www).
    4. Remove Emails.
    5. Remove non-alphanumeric characters (regex: [^a-z0-9\s]), removing emojis & symbols.
    6. De-elongate characters (3+ consecutive duplicate chars reduced to 2).
    7. Normalize Indonesian slang words using SLANG_DICT.
    8. Normalize multiple whitespaces and strip boundaries.
    """
    # Member 1's notebook returns an empty string for missing and non-string
    # values. Checking the type first also keeps this safe for NaN/pd.NA.
    if not isinstance(text, str):
        return ""
    
    # 1. Unicode decomposition (NFKD)
    t = unicodedata.normalize('NFKD', text)
    
    # 2. Case folding
    t = t.lower()
    
    # 3. Remove URLs
    t = re.sub(r'https?://\S+|www\.\S+', ' ', t)
    
    # 4. Remove Email addresses
    t = re.sub(r'\S+@\S+\.\S+', ' ', t)
    
    # 5. Remove non-alphanumeric characters (strips emojis directly)
    t = re.sub(r'[^a-z0-9\s]', ' ', t)
    
    # 6. Reduce character elongation (3+ to 2)
    t = re.sub(r'(.)\1{2,}', r'\1\1', t)
    
    # 7. Normalize slang & colloquial forms
    words = t.split()
    words = [SLANG_DICT.get(w, w) for w in words]
    t = ' '.join(words)
    
    # 8. Normalize whitespace
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def analyze_text_features(raw_text: str):
    """
    Extracts text statistics and slang substitutions for dashboard explanation.
    """
    cleaned = clean_text(raw_text)
    raw_words = raw_text.split()
    clean_words = cleaned.split()
    
    detected_slang = {}
    for word in raw_words:
        w_lower = re.sub(r'[^a-z0-9]', '', word.lower())
        if w_lower in SLANG_DICT:
            detected_slang[w_lower] = SLANG_DICT[w_lower]
            
    return {
        "raw_char_count": len(raw_text),
        "raw_word_count": len(raw_words),
        "cleaned_text": cleaned,
        "clean_word_count": len(clean_words),
        "detected_slang": detected_slang
    }
