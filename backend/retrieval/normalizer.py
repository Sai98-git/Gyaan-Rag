import sys
import os
import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Comprehensive Multilingual Indic-English Term Mapping Dictionary
INDIC_TRANSLITERATION_MAP = {
    # Business / Corporate
    "corporation": ["कॉर्पोरेशन", "निगम", "कंपनी"],
    "corporations": ["कॉर्पोरेशन", "निगम", "कंपनियों", "निगमों"],
    "corporate": ["कॉर्पोरेट", "निगमित", "व्यावसायिक"],
    "company": ["कंपनी", "निगम", "संस्था"],
    "companies": ["कंपनियां", "कंपनियों", "निगमों"],
    "shareholder": ["शेयरधारक", "शेयरधारकों", "स्वामित्व", "हिस्सेदार"],
    "shareholders": ["शेयरधारक", "शेयरधारकों", "हिस्सेदारों"],
    "shareholding": ["शेयरधारिता", "स्वामित्व"],
    "stakeholder": ["हितधारक", "साझेदार"],
    "stakeholders": ["हितधारकों", "साझेदारों"],
    "ownership": ["स्वामित्व", "मालिकाना"],
    "owner": ["मालिक", "स्वामी"],
    "owners": ["मालिकों", "स्वामियों"],
    "profit": ["लाभ", "मुनाफा"],
    "profits": ["लाभ", "मुनाफे"],
    "dividend": ["लाभांश"],
    "dividends": ["लाभांश"],
    "revenue": ["राजस्व", "आय"],
    "income": ["आय", "कमाई"],
    "business": ["व्यवसाय", "व्यापार", "कारोबार"],
    "enterprise": ["उद्यम", "कंपनी"],
    
    # B-Corp / Standards / Certification
    "b corp": ["बी कॉर्प", "प्रमाणित बी कोर", "बी कार्पोरेशन"],
    "b-corp": ["बी कॉर्प", "प्रमाणित बी कोर", "बी कार्पोरेशन"],
    "bcorp": ["बी कॉर्प", "प्रमाणित बी कोर"],
    "certification": ["प्रमाणन", "प्रमाणित", "सर्टिफिकेशन"],
    "certified": ["प्रमाणित", "मान्यता प्राप्त"],
    "community": ["समुदाय", "समाज"],
    "sustainability": ["स्थिरता", "पर्यावरण"],
    
    # Legal / Governance
    "legal": ["कानूनी", "वैधानिक", "न्यायिक"],
    "entity": ["इकाई", "अस्तित्व", "संस्था"],
    "entities": ["इकाइयां", "संस्थाएं"],
    "law": ["कानून", "विधि"],
    "rights": ["अधिकार", "हक"],
    "right": ["अधिकार", "हक"],
    "contractor": ["ठेकेदार", "अनुबंधक"],
    "contractors": ["ठेकेदार", "ठेकेदारों"],
    "federal": ["संघीय", "केंद्रीय"],
    "definition": ["परिभाषा", "अर्थ"],
    "define": ["परिभाषित", "परिभाषा"],
    "characteristic": ["विशेषता", "लक्षण"],
    "characteristics": ["विशेषताएं", "लक्षण"],
    
    # Brands / Known Entities in Corpus
    "mcdonalds": ["मैकडॉनल्ड्स", "मैकडॉनल्ड"],
    "mcdonald's": ["मैकडॉनल्ड्स", "मैकडॉनल्ड"],
    "mcdonald": ["मैकडॉनल्ड", "मैकडॉनल्ड्स"],
    "phoenix": ["फीनिक्स"],
    "scottsdale": ["स्कॉट्सडेल"],
    
    # Common Hinglish Grammatical Connectors
    "kya": ["क्या"],
    "hai": ["है"],
    "hain": ["हैं"],
    "hota": ["होता"],
    "hoti": ["होती"],
    "hote": ["होते"],
    "ka": ["का"],
    "ke": ["के"],
    "ki": ["की"],
    "ko": ["को"],
    "se": ["से"],
    "mein": ["में"],
    "me": ["में"],
    "par": ["पर"],
    "matlab": ["मतलब", "अर्थ"],
    "batao": ["बताओ", "बताइए"],
    "kise": ["किसे"],
    "kaise": ["कैसे"],
    "kyun": ["क्यों"],
    "kaun": ["कौन"]
}


def expand_indic_query(query: str) -> str:
    """
    Expands an English or Hinglish query with Devanagari equivalents
    to enable high-recall lexical matching against Indic corpora.
    """
    if not query:
        return ""
        
    clean = re.sub(r'[।॥\|!\?\.,;:\(\)\"\']', ' ', query.lower())
    tokens = clean.split()
    expanded = list(tokens)
    
    # Check single tokens
    for tok in tokens:
        if tok in INDIC_TRANSLITERATION_MAP:
            expanded.extend(INDIC_TRANSLITERATION_MAP[tok])
            
    # Check bigrams (e.g., 'b corp', 'legal entity')
    for i in range(len(tokens) - 1):
        bigram = f"{tokens[i]} {tokens[i+1]}"
        if bigram in INDIC_TRANSLITERATION_MAP:
            expanded.extend(INDIC_TRANSLITERATION_MAP[bigram])
            
    # Deduplicate while preserving order
    seen = set()
    result_tokens = []
    for t in expanded:
        if t not in seen:
            seen.add(t)
            result_tokens.append(t)
            
    return " ".join(result_tokens)
