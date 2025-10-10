""" 
This class is handling all the preprocessing, and feature engineering 


"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from urllib.parse import urlparse

from sklearn.utils.class_weight import compute_class_weight

import joblib

# checking the load, will have to remove 

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class DataProcessor:

    """basically here we preprocess the feature, extract and """

    def __init__(self):
       # regex specific to the task
        self.url_pattern      = re.compile(r'https?://\S+')
        self.hxxp_pattern     = re.compile(r'hxxp[s]?://\S+')
        self.email_pattern    = re.compile(r'\S+@\S+')
        self.html_tag_pattern = re.compile(r'<[^>]+>')
        self.non_alpha_pattern= re.compile(r'[^a-zA-Z\s]')

       # standard part
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        
        self.label_encoder = LabelEncoder()

        # regex for security pattern
        self.cve_pattern = re.compile(r'CVE-\d{4}-\d{4,}')
        self.version_pattern = re.compile(r'\d+\.\d+(\.\d+)?')
        self.attach_pattern = re.compile(r'\.(pdf|docx?|xlsx?|pptx?)', re.I)  # what does the re.I do ?

        # specific to the task
        self.EU_DOMAINS = {'europa.eu', 'ec.europa.eu', 'cert.europa.eu'} # necessary ? 

    
    def load_data(self, file_path):
        """Load JSONL data into a pandas DataFrame."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return pd.DataFrame(data)
    
    def preprocess_text(self, text):
        
        if pd.isna(text):
            return ""

        # standard cleaning with all the operations for our dataset
        t = text.lower()
        t = self.html_tag_pattern.sub(' ', t)
        t = self.hxxp_pattern.sub(' ', t)
        t = self.url_pattern.sub(' ', t)
        t = self.email_pattern.sub(' ', t)
        t = self.non_alpha_pattern.sub(' ', t)
        # Lemmantize --> so that for example attack and attacks are treated the same way
        tokens = word_tokenize(t)
        clean = [
            self.lemmatizer.lemmatize(tok)
            for tok in tokens
            if tok not in self.stop_words and len(tok) > 1
        ]
        return " ".join(clean)
    
    def url_features(self, text):
        """
        From raw content, count:
        - total URLs
        - disguised 'hxxp' URLs
        - unique domains
        - suspicious TLDs (e.g. info, org, site)
        """
        if pd.isna(text):
            text = ""
        
        urls = self.hxxp_pattern.findall(text) + self.url_pattern.findall(text)
        
        domains = set()
        for u in urls:
            try:
                # Clean up the URL
                cleaned_url = u.replace('hxxp', 'http')
                # Parse and extract domain
                parsed = urlparse(cleaned_url)
                if parsed.netloc:  # Only add if netloc exists
                    domains.add(parsed.netloc.lower())
            except (ValueError, Exception):
                # Skip malformed URLs
                continue
        
        tlds = []
        for d in domains:
            if '.' in d:
                tlds.append(d.split('.')[-1])
        
        return {
            'num_urls': len(urls),
            'num_hxxp': sum(u.startswith('hxxp') for u in urls),
            'unique_domains': len(domains),
            'suspicious_tld_count': sum(t in {'info','org','site'} for t in tlds)
        }

    def pattern_flags(self, text):
        """
        Flag presence/count of:
          - CVE identifiers
          - version numbers
          - attachments (.pdf, .docx, .xlsx)
        """
        return {
            'has_cve':int(bool(self.cve_pattern.search(text))),
            'has_version':int(bool(self.version_pattern.search(text))),
            'num_attachments': len(self.attach_pattern.findall(text)),
        }

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build all derived features:
          cleaned text fields
          length & word‑count stats
          URL‑based signals
          CVE/version/attachment flags
          email‑domain metadata
          temporal buckets
          aggregated security keyword count
        """

        df = df.copy()

        # Clean & combine text
        df['title_clean']   = df['title'].apply(self.preprocess_text)
        df['content_clean'] = df['content'].apply(self.preprocess_text)
        df['combined_text'] = df['title_clean'] + ' ' + df['content_clean']

        # Length & word‑counts
        df['title_len']   = df['title_clean'].str.len()
        df['content_len'] = df['content_clean'].str.len()
        df['title_wc']    = df['title_clean'].str.split().str.len()
        df['content_wc']  = df['content_clean'].str.split().str.len()

        # URL features
        url_df = df['content'].apply(self.url_features).apply(pd.Series)
        df = pd.concat([df, url_df], axis=1)

        # Pattern flags
        pat_df = df['content'].apply(self.pattern_flags).apply(pd.Series)
        df = pd.concat([df, pat_df], axis=1)

        # Email‑domain signals
        domains = df['email_address'].str.split('@').str[1].fillna('unknown').str.lower()
        df['email_domain'] = domains
        df['is_internal']  = domains.isin(self.EU_DOMAINS).astype(int)
        df['domain_tld']   = domains.str.split('.').str[-1].fillna('unknown')

        # Temporal buckets
        dt = pd.to_datetime(df['created_date'], errors='coerce')
        df['hour']            = dt.dt.hour.fillna(-1).astype(int)
        df['day_of_week']     = dt.dt.dayofweek.fillna(-1).astype(int)
        df['is_weekend']      = dt.dt.dayofweek.isin([5,6]).astype(int)
        df['is_business_hr']  = dt.dt.hour.between(8,18).astype(int)

        # Aggregate security keyword density
        keywords = [
            'phishing','malware','virus','attack','breach','vulnerability',
            'exploit','incident','threat','fraud','scam','password','login',
            'authentication','firewall','antivirus','encryption','certificate',
            'ssl','tls','vpn','backup','patch','admin','privilege','access',
            'audit','log','monitoring','detection','prevention','response'
        ]
        df['security_kw_count'] = df['combined_text'].apply(
            lambda txt: sum(txt.count(kw) for kw in keywords)
        )

        return df

    def prepare_features(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Prepare features for model training/prediction."""
        text_features = df['combined_text'].values
        

        ## still to investigate that part 
        numerical_features = [
            'title_len', 'content_len', 'title_wc', 'content_wc',
            'num_urls', 'num_hxxp', 'unique_domains', 'suspicious_tld_count',
            'has_cve', 'has_version', 'num_attachments', 'is_internal',
            'hour', 'day_of_week', 'is_weekend', 'is_business_hr', 'security_kw_count'
        ]
        
        X_numerical = df[numerical_features].fillna(0).values
        
        y = None
        if 'assigned_queue' in df.columns:
            if is_training:
                y = self.label_encoder.fit_transform(df['assigned_queue'].values)
            else:
                y = self.label_encoder.transform(df['assigned_queue'].values)
        
        return text_features, X_numerical, y
    

    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and validation sets."""
        return train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['assigned_queue'])

    def get_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Calculate class weights for imbalanced dataset.
        
        goal: give higher weights to underrpresented classes 
        """
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes,weights))
    
    def save_processor(self, filepath: str):
        """Save the processor state."""
        processor_state = {
        'label_encoder': self.label_encoder,
        'lemmatizer': self.lemmatizer,
        'stop_words': self.stop_words
        }   
        joblib.dump(processor_state, filepath)
    
    def load_processor(self, filepath: str):
        processor_state = joblib.load(filepath)
        self.label_encoder = processor_state['label_encoder']
        self.lemmatizer = processor_state['lemmatizer']
        self.stop_words = processor_state['stop_words']





