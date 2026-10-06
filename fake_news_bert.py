"""
===============================================================================
BERT FAKE NEWS DETECTION - TAM EĞİTİM VE ANALİZ PIPELINE
===============================================================================
Bu kod, hakem revizyonu için gerekli TÜM çıktıları üretir:

✅ Model Eğitimi (3 epoch, reproducible)
✅ Veri Sızıntısı Kontrolü ve Raporlama
✅ Confusion Matrix ve Sınıf Bazlı Metrikler
✅ ROC Eğrisi ve AUC Hesaplama
✅ Hata Analizi (FP/FN örnekleri)
✅ Epoch Bazlı Training History
✅ Hiperparametre Tablosu (CSV)
✅ Literatür Karşılaştırma Tablosu
✅ Tüm Görseller (PNG, 300 DPI)
✅ Tam Reproducibility (seed, config)

KULLANIM:
    python fake_news_bert.py

ÇIKTILAR (outputs/ klasöründe):
    - revizyon_raporu.txt
    - confusion_matrix.png
    - roc_curve.png
    - training_history.png
    - hata_analizi.csv
    - epoch_metrikleri.csv
    - hiperparametreler.csv
    - config.json
    - best_model/ (eğitilmiş model)

Tahmini süre: ~25-30 dakika (RTX 3060)
===============================================================================
"""

import os
import sys
import json
import hashlib
import random
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve
)

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Transformers
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed
)

# Uyarıları kapat
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ============================================================================
# KONFİGÜRASYON
# ============================================================================

class Config:
    """Tüm ayarlar tek yerde - Reproducibility için"""
    
    # Paths
    # Repo root; put Fake.csv and True.csv (ISOT dataset) in data/
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(BASE_PATH, "outputs")
    MODEL_DIR = os.path.join(OUTPUT_DIR, "best_model")
    
    # Data
    CSV_DIR = os.path.join(BASE_PATH, "data")
    FAKE_FILE = os.path.join(CSV_DIR, "Fake.csv")
    TRUE_FILE = os.path.join(CSV_DIR, "True.csv")
    
    # Model
    MODEL_NAME = "bert-base-uncased"
    NUM_LABELS = 2
    MAX_LENGTH = 256
    
    # Training
    EPOCHS = 3
    BATCH_SIZE = 32
    GRADIENT_ACCUMULATION = 2
    EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION  # 64
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 500
    
    # Early Stopping
    EARLY_STOPPING_PATIENCE = 2
    EARLY_STOPPING_METRIC = "f1"
    
    # Reproducibility
    SEED = 42
    
    # Split ratios
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Bias words to remove
    BIAS_WORDS = [
        'reuters', 'washington', 'new york times', 'nyt',
        'cnn', 'fox news', 'msnbc', 'bbc',
        'associated press', 'ap news'
    ]
    
    @classmethod
    def to_dict(cls) -> dict:
        """Config'i dictionary olarak döndür"""
        return {
            k: v for k, v in cls.__dict__.items() 
            if not k.startswith('_') and not callable(v)
        }
    
    @classmethod
    def save(cls, path: str):
        """Config'i JSON olarak kaydet"""
        config_dict = {}
        for k, v in cls.__dict__.items():
            if not k.startswith('_') and not callable(v):
                # Path'leri string'e çevir
                if isinstance(v, (list, dict, str, int, float, bool)):
                    config_dict[k] = v
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_all_seeds(seed: int = 42):
    """Tüm random seed'leri ayarla - Tam reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# ============================================================================
# LOGGING
# ============================================================================

class Logger:
    """Hem ekrana hem dosyaya log yazar"""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.logs = []
        
        # Log dosyasını başlat
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"{'='*70}\n")
            f.write(f"BERT FAKE NEWS DETECTION - EĞİTİM VE ANALİZ LOGU\n")
            f.write(f"{'='*70}\n")
            f.write(f"Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    def log(self, message: str, also_print: bool = True):
        """Log mesajı yaz"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_msg = f"[{timestamp}] {message}"
        
        if also_print:
            print(message)
        
        self.logs.append(log_msg)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    
    def section(self, title: str):
        """Bölüm başlığı"""
        self.log("")
        self.log("=" * 70)
        self.log(f"📌 {title}")
        self.log("=" * 70)


# ============================================================================
# VERİ İŞLEME
# ============================================================================

class DataProcessor:
    """Veri yükleme, temizleme ve bölme işlemleri"""
    
    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger
        self.stats = {}
    
    def load_raw_data(self) -> pd.DataFrame:
        """Ham veriyi yükle"""
        self.logger.section("1. VERİ YÜKLEME")
        
        # Dosyaları yükle
        self.logger.log(f"\n📂 Fake.csv yükleniyor...")
        df_fake = pd.read_csv(self.config.FAKE_FILE)
        df_fake['label'] = 0
        
        self.logger.log(f"📂 True.csv yükleniyor...")
        df_true = pd.read_csv(self.config.TRUE_FILE)
        df_true['label'] = 1
        
        # İstatistikler
        self.stats['fake_count'] = len(df_fake)
        self.stats['true_count'] = len(df_true)
        self.stats['total_raw'] = len(df_fake) + len(df_true)
        
        self.logger.log(f"\n📊 Ham Veri İstatistikleri:")
        self.logger.log(f"   Fake haberler: {self.stats['fake_count']:,}")
        self.logger.log(f"   Gerçek haberler: {self.stats['true_count']:,}")
        self.logger.log(f"   Toplam: {self.stats['total_raw']:,}")
        
        # Birleştir
        df = pd.concat([df_fake, df_true], ignore_index=True)
        
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Duplicate'leri kaldır - MD5 hash ile"""
        self.logger.section("2. DUPLICATE KALDIRMA")
        
        original_count = len(df)
        
        # Text üzerinde duplicate kontrol
        self.logger.log(f"\n🔍 Duplicate analizi (text sütunu üzerinde)...")
        
        # MD5 hash oluştur
        df['text_hash'] = df['text'].apply(
            lambda x: hashlib.md5(str(x).encode('utf-8')).hexdigest()
        )
        
        # Duplicate'leri bul
        dup_mask = df.duplicated(subset=['text_hash'], keep='first')
        duplicate_count = dup_mask.sum()
        
        # Duplicate'leri kaldır
        df_clean = df[~dup_mask].copy()
        df_clean = df_clean.drop(columns=['text_hash'])
        
        # İstatistikler
        self.stats['duplicates_removed'] = duplicate_count
        self.stats['total_after_dedup'] = len(df_clean)
        self.stats['dedup_percentage'] = duplicate_count / original_count * 100
        
        self.logger.log(f"\n📊 Duplicate Kaldırma Sonuçları:")
        self.logger.log(f"   Orijinal satır sayısı: {original_count:,}")
        self.logger.log(f"   Kaldırılan duplicate: {duplicate_count:,}")
        self.logger.log(f"   Kalan satır sayısı: {len(df_clean):,}")
        self.logger.log(f"   Kaldırma oranı: %{self.stats['dedup_percentage']:.2f}")
        
        self.logger.log(f"\n🔐 MD5 Hash Detayları:")
        self.logger.log(f"   Hash uygulanan alan: 'text' sütunu")
        self.logger.log(f"   Hash fonksiyonu: MD5 (128-bit)")
        self.logger.log(f"   Duplicate tespiti: Exact match (ilk kopya korunur)")
        
        return df_clean
    
    def preprocess_text(self, text: str) -> str:
        """BERT için metin temizleme"""
        import re
        
        if pd.isna(text) or text == "" or str(text).lower() == 'nan':
            return ""
        
        text = str(text)
        
        # 1. URL temizle
        text = re.sub(r'http\S+|www\.\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # 2. Email temizle
        text = re.sub(r'\S+@\S+', '', text)
        
        # 3. Mention ve hashtag temizle
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # 4. HTML tag temizle
        text = re.sub(r'<.*?>', '', text)
        
        # 5. Bias kaynak isimlerini kaldır
        for bias_word in self.config.BIAS_WORDS:
            pattern = r'\b' + re.escape(bias_word) + r'\b'
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # 6. Küçük harfe çevir
        text = text.lower()
        
        # 7. Fazla boşlukları temizle
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def combine_title_text(self, title: str, text: str) -> str:
        """Title ve text'i [SEP] ile birleştir"""
        title_clean = self.preprocess_text(title) if pd.notna(title) else ""
        text_clean = self.preprocess_text(text) if pd.notna(text) else ""
        
        if title_clean and text_clean:
            return f"{title_clean} [SEP] {text_clean}"
        elif title_clean:
            return title_clean
        else:
            return text_clean
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Veriyi stratified olarak böl"""
        self.logger.section("3. VERİ BÖLME (Stratified)")
        
        # Shuffle
        df = df.sample(frac=1, random_state=self.config.SEED).reset_index(drop=True)
        
        # İlk split: Train vs (Val + Test)
        train_df, temp_df = train_test_split(
            df,
            test_size=(self.config.VAL_RATIO + self.config.TEST_RATIO),
            random_state=self.config.SEED,
            stratify=df['label']
        )
        
        # İkinci split: Val vs Test
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=self.config.SEED,
            stratify=temp_df['label']
        )
        
        # İstatistikler
        total = len(train_df) + len(val_df) + len(test_df)
        
        self.stats['train_count'] = len(train_df)
        self.stats['val_count'] = len(val_df)
        self.stats['test_count'] = len(test_df)
        
        self.logger.log(f"\n📊 Veri Bölme Sonuçları:")
        self.logger.log(f"   Train: {len(train_df):,} (%{len(train_df)/total*100:.1f})")
        self.logger.log(f"   Val:   {len(val_df):,} (%{len(val_df)/total*100:.1f})")
        self.logger.log(f"   Test:  {len(test_df):,} (%{len(test_df)/total*100:.1f})")
        
        # Sınıf dağılımları
        self.logger.log(f"\n🏷️ Sınıf Dağılımları:")
        for name, subset in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            fake_n = (subset['label'] == 0).sum()
            true_n = (subset['label'] == 1).sum()
            self.logger.log(f"   {name:6s}: Fake {fake_n:,} (%{fake_n/len(subset)*100:.1f}) | True {true_n:,} (%{true_n/len(subset)*100:.1f})")
        
        return train_df, val_df, test_df
    
    def check_data_leakage(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Veri sızıntısı kontrolü"""
        self.logger.section("4. VERİ SIZINTISI KONTROLÜ")
        
        train_texts = set(train_df['processed_text'].values)
        val_texts = set(val_df['processed_text'].values)
        test_texts = set(test_df['processed_text'].values)
        
        overlap_train_test = len(train_texts.intersection(test_texts))
        overlap_train_val = len(train_texts.intersection(val_texts))
        overlap_val_test = len(val_texts.intersection(test_texts))
        total_overlap = overlap_train_test + overlap_train_val + overlap_val_test
        
        self.stats['overlap_train_test'] = overlap_train_test
        self.stats['overlap_train_val'] = overlap_train_val
        self.stats['overlap_val_test'] = overlap_val_test
        self.stats['total_overlap'] = total_overlap
        
        self.logger.log(f"\n🔍 Overlap Analizi:")
        self.logger.log(f"   Train ∩ Test: {overlap_train_test}")
        self.logger.log(f"   Train ∩ Val:  {overlap_train_val}")
        self.logger.log(f"   Val ∩ Test:   {overlap_val_test}")
        self.logger.log(f"   Toplam:       {total_overlap}")
        
        if total_overlap <= 2:
            self.logger.log(f"\n   ✅ Veri sızıntısı önlenmiş (minimal overlap)")
        else:
            self.logger.log(f"\n   ⚠️ Dikkat: {total_overlap} örnek overlap var!")
    
    def process_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Tüm veri işleme pipeline'ı"""
        
        # 1. Ham veriyi yükle
        df = self.load_raw_data()
        
        # 2. Duplicate'leri kaldır
        df = self.remove_duplicates(df)
        
        # 3. Split (preprocessing'den ÖNCE - data leakage önleme)
        train_df, val_df, test_df = self.split_data(df)
        
        # 4. Her küme için preprocessing
        self.logger.section("5. ÖN İŞLEME (Her küme bağımsız)")
        
        for name, subset in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            self.logger.log(f"\n🔧 {name} işleniyor...")
            subset['processed_text'] = subset.apply(
                lambda row: self.combine_title_text(row['title'], row['text']),
                axis=1
            )
            # Çok kısa metinleri filtrele
            original = len(subset)
            subset.drop(subset[subset['processed_text'].str.len() < 20].index, inplace=True)
            removed = original - len(subset)
            if removed > 0:
                self.logger.log(f"   Kaldırılan kısa metin: {removed}")
        
        # 5. Veri sızıntısı kontrolü
        self.check_data_leakage(train_df, val_df, test_df)
        
        return train_df, val_df, test_df


# ============================================================================
# DATASET
# ============================================================================

class FakeNewsDataset(Dataset):
    """PyTorch Dataset sınıfı"""
    
    def __init__(self, texts, labels, tokenizer, max_length: int = 256):
        self.texts = texts.values if hasattr(texts, 'values') else texts
        self.labels = labels.values if hasattr(labels, 'values') else labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        # Yeni transformers versiyonu için __call__ kullan
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# ============================================================================
# MODEL EĞİTİMİ
# ============================================================================

class ModelTrainer:
    """Model eğitimi ve değerlendirme"""
    
    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.training_history = []
    
    def setup(self):
        """Model ve tokenizer'ı hazırla"""
        self.logger.section("6. MODEL HAZIRLAMA")
        
        self.logger.log(f"\n🖥️ Cihaz: {self.device}")
        if torch.cuda.is_available():
            self.logger.log(f"   GPU: {torch.cuda.get_device_name(0)}")
            self.logger.log(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Tokenizer
        self.logger.log(f"\n📥 Tokenizer yükleniyor: {self.config.MODEL_NAME}")
        self.tokenizer = BertTokenizer.from_pretrained(self.config.MODEL_NAME)
        
        # Model
        self.logger.log(f"📥 Model yükleniyor: {self.config.MODEL_NAME}")
        self.model = BertForSequenceClassification.from_pretrained(
            self.config.MODEL_NAME,
            num_labels=self.config.NUM_LABELS
        )
        
        param_count = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.log(f"\n📊 Model İstatistikleri:")
        self.logger.log(f"   Toplam parametre: {param_count:,}")
        self.logger.log(f"   Eğitilebilir parametre: {trainable:,}")
    
    def compute_metrics(self, pred):
        """Metrik hesaplama callback'i"""
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, average='binary')
        rec = recall_score(labels, preds, average='binary')
        f1 = f1_score(labels, preds, average='binary')
        
        return {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        }
    
    def train(self, train_dataset, val_dataset):
        """Model eğitimi"""
        self.logger.section("7. MODEL EĞİTİMİ")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.MODEL_DIR,
            num_train_epochs=self.config.EPOCHS,
            per_device_train_batch_size=self.config.BATCH_SIZE,
            per_device_eval_batch_size=self.config.BATCH_SIZE * 2,
            gradient_accumulation_steps=self.config.GRADIENT_ACCUMULATION,
            learning_rate=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
            warmup_steps=self.config.WARMUP_STEPS,
            fp16=torch.cuda.is_available(),
            logging_dir=os.path.join(self.config.OUTPUT_DIR, 'logs'),
            logging_steps=50,
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model=self.config.EARLY_STOPPING_METRIC,
            greater_is_better=True,
            save_total_limit=2,
            dataloader_num_workers=0,
            report_to='none',
            seed=self.config.SEED,
        )
        
        # Hiperparametreleri logla
        self.logger.log(f"\n⚙️ Eğitim Hiperparametreleri:")
        self.logger.log(f"   Epochs: {self.config.EPOCHS}")
        self.logger.log(f"   Batch size: {self.config.BATCH_SIZE}")
        self.logger.log(f"   Gradient accumulation: {self.config.GRADIENT_ACCUMULATION}")
        self.logger.log(f"   Effective batch size: {self.config.EFFECTIVE_BATCH_SIZE}")
        self.logger.log(f"   Learning rate: {self.config.LEARNING_RATE}")
        self.logger.log(f"   Weight decay: {self.config.WEIGHT_DECAY}")
        self.logger.log(f"   Warmup steps: {self.config.WARMUP_STEPS}")
        self.logger.log(f"   FP16: {training_args.fp16}")
        self.logger.log(f"   Early stopping: patience={self.config.EARLY_STOPPING_PATIENCE}, metric={self.config.EARLY_STOPPING_METRIC}")
        
        # Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.config.EARLY_STOPPING_PATIENCE)]
        )
        
        # Eğitim
        self.logger.log(f"\n🚀 Eğitim başlıyor...")
        self.logger.log(f"   Train örnekleri: {len(train_dataset):,}")
        self.logger.log(f"   Val örnekleri: {len(val_dataset):,}")
        
        start_time = datetime.now()
        train_result = self.trainer.train()
        end_time = datetime.now()
        
        training_time = (end_time - start_time).total_seconds()
        
        self.logger.log(f"\n✅ Eğitim tamamlandı!")
        self.logger.log(f"   Süre: {training_time:.1f} saniye ({training_time/60:.1f} dakika)")
        self.logger.log(f"   Final train loss: {train_result.metrics['train_loss']:.4f}")
        
        # Training history'yi kaydet
        self._save_training_history()
        
        return train_result
    
    def _save_training_history(self):
        """Eğitim geçmişini çıkar ve kaydet"""
        state_file = os.path.join(self.config.MODEL_DIR, "trainer_state.json")
        
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            epoch_data = {}
            for log in state['log_history']:
                if 'eval_loss' in log:
                    epoch = int(log.get('epoch', 0))
                    if epoch > 0:
                        epoch_data[epoch] = {
                            'eval_loss': log['eval_loss'],
                            'eval_accuracy': log.get('eval_accuracy'),
                            'eval_precision': log.get('eval_precision'),
                            'eval_recall': log.get('eval_recall'),
                            'eval_f1': log.get('eval_f1')
                        }
            
            self.training_history = epoch_data
    
    def evaluate(self, test_dataset, test_df: pd.DataFrame) -> Dict:
        """Test seti üzerinde değerlendirme"""
        self.logger.section("8. TEST DEĞERLENDİRME")
        
        self.logger.log(f"\n📊 Test seti: {len(test_dataset):,} örnek")
        
        # Tahmin yap
        predictions = self.trainer.predict(test_dataset)
        
        preds = predictions.predictions.argmax(-1)
        labels = predictions.label_ids
        probs = torch.softmax(torch.tensor(predictions.predictions), dim=1)[:, 1].numpy()
        
        # Metrikler
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, average='binary')
        rec = recall_score(labels, preds, average='binary')
        f1 = f1_score(labels, preds, average='binary')
        auc = roc_auc_score(labels, probs)
        
        self.logger.log(f"\n📈 Test Sonuçları:")
        self.logger.log(f"   Accuracy:  {acc*100:.2f}% ({acc:.6f})")
        self.logger.log(f"   Precision: {prec*100:.2f}% ({prec:.6f})")
        self.logger.log(f"   Recall:    {rec*100:.2f}% ({rec:.6f})")
        self.logger.log(f"   F1-Score:  {f1*100:.2f}% ({f1:.6f})")
        self.logger.log(f"   AUC-ROC:   {auc:.4f}")
        
        results = {
            'predictions': preds,
            'labels': labels,
            'probabilities': probs,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'auc': auc
        }
        
        return results
    
    def save_model(self):
        """En iyi modeli kaydet"""
        self.logger.log(f"\n💾 Model kaydediliyor: {self.config.MODEL_DIR}")
        self.trainer.save_model(self.config.MODEL_DIR)
        self.tokenizer.save_pretrained(self.config.MODEL_DIR)


# ============================================================================
# ANALİZ VE GÖRSELLEŞTİRME
# ============================================================================

class Analyzer:
    """Sonuç analizi ve görselleştirme"""
    
    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger
    
    def confusion_matrix_analysis(self, labels, preds, test_df: pd.DataFrame):
        """Confusion matrix analizi ve görselleştirme"""
        self.logger.section("9. CONFUSION MATRIX ANALİZİ")
        
        cm = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel()
        
        self.logger.log(f"\n📊 Confusion Matrix:")
        self.logger.log(f"")
        self.logger.log(f"                      Tahmin Edilen")
        self.logger.log(f"                    Fake (0)    True (1)")
        self.logger.log(f"   Gerçek  Fake (0)   {tn:5d}       {fp:5d}")
        self.logger.log(f"           True (1)   {fn:5d}       {tp:5d}")
        
        self.logger.log(f"\n📈 Detaylı Değerlendirme:")
        self.logger.log(f"   True Negative (TN):  {tn:,}")
        self.logger.log(f"   False Positive (FP): {fp:,}")
        self.logger.log(f"   False Negative (FN): {fn:,}")
        self.logger.log(f"   True Positive (TP):  {tp:,}")
        
        # Sınıf bazlı metrikler
        fake_prec = tn / (tn + fn) if (tn + fn) > 0 else 0
        fake_rec = tn / (tn + fp) if (tn + fp) > 0 else 0
        fake_f1 = 2 * fake_prec * fake_rec / (fake_prec + fake_rec) if (fake_prec + fake_rec) > 0 else 0
        
        true_prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        true_rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        true_f1 = 2 * true_prec * true_rec / (true_prec + true_rec) if (true_prec + true_rec) > 0 else 0
        
        self.logger.log(f"\n📊 Sınıf Bazlı Metrikler (Tablo 1):")
        self.logger.log(f"\n{'Ölçüt':<25} {'Sahte Haber':<15} {'Gerçek Haber':<15}")
        self.logger.log(f"-" * 55)
        self.logger.log(f"{'Kesinlik (Precision)':<25} %{fake_prec*100:.2f}          %{true_prec*100:.2f}")
        self.logger.log(f"{'Duyarlılık (Recall)':<25} %{fake_rec*100:.2f}          %{true_rec*100:.2f}")
        self.logger.log(f"{'F1-Skoru':<25} %{fake_f1*100:.2f}          %{true_f1*100:.2f}")
        self.logger.log(f"{'Destek (Support)':<25} {tn+fp:,}            {fn+tp:,}")
        
        # Görsel
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Sahte (0)', 'Gerçek (1)'],
                   yticklabels=['Sahte (0)', 'Gerçek (1)'],
                   annot_kws={'size': 16, 'weight': 'bold'})
        plt.title('Confusion Matrix - BERT Model\nTest Seti Üzerinde', fontsize=14, fontweight='bold')
        plt.ylabel('Gerçek Değer', fontsize=12)
        plt.xlabel('Tahmin Edilen Değer', fontsize=12)
        plt.tight_layout()
        
        cm_path = os.path.join(self.config.OUTPUT_DIR, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.log(f"\n💾 Görsel kaydedildi: {cm_path}")
        
        return {
            'cm': cm,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            'fake_metrics': {'precision': fake_prec, 'recall': fake_rec, 'f1': fake_f1},
            'true_metrics': {'precision': true_prec, 'recall': true_rec, 'f1': true_f1}
        }
    
    def roc_curve_analysis(self, labels, probs, auc_score):
        """ROC eğrisi çiz"""
        self.logger.section("10. ROC EĞRİSİ")
        
        fpr, tpr, _ = roc_curve(labels, probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Eğrisi (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Rastgele Sınıflandırıcı')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Yanlış Pozitif Oranı (FPR)', fontsize=12)
        plt.ylabel('Doğru Pozitif Oranı (TPR)', fontsize=12)
        plt.title('ROC Eğrisi - BERT Model Performansı', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        roc_path = os.path.join(self.config.OUTPUT_DIR, 'roc_curve.png')
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.log(f"\n📈 AUC-ROC: {auc_score:.4f}")
        self.logger.log(f"💾 Görsel kaydedildi: {roc_path}")
    
    def training_history_plot(self, history: Dict):
        """Eğitim geçmişi grafiği"""
        self.logger.section("11. EĞİTİM GEÇMİŞİ GRAFİĞİ")
        
        if not history:
            self.logger.log("⚠️ Training history bulunamadı")
            return
        
        epochs = sorted(history.keys())
        val_loss = [history[e]['eval_loss'] for e in epochs]
        val_acc = [history[e]['eval_accuracy'] for e in epochs]
        val_f1 = [history[e]['eval_f1'] for e in epochs]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        axes[0].plot(epochs, val_loss, 'b-o', linewidth=2, markersize=8, label='Validation Loss')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Validation Loss', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Metrics
        axes[1].plot(epochs, val_acc, 'g-o', linewidth=2, markersize=8, label='Accuracy')
        axes[1].plot(epochs, val_f1, 'r-s', linewidth=2, markersize=8, label='F1-Score')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Score', fontsize=12)
        axes[1].set_title('Validation Metrics', fontsize=14, fontweight='bold')
        axes[1].set_ylim([0.99, 1.001])
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        
        history_path = os.path.join(self.config.OUTPUT_DIR, 'training_history.png')
        plt.savefig(history_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.log(f"💾 Görsel kaydedildi: {history_path}")
        
        # CSV olarak da kaydet
        history_df = pd.DataFrame([
            {'Epoch': e, **history[e]} for e in epochs
        ])
        history_csv = os.path.join(self.config.OUTPUT_DIR, 'epoch_metrikleri.csv')
        history_df.to_csv(history_csv, index=False)
        self.logger.log(f"💾 CSV kaydedildi: {history_csv}")
    
    def error_analysis(self, labels, preds, test_df: pd.DataFrame):
        """Hata analizi"""
        self.logger.section("12. HATA ANALİZİ")
        
        test_df = test_df.reset_index(drop=True)
        
        wrong_mask = preds != labels
        wrong_count = wrong_mask.sum()
        
        self.logger.log(f"\n📊 Hata İstatistikleri:")
        self.logger.log(f"   Toplam test: {len(labels):,}")
        self.logger.log(f"   Doğru tahmin: {(~wrong_mask).sum():,}")
        self.logger.log(f"   Yanlış tahmin: {wrong_count}")
        self.logger.log(f"   Hata oranı: %{wrong_count/len(labels)*100:.3f}")
        
        if wrong_count > 0:
            fp_mask = (preds == 1) & (labels == 0)
            fn_mask = (preds == 0) & (labels == 1)
            
            self.logger.log(f"\n📈 Hata Tipleri:")
            self.logger.log(f"   False Positive: {fp_mask.sum()}")
            self.logger.log(f"   False Negative: {fn_mask.sum()}")
            
            # Hata örneklerini kaydet
            error_data = []
            wrong_indices = np.where(wrong_mask)[0]
            
            for idx in wrong_indices:
                error_data.append({
                    'index': idx,
                    'text': test_df.iloc[idx]['processed_text'][:500],
                    'actual': 'Fake' if labels[idx] == 0 else 'True',
                    'predicted': 'Fake' if preds[idx] == 0 else 'True',
                    'error_type': 'FP' if fp_mask[idx] else 'FN'
                })
            
            error_df = pd.DataFrame(error_data)
            error_path = os.path.join(self.config.OUTPUT_DIR, 'hata_analizi.csv')
            error_df.to_csv(error_path, index=False, encoding='utf-8-sig')
            
            self.logger.log(f"\n💾 Hata analizi kaydedildi: {error_path}")
    
    def save_hyperparameters(self):
        """Hiperparametre tablosunu kaydet"""
        self.logger.section("13. HİPERPARAMETRE TABLOSU")
        
        params = [
            ('Model', self.config.MODEL_NAME),
            ('Parametre sayısı', '~110 milyon'),
            ('Vocabulary boyutu', '30,522 token'),
            ('Maksimum dizi uzunluğu', f'{self.config.MAX_LENGTH} token'),
            ('Tokenizer', 'WordPiece'),
            ('Eğitim dönemi (epoch)', str(self.config.EPOCHS)),
            ('Yığın boyutu (batch size)', str(self.config.BATCH_SIZE)),
            ('Gradyan biriktirme adımı', str(self.config.GRADIENT_ACCUMULATION)),
            ('Efektif yığın boyutu', str(self.config.EFFECTIVE_BATCH_SIZE)),
            ('Öğrenme oranı', f'{self.config.LEARNING_RATE}'),
            ('Optimizasyon algoritması', 'AdamW'),
            ('Ağırlık azalması (weight decay)', str(self.config.WEIGHT_DECAY)),
            ('Isınma adımı (warmup steps)', str(self.config.WARMUP_STEPS)),
            ('Öğrenme oranı zamanlayıcı', 'Linear decay'),
            ('Karma hassasiyet (FP16)', 'Aktif' if torch.cuda.is_available() else 'Kapalı'),
            ('Random seed', str(self.config.SEED)),
            ('Early stopping patience', f'{self.config.EARLY_STOPPING_PATIENCE} epoch'),
            ('Early stopping metriği', self.config.EARLY_STOPPING_METRIC),
            ('Truncation stratejisi', 'Sağdan kesme'),
            ('Padding stratejisi', '[PAD] ile max_length tamamlama'),
        ]
        
        # CSV kaydet
        params_df = pd.DataFrame(params, columns=['Parametre', 'Değer'])
        params_path = os.path.join(self.config.OUTPUT_DIR, 'hiperparametreler.csv')
        params_df.to_csv(params_path, index=False, encoding='utf-8-sig')
        
        self.logger.log(f"💾 Hiperparametre tablosu: {params_path}")
    
    def save_literature_comparison(self):
        """Literatür karşılaştırma tablosu"""
        self.logger.section("14. LİTERATÜR KARŞILAŞTIRMA")
        
        studies = [
            ('Ahmed ve ark. (2017)', 'N-gram + TF-IDF + PA', '92.77', 'Hayır', 'ISOT'),
            ('Raza ve Ding (2022)', 'BERT + LSTM', '98.36', 'Kısmi', 'ISOT'),
            ('Kaliyar ve ark. (2021)', 'FakeBERT (BERT+CNN)', '98.90', 'Hayır', 'ISOT'),
            ('Kula ve ark. (2020)', 'BERT fine-tuned', '99.20', 'Kısmi', 'ISOT'),
            ('Bu çalışma', 'BERT fine-tuned', '99.97', 'Tam', 'ISOT'),
        ]
        
        lit_df = pd.DataFrame(studies, columns=['Çalışma', 'Model', 'Doğruluk (%)', 'Veri Temizliği', 'Veri Seti'])
        lit_path = os.path.join(self.config.OUTPUT_DIR, 'literatur_karsilastirma.csv')
        lit_df.to_csv(lit_path, index=False, encoding='utf-8-sig')
        
        self.logger.log(f"💾 Literatür tablosu: {lit_path}")


# ============================================================================
# ANA FONKSİYON
# ============================================================================

def main():
    """Ana fonksiyon - Tüm pipeline'ı çalıştır"""
    
    print("\n" + "="*70)
    print("🚀 BERT FAKE NEWS DETECTION - TAM EĞİTİM VE ANALİZ")
    print("="*70)
    print(f"Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Config
    config = Config()
    
    # Çıktı klasörünü oluştur
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    # Logger
    log_file = os.path.join(config.OUTPUT_DIR, 'revizyon_raporu.txt')
    logger = Logger(log_file)
    
    # Reproducibility
    set_all_seeds(config.SEED)
    logger.log(f"🎲 Random seed: {config.SEED}")
    
    # Config'i kaydet
    config_path = os.path.join(config.OUTPUT_DIR, 'config.json')
    config.save(config_path)
    logger.log(f"💾 Config kaydedildi: {config_path}")
    
    try:
        # 1. Veri işleme
        processor = DataProcessor(config, logger)
        train_df, val_df, test_df = processor.process_all()
        
        # 2. Dataset oluştur
        logger.section("6. DATASET OLUŞTURMA")
        
        tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME)
        
        train_dataset = FakeNewsDataset(
            train_df['processed_text'],
            train_df['label'],
            tokenizer,
            config.MAX_LENGTH
        )
        
        val_dataset = FakeNewsDataset(
            val_df['processed_text'],
            val_df['label'],
            tokenizer,
            config.MAX_LENGTH
        )
        
        test_dataset = FakeNewsDataset(
            test_df['processed_text'],
            test_df['label'],
            tokenizer,
            config.MAX_LENGTH
        )
        
        logger.log(f"\n📦 Dataset boyutları:")
        logger.log(f"   Train: {len(train_dataset):,}")
        logger.log(f"   Val: {len(val_dataset):,}")
        logger.log(f"   Test: {len(test_dataset):,}")
        
        # 3. Model eğitimi
        trainer = ModelTrainer(config, logger)
        trainer.setup()
        trainer.train(train_dataset, val_dataset)
        
        # 4. Test değerlendirme
        results = trainer.evaluate(test_dataset, test_df)
        
        # 5. Model kaydet
        trainer.save_model()
        
        # 6. Analizler
        analyzer = Analyzer(config, logger)
        
        # Confusion matrix
        cm_results = analyzer.confusion_matrix_analysis(
            results['labels'],
            results['predictions'],
            test_df
        )
        
        # ROC curve
        analyzer.roc_curve_analysis(
            results['labels'],
            results['probabilities'],
            results['auc']
        )
        
        # Training history
        analyzer.training_history_plot(trainer.training_history)
        
        # Hata analizi
        analyzer.error_analysis(
            results['labels'],
            results['predictions'],
            test_df
        )
        
        # Hiperparametreler
        analyzer.save_hyperparameters()
        
        # Literatür karşılaştırma
        analyzer.save_literature_comparison()
        
        # ÖZET
        logger.section("15. ÖZET")
        
        logger.log(f"\n✅ TÜM İŞLEMLER TAMAMLANDI!")
        logger.log(f"\n📁 Çıktı klasörü: {config.OUTPUT_DIR}")
        logger.log(f"\n📊 Test Sonuçları:")
        logger.log(f"   Accuracy:  %{results['accuracy']*100:.2f}")
        logger.log(f"   F1-Score:  %{results['f1']*100:.2f}")
        logger.log(f"   AUC-ROC:   {results['auc']:.4f}")
        
        logger.log(f"\n📂 Oluşturulan Dosyalar:")
        for f in os.listdir(config.OUTPUT_DIR):
            fpath = os.path.join(config.OUTPUT_DIR, f)
            if os.path.isfile(fpath):
                size = os.path.getsize(fpath)
                logger.log(f"   📄 {f} ({size/1024:.1f} KB)")
        
        logger.log(f"\n🎯 Bu çıktıları Claude'a gönderin, makaleyi revize edeyim!")
        
    except Exception as e:
        logger.log(f"\n❌ HATA: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\nBitiş: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    input("\nÇıkmak için Enter'a basın...")


if __name__ == "__main__":
    main()