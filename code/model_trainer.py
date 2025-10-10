import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from sklearn.model_selection import StratifiedKFold





class HybridTicketDataset(Dataset):
    """Dataset for hybrid model with text and numerical features."""
    
    def __init__(self, texts, numerical_features, labels, tokenizer, max_length=512):
        self.texts = texts
        self.numerical_features = numerical_features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        numerical = self.numerical_features[idx]
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'numerical_features': torch.tensor(numerical, dtype=torch.float32),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class AttentionFusion(nn.Module):
    """
    Attention-based fusion for combining text and numerical features.
    
    """
    
    def __init__(self, text_dim, numerical_dim, hidden_dim, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Project both modalities to same dimension
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.numerical_proj = nn.Linear(numerical_dim, hidden_dim)
        
        # Multi-head attention for cross-modal interaction
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            dropout=0.1,
            batch_first=True
        )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Final projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, text_emb, numerical_emb):
        
        text_proj = self.text_proj(text_emb)  # [batch, hidden_dim]
        numerical_proj = self.numerical_proj(numerical_emb)  # [batch, hidden_dim]
        
        text_seq = text_proj.unsqueeze(1)  # [batch, 1, hidden_dim]
        numerical_seq = numerical_proj.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        
        attended_text, attention_weights = self.cross_attention(
            query=text_seq,
            key=numerical_seq, 
            value=numerical_seq
        )
        
        # Remove sequence dimension
        attended_text = attended_text.squeeze(1)  # [batch, hidden_dim]
        
        # Residual connection + layer norm
        fused = self.layer_norm(text_proj + attended_text)
        
        # Final projection
        output = self.output_proj(fused)
        
        return output, attention_weights


class HybridRoBERTaModel(nn.Module):
    """Enhanced hybrid model with attention-based fusion and class weight support."""
    
    def __init__(self, model_name='roberta-base', num_labels=7, num_numerical_features=17, 
                 hidden_dim=256, dropout_rate=0.3, use_attention_fusion=True):
        super().__init__()
        
        self.use_attention_fusion = use_attention_fusion
        
        # Load pre-trained RoBERTa
        self.roberta = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels,
            output_hidden_states=True
        )
        
        # Get RoBERTa hidden size
        config = AutoConfig.from_pretrained(model_name)
        roberta_hidden_size = config.hidden_size
        
        # Numerical features branch
        self.numerical_fc = nn.Sequential(
            nn.Linear(num_numerical_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
    
        if self.use_attention_fusion:
            self.fusion = AttentionFusion(
                text_dim=roberta_hidden_size,
                numerical_dim=hidden_dim // 2, 
                hidden_dim=hidden_dim
            )
            fusion_output_size = hidden_dim
        else:
            fusion_output_size = roberta_hidden_size + (hidden_dim // 2)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_output_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_labels)
        )
        
        # Replace RoBERTa's classifier with identity to get embeddings
        self.roberta.classifier = nn.Identity()
        
    def forward(self, input_ids, attention_mask, numerical_features, labels=None, class_weights=None):
        # Text branch - get [CLS] token embeddings
        roberta_outputs = self.roberta.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Extract [CLS] token embedding (first token)
        text_embeddings = roberta_outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Numerical branch
        numerical_embeddings = self.numerical_fc(numerical_features)  # [batch_size, hidden_dim//2]
        
        # Fusion
        if self.use_attention_fusion:
            fused_embeddings, attention_weights = self.fusion(text_embeddings, numerical_embeddings)
        else:
            # Simple concatenation fallback
            fused_embeddings = torch.cat([text_embeddings, numerical_embeddings], dim=1)
            attention_weights = None
        
        # Final classification
        logits = self.classifier(fused_embeddings)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            if class_weights is not None:
                # Convert class_weights dict to tensor
                weight_tensor = torch.tensor(
                    [class_weights[i] for i in range(len(class_weights))], 
                    dtype=torch.float32, 
                    device=labels.device
                )
                loss_fct = nn.CrossEntropyLoss(weight=weight_tensor)
            else:
                loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'text_embeddings': text_embeddings,
            'numerical_embeddings': numerical_embeddings,
            'attention_weights': attention_weights
        }


class HybridTransformerModel:
    """Enhanced hybrid transformer model with class weights and confidence routing."""
    
    def __init__(self, model_name='roberta-base', num_labels=7, num_numerical_features=17, 
                 use_gpu=True, use_attention_fusion=True):
        self.model_name = model_name
        self.num_labels = num_labels
        self.num_numerical_features = num_numerical_features
        self.use_attention_fusion = use_attention_fusion
        self.tokenizer = None
        self.model = None
        self.scaler = StandardScaler()
        self.class_weights = None
        
        # GPU setup
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")
        
    def setup_model(self):
        """Setup the hybrid model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = HybridRoBERTaModel(
            model_name=self.model_name,
            num_labels=self.num_labels,
            num_numerical_features=self.num_numerical_features,
            use_attention_fusion=self.use_attention_fusion
        )
        self.model.to(self.device)
    
    def lr_find(self, train_texts, train_numerical, train_labels, 
           start_lr=1e-8, end_lr=10, num_iter=100, batch_size=16):
        """Learning rate finder - run before actual training."""
        self.setup_model()
        
        
        train_numerical_scaled = self.scaler.fit_transform(train_numerical)
        
        
        train_dataset = HybridTicketDataset(
            train_texts, train_numerical_scaled, train_labels, self.tokenizer
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=start_lr)
        self.model.train()
        
        lrs, losses = [], []
        lr = start_lr
        
        for i, batch in enumerate(train_loader):
            if i >= num_iter:
                break
                
            # Update learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            numerical_features = batch['numerical_features'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                numerical_features=numerical_features,
                labels=labels,
                class_weights=self.class_weights
            )
            
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            
            
            lrs.append(lr)
            losses.append(loss.item())
            
            # Exponentially increase LR
            lr *= (end_lr / start_lr) ** (1 / num_iter)
        
        return lrs, losses
        
    # warmup-steps --> useless 
    # can be also val_f1 in the monitor_metric
    def train(self, train_texts, train_numerical, train_labels, 
              val_texts=None, val_numerical=None, val_labels=None,
              epochs=3, batch_size=16, learning_rate=2e-5, 
              class_weights=None, early_stopping=True, patience=2,monitor_metric='val_loss'):
        """
        Train the hybrid model with class weight support and early stopping.
        """
        self.setup_model()
        self.class_weights = class_weights

        # Scale numerical features
        train_numerical_scaled = self.scaler.fit_transform(train_numerical)

        # Create datasets
        train_dataset = HybridTicketDataset(
            train_texts, train_numerical_scaled, train_labels, self.tokenizer
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

        val_loader = None
        if val_texts is not None and val_numerical is not None and val_labels is not None:
            val_numerical_scaled = self.scaler.transform(val_numerical)
            val_dataset = HybridTicketDataset(
                val_texts, val_numerical_scaled, val_labels, self.tokenizer
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)

        self.model.train()
        best_metric = float('inf') if monitor_metric == 'val_loss' else -float('inf')
        best_epoch = 0
        epochs_no_improve = 0
        best_model_state = None

        val_metrics_history = []

        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for step, batch in enumerate(progress_bar):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                numerical_features = batch['numerical_features'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    numerical_features=numerical_features,
                    labels=labels,
                    class_weights=self.class_weights
                )

                loss = outputs['loss']
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}')

            # Validation and early stopping check
            if val_loader is not None:
                val_loss, val_acc, val_f1 = self.evaluate(val_loader)
                val_metrics_history.append({'val_loss': val_loss, 'val_acc': val_acc, 'val_f1': val_f1})

                # Early stopping logic
                if monitor_metric == 'val_loss':
                    this_metric = val_loss
                    improved = this_metric < best_metric
                elif monitor_metric == 'val_f1':
                    this_metric = val_f1
                    improved = this_metric > best_metric
                elif monitor_metric == 'val_acc':
                    this_metric = val_acc
                    improved = this_metric > best_metric
                else:
                    raise ValueError(f"Unsupported monitor_metric: {monitor_metric}")

                if improved:
                    best_metric = this_metric
                    best_epoch = epoch
                    epochs_no_improve = 0
                    # Save the current best model weights
                    best_model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                else:
                    epochs_no_improve += 1
                    print(f"No improvement in {monitor_metric} for {epochs_no_improve} epoch(s)")

                if early_stopping and epochs_no_improve >= patience:
                    print(f"Early stopping triggered! Best {monitor_metric}: {best_metric:.4f} at epoch {best_epoch+1}")
                    break

        # Restore best model weights if early stopped
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Restored best model weights from epoch {best_epoch+1}")

        return val_metrics_history  # (optional, can also be None)
    
    def predict(self, texts, numerical_features, batch_size=16):
        """Make predictions."""
        self.model.eval()
        predictions = []
        
        # Scale numerical features
        numerical_scaled = self.scaler.transform(numerical_features)
        
        dataset = HybridTicketDataset(
            texts, numerical_scaled, [0] * len(texts), self.tokenizer
        )
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Predicting'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                numerical_features_batch = batch['numerical_features'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    numerical_features=numerical_features_batch
                )
                
                logits = outputs['logits']
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())
        
        return np.array(predictions)
    
    def predict_proba(self, texts, numerical_features, batch_size=16):
        """Get prediction probabilities."""
        self.model.eval()
        probabilities = []
        
        # Scale numerical features
        numerical_scaled = self.scaler.transform(numerical_features)
        
        dataset = HybridTicketDataset(
            texts, numerical_scaled, [0] * len(texts), self.tokenizer
        )
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Predicting probabilities'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                numerical_features_batch = batch['numerical_features'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    numerical_features=numerical_features_batch
                )
                
                logits = outputs['logits']
                probs = torch.softmax(logits, dim=1)
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(probabilities)
    
    def confidence_based_routing(self, probabilities, threshold_high=0.7, threshold_low=0.3):
        """
        
        Args:
            probabilities: Model prediction probabilities [n_samples, n_classes]
            threshold_high: Confidence threshold for auto-routing
            threshold_low: Confidence threshold for manual triage
            
        Returns:
            Dict with routing decisions and confidence scores
        """
        max_confidence = np.max(probabilities, axis=1)
        print(f"Max confidence range: {np.min(max_confidence):.3f} to {np.max(max_confidence):.3f}")
        print(f"Threshold high: {threshold_high}")
        print(f"Tickets above threshold: {np.sum(max_confidence >= threshold_high)}")
        predicted_classes = np.argmax(probabilities, axis=1)
        
        # Routing decisions
        auto_route = max_confidence >= threshold_high
        human_verify = (max_confidence >= threshold_low) & (max_confidence < threshold_high)
        manual_triage = max_confidence < threshold_low
        
        # Statistics
        auto_pct = np.mean(auto_route) * 100
        verify_pct = np.mean(human_verify) * 100
        triage_pct = np.mean(manual_triage) * 100


        # Check confidence distribution
        print(f"Confidence stats:")
        print(f"Min: {np.min(max_confidence):.3f}") # could be just one so idk how much relevant
        print(f"Max: {np.max(max_confidence):.3f}")
        print(f"Mean: {np.mean(max_confidence):.3f}")
        print(f"Median: {np.median(max_confidence):.3f}") # --W this is important
        print(f"95th percentile: {np.percentile(max_confidence, 95):.3f}")
        
        print(f"Confidence-Based Routing Results:")
        print(f"Auto-route: {np.sum(auto_route):,} tickets ({auto_pct:.1f}%)")
        print(f"Human verify: {np.sum(human_verify):,} tickets ({verify_pct:.1f}%)")
        print(f"Manual triage: {np.sum(manual_triage):,} tickets ({triage_pct:.1f}%)")

        
        
        return {
            'routing_decisions': {
                'auto_route': auto_route,
                'human_verify': human_verify, 
                'manual_triage': manual_triage
            },
            'confidence_scores': max_confidence,
            'predicted_classes': predicted_classes,
            'statistics': {
                'auto_route_pct': auto_pct,
                'human_verify_pct': verify_pct,
                'manual_triage_pct': triage_pct
            }
        }
    
    def evaluate(self, dataloader):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                numerical_features = batch['numerical_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    numerical_features=numerical_features,
                    labels=labels,
                    class_weights=self.class_weights
                )
                
                loss = outputs['loss']
                logits = outputs['logits']
                
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(true_labels, predictions)
        f1_macro = f1_score(true_labels, predictions, average='macro')
        
        print(f'Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1-Macro: {f1_macro:.4f}')
        return avg_loss, accuracy, f1_macro
    
    def save_model(self, filepath):
        """Save the trained model and scaler."""
        # Create directory if it doesn't exist
        os.makedirs(filepath, exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'class_weights': self.class_weights,
            'model_config': {
                'model_name': self.model_name,
                'num_labels': self.num_labels,
                'num_numerical_features': self.num_numerical_features,
                'use_attention_fusion': self.use_attention_fusion
            }
        }, f"{filepath}/model.pt")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(filepath)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(filepath)
        
        # Load model
        checkpoint = torch.load(f"{filepath}/model.pt", map_location=self.device, weights_only=False)
        
        # Recreate model
        config = checkpoint['model_config']
        self.model = HybridRoBERTaModel(
            model_name=config['model_name'],
            num_labels=config['num_labels'],
            num_numerical_features=config['num_numerical_features'],
            use_attention_fusion=config.get('use_attention_fusion', True)
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        
        # Load scaler and class weights
        self.scaler = checkpoint['scaler']
        self.class_weights = checkpoint.get('class_weights', None)
        
        print(f"Model loaded from {filepath}")
    
    def cross_validate(self, texts, numerical_features, labels, 
                      cv_folds=5, epochs=3, batch_size=16, learning_rate=2e-5,
                      class_weights=None, stratified=True, random_state=42):
        """
        Args:
            texts: List of text inputs
            numerical_features: Numerical feature array
            labels: Target labels
            cv_folds: Number of cross-validation folds
            epochs: Training epochs per fold
            batch_size: Training batch size
            learning_rate: Learning rate
            class_weights: Class weights for imbalanced data
            stratified: Whether to use stratified CV (recommended for imbalanced data)
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with CV results and metrics
        """
        print(f"Starting {cv_folds}-fold cross-validation...")
        
        # Convert to numpy arrays if needed
        texts = np.array(texts)
        numerical_features = np.array(numerical_features)
        labels = np.array(labels)
        
        # Initialize cross-validation splitter
        if stratified:
            cv_splitter = StratifiedKFold(
                n_splits=cv_folds, 
                shuffle=True, 
                random_state=random_state
            )
        else:
            from sklearn.model_selection import KFold
            cv_splitter = KFold(
                n_splits=cv_folds, 
                shuffle=True, 
                random_state=random_state
            )
        
        # Storage for results
        cv_results = {
            'fold_scores': [],
            'fold_f1_macro': [],
            'fold_f1_weighted': [],
            'fold_predictions': [],
            'fold_probabilities': [],
            'fold_true_labels': [],
            'fold_confusion_matrices': []
        }
        
        # Perform cross-validation
        for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(texts, labels)):
            print(f"\n--- Fold {fold + 1}/{cv_folds} ---")
            
            # Split data
            train_texts = texts[train_idx]
            train_numerical = numerical_features[train_idx]
            train_labels = labels[train_idx]
            
            val_texts = texts[val_idx]
            val_numerical = numerical_features[val_idx]
            val_labels = labels[val_idx]
            
            print(f"Train samples: {len(train_texts)}, Validation samples: {len(val_texts)}")
            
            # Create a fresh model for this fold
            fold_model = HybridTransformerModel(
                model_name=self.model_name,
                num_labels=self.num_labels,
                num_numerical_features=self.num_numerical_features,
                use_gpu=self.device.type == 'cuda',
                use_attention_fusion=self.use_attention_fusion
            )
            
            # Train the model
            fold_model.train(
                train_texts=train_texts,
                train_numerical=train_numerical,
                train_labels=train_labels,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                class_weights=class_weights
            )
            
            # Make predictions
            val_predictions = fold_model.predict(val_texts, val_numerical, batch_size=batch_size)
            val_probabilities = fold_model.predict_proba(val_texts, val_numerical, batch_size=batch_size)
            

            fold_cm = confusion_matrix(val_labels, val_predictions)
            cv_results['fold_confusion_matrices'].append(fold_cm)

            # Calculate metrics
            fold_accuracy = accuracy_score(val_labels, val_predictions)
            fold_f1_macro = f1_score(val_labels, val_predictions, average='macro')
            fold_f1_weighted = f1_score(val_labels, val_predictions, average='weighted')
            
            print(f"Fold {fold + 1} Results:")
            print(f"  Accuracy: {fold_accuracy:.4f}")
            print(f"  F1-Macro: {fold_f1_macro:.4f}")
            print(f"  F1-Weighted: {fold_f1_weighted:.4f}")
            
            # Store results
            cv_results['fold_scores'].append(fold_accuracy)
            cv_results['fold_f1_macro'].append(fold_f1_macro)
            cv_results['fold_f1_weighted'].append(fold_f1_weighted)
            cv_results['fold_predictions'].append(val_predictions)
            cv_results['fold_probabilities'].append(val_probabilities)
            cv_results['fold_true_labels'].append(val_labels)
            
            # Clean up GPU memory
            del fold_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Aggregate results
        cv_results['mean_accuracy'] = np.mean(cv_results['fold_scores'])
        cv_results['std_accuracy'] = np.std(cv_results['fold_scores'])
        cv_results['mean_f1_macro'] = np.mean(cv_results['fold_f1_macro'])
        cv_results['std_f1_macro'] = np.std(cv_results['fold_f1_macro'])
        cv_results['mean_f1_weighted'] = np.mean(cv_results['fold_f1_weighted'])
        cv_results['std_f1_weighted'] = np.std(cv_results['fold_f1_weighted'])
        
        # Print final results
        print(f"\n{'='*50}")
        print(f"CROSS-VALIDATION RESULTS ({cv_folds} folds)")
        print(f"{'='*50}")
        print(f"Accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
        print(f"F1-Macro: {cv_results['mean_f1_macro']:.4f} ± {cv_results['std_f1_macro']:.4f}")
        print(f"F1-Weighted: {cv_results['mean_f1_weighted']:.4f} ± {cv_results['std_f1_weighted']:.4f}")
        print(f"{'='*50}")
        
        return cv_results
    
    def analyze_cv_results(self, cv_results, class_names=None):
        """
        Analyze cross-validation results in detail.
        
        Args:
            cv_results: Results from cross_validate method
            class_names: List of class names for detailed analysis
        """
        print("\n--- DETAILED CROSS-VALIDATION ANALYSIS ---")
        
        # Combine all fold predictions for overall metrics
        all_true = np.concatenate(cv_results['fold_true_labels'])
        all_pred = np.concatenate(cv_results['fold_predictions'])
        all_proba = np.concatenate(cv_results['fold_probabilities'])
        
        print("\nOverall Performance (All Folds Combined):")
        ModelEvaluator.evaluate_model(all_true, all_pred, class_names)
        
        if class_names is not None:
            print("\nCERT-EU Specific Analysis:")
            ModelEvaluator.cert_specific_metrics(all_true, all_pred, class_names)
        
        # Confidence analysis
        print("\nConfidence Analysis:")
        confidence_results = ModelEvaluator.analyze_predictions(
            all_true, all_pred, all_proba, class_names, threshold=0.9
        )
        
        # Fold stability analysis
        print("\nFold Stability Analysis:")
        fold_scores = cv_results['fold_scores']
        print(f"Score range: {np.min(fold_scores):.4f} - {np.max(fold_scores):.4f}")
        print(f"Coefficient of variation: {np.std(fold_scores)/np.mean(fold_scores):.4f}")
        
        if np.std(fold_scores) > 0.05:
            print("⚠️  High variance between folds - consider more data or regularization")
        else:
            print("✅ Good stability across folds")
        
        return {
            'overall_metrics': ModelEvaluator.evaluate_model(all_true, all_pred, class_names),
            'confidence_analysis': confidence_results,
            'stability_metrics': {
                'cv_std': np.std(fold_scores),
                'cv_range': np.max(fold_scores) - np.min(fold_scores),
                'coefficient_of_variation': np.std(fold_scores)/np.mean(fold_scores)
            }
        }


class ModelEvaluator:
    """Enhanced model evaluation for CERT-EU classification."""
    
    @staticmethod
    def evaluate_model(y_true, y_pred, class_names=None):
        """Evaluate model performance with multiple metrics."""
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score (Macro): {f1_macro:.4f}")
        print(f"F1 Score (Weighted): {f1_weighted:.4f}")
        
        if class_names is not None:
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred, target_names=class_names))
            
            print("\nConfusion Matrix:")
            cm = confusion_matrix(y_true, y_pred)
            print(cm)
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted
        }
    
    @staticmethod
    def cert_specific_metrics(y_true, y_pred, class_names):
        """
        CERT-specific metrics focusing on critical queues.
        High precision/recall is crucial for incident and vulnerability queues.
        """
        critical_queues = ['DFIR::incidents', 'OFFSEC::CVD', 'DFIR::phishing']

         # Ensure class_names is a list for .index() method
        if isinstance(class_names, np.ndarray):
            class_names = list(class_names)
        
        print("\n=== CERT-EU Critical Queue Performance ===")
        for queue in critical_queues:
            if queue in class_names:
                idx = class_names.index(queue)
                mask_true = y_true == idx
                mask_pred = y_pred == idx
                
                if np.sum(mask_true) > 0:  # Only if queue exists in data
                    y_true_binary = (y_true == idx).astype(int)
                    y_pred_binary = (y_pred == idx).astype(int)
                    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
                    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
                    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
                    
                    print(f"{queue}:")
                    print(f"  Precision: {precision:.3f} (% of predicted tickets that were correct)")
                    print(f"  Recall: {recall:.3f} (% of actual tickets that were caught)")
                    print(f"  F1-Score: {f1:.3f}")
                    print(f"  Support: {np.sum(mask_true)} tickets")
                    print()
    
    @staticmethod
    def analyze_predictions(y_true, y_pred, y_proba, class_names, threshold=0.8):
        """Analyze predictions with confidence thresholds."""
        max_proba = np.max(y_proba, axis=1)
        high_confidence = max_proba >= threshold
        
        print(f"\nConfidence Analysis (threshold={threshold}):")
        print(f"High confidence predictions: {np.sum(high_confidence)} ({np.mean(high_confidence)*100:.1f}%)")
        
        if np.sum(high_confidence) > 0:
            high_conf_accuracy = accuracy_score(y_true[high_confidence], y_pred[high_confidence])
            print(f"High confidence accuracy: {high_conf_accuracy:.4f}")
        
        print(f"Low confidence predictions: {np.sum(~high_confidence)} ({np.mean(~high_confidence)*100:.1f}%)")
        
        return {
            'high_confidence_mask': high_confidence,
            'high_confidence_accuracy': high_conf_accuracy if np.sum(high_confidence) > 0 else 0,
            'confidence_scores': max_proba
        }

    @staticmethod
    def compare_cv_results(results_dict, metric='mean_f1_macro'):
        """
        Compare multiple CV results (e.g., different hyperparameters).
        
        Args:
            results_dict: Dict with format {'model_name': cv_results}
            metric: Metric to compare ('mean_accuracy', 'mean_f1_macro', etc.)
        """
        print(f"\n--- MODEL COMPARISON ({metric}) ---")
        
        comparison_data = []
        for model_name, results in results_dict.items():
            mean_score = results[metric]
            std_score = results[f"std_{metric.split('_')[1]}"]
            comparison_data.append({
                'model': model_name,
                'mean': mean_score,
                'std': std_score
            })
        
        # Sort by mean score (descending)
        comparison_data.sort(key=lambda x: x['mean'], reverse=True)
        
        print(f"{'Model':<20} {'Mean':<10} {'Std':<10} {'95% CI':<20}")
        print("-" * 60)
        
        for data in comparison_data:
            ci_lower = data['mean'] - 1.96 * data['std']
            ci_upper = data['mean'] + 1.96 * data['std']
            print(f"{data['model']:<20} {data['mean']:.4f}    {data['std']:.4f}    "
                  f"[{ci_lower:.4f}, {ci_upper:.4f}]")
        
        return comparison_data